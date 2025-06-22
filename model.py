import os
import jax
from typing import Any
from jax import lax
import jax.numpy as jnp
import orbax
import orbax.checkpoint as ocp
import flax
from flax.training import train_state
from flax import jax_utils
from flax import traverse_util
import torch
from torch.utils.data import DataLoader
from network import PiMAE
from utils_data import dataset_3d, get_sample, dataset_3d_infer
from utils_imageJ import save_tiff_imagej_compatible
from utils_metrics import ms_ssim_3d
from utils_huggingface_jax import load_jax_vit_base_model
import optax
import pickle
import numpy as np
import random
import tqdm
import functools
import glob
from skimage.io import imread
from skimage.transform import resize
from utils_eval import eval_nrmse
from utils_eval import percentile_norm
from utils_data import min_max_norm
import pprint
import pdb
from jax.experimental import enable_x64

class TrainState(train_state.TrainState):
  batch_stats: Any


def rec_loss(x, rec, mask=None):
    if mask is None:
        mask = jnp.ones_like(x)
    l1_loss = jnp.abs((rec - x))
    l1_loss = (l1_loss * mask).sum() / mask.sum()
    x_norm = (x - x.min()) / (x.max() - x.min())
    rec_norm = (rec - x.min()) / (x.max() - x.min())
    ms_ssim_loss = jnp.mean(1 - ms_ssim_3d(x_norm, rec_norm, win_size=5))
    loss = 0.875 * l1_loss + 0.125 * ms_ssim_loss
    return loss



def TV_Loss(img):
    img = img.reshape([-1, img.shape[-3], img.shape[-2], img.shape[-1]])
    img = img / img.mean()
    batch_size = img.shape[0]
    z, y, x = img.shape[1:4]
    
    def _tensor_size(t):
        return t.shape[-3]*t.shape[-2]*t.shape[-1]
    
    cz = _tensor_size(img[:, 1:, :, :])
    cy = _tensor_size(img[:, :, 1:, :])
    cx = _tensor_size(img[:, :, :, 1:])
    
    hz = lax.pow(jnp.abs(img[:, 1:, :, :] - img[:, :z-1, :, :]), 2.).sum()
    hy = lax.pow(jnp.abs(img[:, :, 1:, :] - img[:, :, :y-1, :]), 2.).sum()
    hx = lax.pow(jnp.abs(img[:, :, :, 1:] - img[:, :, :, :x-1]), 2.).sum()
    
    if cz == 0:
        return (hy/cy + hx/cx) / batch_size
    else:
        return (hz/cz + hy/cy + hx/cx) / batch_size



def center_loss(img):
    if img.shape[-3] == 1:
        vz = jnp.array([0])
    else:
        vz = jnp.linspace(-1, 1, img.shape[-3])
    vy = jnp.linspace(-1, 1, img.shape[-2])
    vx = jnp.linspace(-1, 1, img.shape[-1])
    grid_z, grid_y, grid_x = jnp.meshgrid(vz, vy, vx, indexing='ij')
    grid_z = grid_z[jnp.newaxis, jnp.newaxis, ...]
    grid_y = grid_y[jnp.newaxis, jnp.newaxis, ...]
    grid_x = grid_x[jnp.newaxis, jnp.newaxis, ...]

    img_z = (grid_z * img).sum(axis=(-3, -2, -1)) / img.sum(axis=(-3, -2, -1))
    img_y = (grid_y * img).sum(axis=(-3, -2, -1)) / img.sum(axis=(-3, -2, -1))
    img_x = (grid_x * img).sum(axis=(-3, -2, -1)) / img.sum(axis=(-3, -2, -1))
    img_c = jnp.sqrt(img_z**2 + img_y**2 + img_x**2)
    return img_c.mean()




def compute_metrics(x, state, params, args, rng, train=True):
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    
    # Use remat (gradient checkpointing) to save memory if requested
    apply_fn = state.apply_fn
    if train and args.use_remat:
        apply_fn = jax.remat(state.apply_fn)
    
    result, updates = apply_fn({'params': params, 'batch_stats': state.batch_stats}, x, args, train, rngs={'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, mutable=['batch_stats'])
    
    if train and args.mask_ratio > 0.3:
        rec = rec_loss(result["x_real"], result["rec_real"], result["mask"]).mean()
    else:
        rec = rec_loss(result["x_real"], result["rec_real"]).mean()
    psf_tv = TV_Loss(result["psf"]).mean()
    lp_tv = TV_Loss(result["light_pattern"]).mean()
    ct_loss = center_loss(result["psf"]).mean()
    deconv_tv = TV_Loss(result["deconv"]).mean()

    loss = rec + args.tv_loss * psf_tv + args.psfc_loss * ct_loss + args.lp_tv * lp_tv
    return {"loss": loss, "rec_loss": rec, "psf_tv_loss": psf_tv, "lp_tv_loss": lp_tv, "psf_center_loss": ct_loss, "deconv_tv": deconv_tv}, result, updates


def worker_init_fn(_):
    seed = torch.utils.data.get_worker_info().seed
    np.random.seed(seed % 2**32)
    random.seed(seed)


def pipeline(args, writer):
    # data
    patch_size_z = args.patch_size[0] if len(args.patch_size) > 2 else 1
    train_set = dataset_3d(args.trainset, args.crop_size, minimum_number=args.min_datasize, 
                          patch_size_z=patch_size_z, adapt_z_dimension=args.adapt_z_dimension, 
                          target_z_frames=args.target_z_frames, random_z_sampling=args.random_z_sampling)
    trainloader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True, worker_init_fn=worker_init_fn)

    test_set = dataset_3d(args.testset, args.crop_size, use_gt=args.use_gt, minimum_number=args.batchsize*2, 
                         sampling_rate=args.sampling_rate, patch_size_z=patch_size_z, 
                         adapt_z_dimension=args.adapt_z_dimension, target_z_frames=args.target_z_frames, random_z_sampling=args.random_z_sampling)
    testloader = DataLoader(test_set, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True, worker_init_fn=worker_init_fn)

    x_exp = get_sample(trainloader)
    print("Tensor shape:", x_exp.shape)

    # Set mixed precision policy if requested
    if args.use_mp:
        print("\033[92m", "Using mixed precision training (bfloat16)", "\033[0m")
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        dtype = jnp.bfloat16
    else:
        dtype = jnp.float32

    def net_model():
        image_size = [x_exp.shape[2], args.crop_size[0] * args.rescale[0], args.crop_size[1] * args.rescale[1]]
        return PiMAE(image_size, args.patch_size, args.psf_size, args.lrc)
    
    # init
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # train step
    @functools.partial(jax.pmap, axis_name='Parallelism')
    def apply_model(state, x, rng):
        def loss_fn(params):
            metrics, res, updates = compute_metrics(x, state, params, args, rng, train=True)
            return metrics['loss'], (metrics, res, updates)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)
        metrics, res, updates = aux
        grads = jax.lax.pmean(grads, 'Parallelism')
        loss = jax.lax.pmean(loss, 'Parallelism')
        batch_stats = jax.lax.pmean(updates["batch_stats"], "Parallelism")
        new_state = state.apply_gradients(grads=grads, batch_stats=batch_stats)
        return metrics, res, new_state
    
    @functools.partial(jax.pmap, axis_name='Parallelism')
    def infer_model(state, x, rng):
        metrics, res, updates = compute_metrics(x, state, state.params, args, rng, train=False)
        return metrics, res, state
    
    warmup_steps = 7 * len(trainloader)
    lr_schedule = optax.linear_schedule(init_value=0, end_value=args.lr, transition_steps=warmup_steps)
    # warmup_steps = 0.1 * args.epoch * len(trainloader)
    # decay_steps = (args.epoch - 0.1 * args.epoch) * len(trainloader)
    # lr_schedule = optax.warmup_cosine_decay_schedule(init_value=0, peak_value=args.lr, warmup_steps=warmup_steps, decay_steps=decay_steps, end_value=0.0)
    
    # opt = optax.chain(optax.adam(args.lr), 
    #                   optax.contrib.reduce_on_plateau(factor=0.5, patience=10, rtol=0.0001, atol=0.0, cooldown=0, accumulation_size=100))

    #################### create_train_state ####################
    @jax.jit
    def create_train_state(rng):
        net = net_model()
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        x_init = get_sample(trainloader)
        variables = net.init({"params": rng1, 'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, x_init, args, True)
        
        # print("\033[92m", "frozing encoder blocks", "\033[0m")
        # opt = optax.chain(
        #     optax.clip_by_global_norm(1.0),
        #     optax.adamw(learning_rate=lr_schedule),
        #     )
        # partition_optimizers = {'trainable': opt, 'frozen': optax.set_to_zero()}
        # param_partitions = traverse_util.path_aware_map(
        #     lambda path, v: 'frozen' if 'encoder_block_' in '.'.join(path) else 'trainable', variables['params'])

        # tx = optax.multi_transform(partition_optimizers, param_partitions)
        # if args.accumulation_step is not None:
        #     tx = optax.MultiSteps(tx, every_k_schedule=args.accumulation_step)
        # else:
        #     print("\033[92m", "unfrozing encoder blocks", "\033[0m")
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=lr_schedule),
            )
        
        if args.accumulation_step is not None:
            tx = optax.MultiSteps(tx, every_k_schedule=args.accumulation_step)

        return TrainState.create(
            apply_fn=net.apply, params=variables['params'], batch_stats=variables['batch_stats'], tx=tx)
    
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng)
    variables = state.params
    start_epoch = 0

    #################### Resume ####################
    # checkpoint
    options = ocp.CheckpointManagerOptions(
        max_to_keep=5,
        best_fn=lambda metrics: -metrics['test_rec_loss'] if 'test_rec_loss' in metrics else 0.0,  # negative because we want minimum
        keep_time_interval=None,
        keep_period=None
    )
    
    # Initialize best loss tracking
    best_test_rec_loss = float('inf')
    
    if args.resume_pretrain and not args.resume:
        with open("./ckpt/pretrain_params.pkl", 'rb') as f: 
            pretrain_params = pickle.load(f)

        N = {"Loaded": 0, "Excluded": 0, "Unloaded": 0}
        for key, value in pretrain_params.items():
            # Skip patch embedding and position embedding
            if key in ['patch_embed', 'pos_embed'] or 'patch_embed' in key or 'pos_embed' in key:
                N["Excluded"] += 1
                print(f"Excluded {key} from loading (patch/pos embedding)")
                continue
                
            if key in variables['pt_predictor']['MAE'].keys():
                variables['pt_predictor']['MAE'][key] = value
                N["Loaded"] += 1
            else:
                N["Unloaded"] += 1
        print("Resume pretrain loading summary:", N)
        print(f"\033[93mNote: patch_embed and pos_embed were excluded and will be re-initialized\033[0m")
        state = state.replace(params=variables)
    
    
    if args.resume_s1_path is not None:
        checkpoint_dir = os.path.abspath(os.path.join(args.resume_s1_path, "state"))
        
        if args.not_resume_s1_opt:
            with ocp.CheckpointManager(
                checkpoint_dir,
                options=options,
            ) as mngr:
                if args.resume_s1_iter is not None:
                    step = args.resume_s1_iter
                else:
                    step = mngr.latest_step()
                resume_state = mngr.restore(step, args=ocp.args.StandardRestore())
            state = state.replace(params=resume_state['params'])
        else:
            with ocp.CheckpointManager(
                checkpoint_dir,
                options=options,
            ) as mngr:
                if args.resume_s1_iter is not None:
                    step = args.resume_s1_iter
                else:
                    step = mngr.latest_step()
                state = mngr.restore(step, args=ocp.args.StandardRestore())
        
        state = state.replace(step=0)
        print("\033[94m" + args.resume_s1_path + " %d"%step + "\033[0m")
    
    if args.resume:
        checkpoint_dir = os.path.abspath(os.path.join(args.save_dir, "state"))
        if os.path.exists(checkpoint_dir):
            with ocp.CheckpointManager(
                checkpoint_dir,
                options=options,
            ) as mngr:
                step = mngr.latest_step()
                state = mngr.restore(step, args=ocp.args.StandardRestore())
            start_epoch = step // len(trainloader)
            print("\033[34m", "Resume from", checkpoint_dir, "at epoch", start_epoch, "\033[0m")
    
    #################### Training ####################
    state = jax_utils.replicate(state)
    
    for epoch in range(start_epoch, args.epoch):
        # train
        def add_image_writer(name, img, df='CNHW'):
            if img.max() > img.min():
                img = percentile_norm(img)
            writer.add_image(name, np.array(img), epoch, dataformats=df)
        
        pbar = tqdm.tqdm(trainloader)
        metrics_train = {'loss': 0.0, "rec_loss": 0.0, "psf_tv_loss": 0.0, "lp_tv_loss": 0.0, "psf_center_loss": 0.0, "deconv_tv": 0.0}
        
        for data in pbar:
            rng_new, rng = jax.random.split(rng, 2)
            x = jnp.array(data['img'])
            x = jax.lax.stop_gradient(x)
            x = x.reshape([jax.local_device_count(), -1, *x.shape[1:]])
            # jax.profiler.start_trace('/tmp/timeline')
            metrics, res, state = apply_model(state, x, jax_utils.replicate(rng_new))
            current_lr = np.array(lr_schedule(state.step)).mean()
            # jax.profiler.stop_trace()
            pbar.set_postfix(rec="%.2e"%np.array(metrics['loss']).mean(), 
                             R="%.2f%%"%np.array(100*res['rec_real'].mean()/res['x_real'].mean()), 
                             lr="%.2e"%current_lr.mean(), 
                             mask="%.2f"%np.array(res['mask']).mean(),
                             lp_tv="%.2e"%np.array(metrics['lp_tv_loss']).mean())
            metrics_train = {k: v + metrics[k].mean() for k, v in metrics_train.items()}      
        metrics_train = {k:np.asarray(v / len(trainloader)) for k, v in metrics_train.items()}
        
        # tensorboard 
        res = jax_utils.unreplicate(res)
        writer.add_scalar('train/loss', metrics_train['loss'], epoch+1)
        writer.add_scalar('train/rec_loss', metrics_train['rec_loss'], epoch+1)
        writer.add_scalar('train/psf_tv_loss', metrics_train['psf_tv_loss'], epoch+1)
        writer.add_scalar('train/lp_tv_loss', metrics_train['lp_tv_loss'], epoch+1)
        writer.add_scalar('train/psf_center_loss', metrics_train['psf_center_loss'], epoch+1)
        writer.add_scalar('train/deconv_tv', metrics_train['deconv_tv'], epoch+1)
        writer.add_scalar('train/lr', current_lr, epoch+1)

        add_image_writer('train/1_x', res['x_up'][0][0:1], 'CNHW')
        add_image_writer('train/2_deconv', res['deconv'][0][0:1], 'CNHW')
        add_image_writer('train/3_psf', res['psf'][0], 'CNHW')
        add_image_writer('train/4_reconstruction', res['rec_up'][0][0:1], 'CNHW')
        add_image_writer('train/5_light_pattern', res['light_pattern'][0][0:1], 'CNHW')
        add_image_writer('train/6_background', res['background'][0][0:1], 'CNHW')
        
        # save
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "psf.tif"), np.array(res['psf'][0][0]), "ZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_x.tif"), np.array(res['x_up']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_mask_real.tif"), np.array(res['mask']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_rec.tif"), np.array(res['rec_up']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_deconv.tif"), np.array(res['deconv']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_light_pattern.tif"), np.array(res['light_pattern']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "train_background.tif"), np.array(res['background']), "TCZYX")

        # snapshot - removed from here, will be done after test evaluation
        
        # test
        if not args.use_gt:
            metrics_eval = {"rec_loss": 0.0}
            for data in testloader:
                rng_new, rng = jax.random.split(rng, 2)
                x = jnp.array(data['img'])
                x = jax.lax.stop_gradient(x)
                x = x.reshape([jax.local_device_count(), -1, *x.shape[1:]])
                metrics, res, _ = infer_model(state, x, jax_utils.replicate(rng_new))
                metrics_eval = {k: v + metrics[k].mean().astype(np.float32) for k, v in metrics_eval.items()}      
            metrics_eval = {k: np.asarray(v / len(testloader)) for k, v in metrics_eval.items()}

            writer.add_scalar('test/rec_loss', metrics_eval['rec_loss'], epoch+1)
            print("epoch=%d/%d"%(epoch+1, args.epoch), "rec=%.2e"%np.array(metrics_eval['rec_loss']))
        else:
            folder = os.path.dirname(args.testset)
            psf = imread(os.path.join(folder, "..", "psf.tif")).astype(np.float32)
            metrics_eval = {"rec_loss": 0.0}

            emitter_nrmse_list = []
            lp_nrmse_list = []
            psf_nrmse_list = []
            for data in testloader:
                rng_new, rng = jax.random.split(rng, 2)
                x = jnp.array(data['img'])
                x = jax.lax.stop_gradient(x)
                x = x.reshape([jax.local_device_count(), -1, *x.shape[1:]])
                metrics, res, _ = infer_model(state, x, jax_utils.replicate(rng_new))
                metrics_eval = {k: v + metrics[k].mean().astype(np.float32) for k, v in metrics_eval.items()}
                
                emitter_gt = data['emitter_gt'].reshape([jax.local_device_count(), -1, *data['emitter_gt'].shape[1:]])
                emitter_nrmse = eval_nrmse(res['deconv'], emitter_gt)
                emitter_nrmse_list.append(emitter_nrmse)
                lp_gt = data['lp_gt'].reshape([jax.local_device_count(), -1, *data['lp_gt'].shape[1:]])
                lp_nrmse = eval_nrmse(res['light_pattern'], lp_gt)
                lp_nrmse_list.append(lp_nrmse)
                
                psf_f = np.squeeze(jax_utils.unreplicate(res['psf']))
                psf_nrmse = eval_nrmse(psf_f, psf)
                psf_nrmse_list.append(psf_nrmse)


            metrics_eval = {k: np.asarray(v / len(testloader)) for k, v in metrics_eval.items()}
            writer.add_scalar('test/rec_loss', metrics_eval['rec_loss'], epoch+1)
            writer.add_scalar('test/emitter_loss', np.mean(emitter_nrmse_list), epoch+1)
            writer.add_scalar('test/light_pattern_loss', np.mean(lp_nrmse_list), epoch+1)
            writer.add_scalar('test/psf_loss', np.mean(psf_nrmse_list), epoch+1)
            print("epoch=%d/%d"%(epoch+1, args.epoch), "rec=%.2e"%np.array(metrics_eval['rec_loss']), "emitter=%.2e"%np.mean(emitter_nrmse_list), "light_pattern=%.2e"%np.mean(lp_nrmse_list), "psf=%.2e"%np.mean(psf_nrmse_list))
            
        # tensorboard 
        res = jax_utils.unreplicate(res)
        add_image_writer('test/1_x', res['x_up'][0][0:1], 'CNHW')
        add_image_writer('test/2_deconv', res['deconv'][0][0:1], 'CNHW')
        add_image_writer('test/4_reconstruction', res['rec_up'][0][0:1], 'CNHW')
        add_image_writer('test/5_light_pattern', res['light_pattern'][0][0:1], 'CNHW')
        add_image_writer('test/7_background', res['background'][0][0:1], 'CNHW')
        if args.use_gt:
            add_image_writer('test/3_deconv_gt', emitter_gt[0, 0, 0:1], 'CNHW')
            add_image_writer('test/6_light_pattern_gt', lp_gt[0, 0, 0:1], 'CNHW')
        
        # save
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "test_x.tif"), np.array(res['x_up']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "test_rec.tif"), np.array(res['rec_up']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "test_deconv.tif"), np.array(res['deconv']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "test_light_pattern.tif"), np.array(res['light_pattern']), "TCZYX")
        save_tiff_imagej_compatible(os.path.join(args.save_dir, "test_background.tif"), np.array(res['background']), "TCZYX")
        
        # Save checkpoint with test metrics for best checkpoint selection
        os.makedirs(os.path.join(args.save_dir, "state"), exist_ok=True)
        state_to_save = jax_utils.unreplicate(state)
        checkpoint_dir = os.path.abspath(os.path.join(args.save_dir, "state"))
        
        # Prepare checkpoint metrics
        checkpoint_metrics = {
            'test_rec_loss': float(metrics_eval['rec_loss']),
            'epoch': epoch,
            'step': int(state_to_save.step)
        }
        
        # Update best loss tracking
        current_test_rec_loss = float(metrics_eval['rec_loss'])
        if current_test_rec_loss < best_test_rec_loss:
            best_test_rec_loss = current_test_rec_loss
            print(f"\033[92m*** New best checkpoint saved! Test rec loss: {current_test_rec_loss:.2e} ***\033[0m")
        else:
            print(f"Current test rec loss: {current_test_rec_loss:.2e}, Best: {best_test_rec_loss:.2e}")
            
        with ocp.CheckpointManager(
            checkpoint_dir,
            options=options,
        ) as mngr:
            mngr.save(
                state_to_save.step, 
                args=ocp.args.StandardSave(state_to_save),
                metrics=checkpoint_metrics
            )
            mngr.wait_until_finished()
            
    return state


def pipeline_supervised(args):
    # data
    train_set = dataset_3d(args.trainset, args.crop_size, minimum_number=args.min_datasize,
                          adapt_z_dimension=args.adapt_z_dimension, target_z_frames=args.target_z_frames,
                          random_z_sampling=args.random_z_sampling)
    trainloader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True)

    test_set = dataset_3d(args.testset, args.crop_size, use_gt=args.use_gt, sampling_rate=0.1,
                         adapt_z_dimension=args.adapt_z_dimension, target_z_frames=args.target_z_frames,
                         random_z_sampling=args.random_z_sampling)
    testloader = DataLoader(test_set, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True)

    x_exp = get_sample(trainloader)
    print("Tensor shape:", x_exp.shape)

def pipeline_infer(args):
    def net_model():
        image_size = [args.num_p, args.crop_size[0] * args.rescale[0], args.crop_size[1] * args.rescale[1]]
        return PiMAE(image_size, args.patch_size, args.psf_size, args.lrc)

    # train_state
    def create_train_state(rng):
        net = net_model()
        x_init = jax.random.normal(rng, (1, 1, args.num_p, *args.crop_size))
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        variables = net.init({"params": rng1, 'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, x_init, args, True)
        return TrainState.create(
            apply_fn=net.apply, params=variables['params'], batch_stats=variables['batch_stats'], tx=optax.adamw(0.0))
    
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng)
    
    print(args.resume_path)
    options = ocp.CheckpointManagerOptions()
    checkpoint_dir = os.path.abspath(os.path.join(args.resume_path, "state"))
    if os.path.exists(checkpoint_dir):
        with ocp.CheckpointManager(
            checkpoint_dir,
            options=options,
        ) as mngr:
            if args.resume_s1_iter is not None:
                step = args.resume_s1_iter
            else:
                step = mngr.latest_step()
            resume_state = mngr.restore(step, args=ocp.args.StandardRestore())
        state = state.replace(params=resume_state['params'])
        print("\033[34m", "Resume from", checkpoint_dir, "\033[0m")
    else:
        raise ValueError("No checkpoint found")
    

    if '*.tif' or '*.png' or '*.jpg' in args.data_dir:
        file_names = glob.glob(args.data_dir)
        print("Images num:", len(file_names))
        
    else:
        file_names = [args.data_dir]

    @jax.jit
    def eval_model(x, rng):
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        result, _ = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, x, args, False, rngs={'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, mutable=['batch_stats'])
        return result
    
    file_names.sort()
    # ###########################################
    # file_names = file_names[:3]

    nrmse_list = []
    for i, file in enumerate(file_names):
        if '*.tif' or '*.png' or '*.jpg' in args.data_dir:
            relative_path = os.path.relpath(file, os.path.commonpath([args.data_dir.replace('*', ''), file]))
            relative_path_star = os.path.relpath(args.data_dir, os.path.commonpath([args.data_dir.replace('*', ''), file]))
            star_indices = [i for i, part in enumerate(relative_path_star.split(os.sep)) if "*" in part]
            relative_parts = relative_path.split(os.sep)
            selected_parts = [relative_parts[i] for i in star_indices]
            target_dir = os.path.join(args.save_dir, *selected_parts[:-1])
        else:
            target_dir = args.save_dir

        img = imread(file).astype(np.float32) # CZXY
        if len(img.shape) == 2:
            img = img[np.newaxis, np.newaxis, :, :]
        elif len(img.shape) == 3:
            img = img[np.newaxis, :, :, :]
        print(file, img.shape)
        
        patch_size_z = args.patch_size[0] if len(args.patch_size) > 2 else 1
        test_set = dataset_3d_infer(img, args.crop_size, args.rescale, patch_size_z=patch_size_z,
                                   adapt_z_dimension=args.adapt_z_dimension, target_z_frames=args.target_z_frames,
                                   random_z_sampling=args.random_z_sampling)
        test_dataloader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False, num_workers=0, pin_memory=False, drop_last=False)
        pbar = tqdm.tqdm(test_dataloader)
        for x, patch_index in pbar:
            x = jnp.array(x)
            result = eval_model(x, rng)
            result = {key: np.array(value) for key, value in result.items()}
            patch_index = [np.array(i) for i in patch_index]
            test_set.assemble_patch(result, patch_index)
        
        
        image_in = test_set.image_in
        image_in = jax.image.resize(image_in, shape=(image_in.shape[0], image_in.shape[1], image_in.shape[2]*args.rescale[0], image_in.shape[3]*args.rescale[1]), method='linear')
        image_in = image_in.astype(np.float32)
        image_in = min_max_norm(image_in)

        # ssl_sim_deconv = percentile_norm(test_set.deconv)
        # reconstruciton = percentile_norm(test_set.reconstruction)
        # lightfield = percentile_norm(test_set.lightfield)
        # background = percentile_norm(test_set.background)
        
        # p_low = np.percentile(img, p_low)
        # wf = image_in.mean(axis=0, keepdims=True)
        # mask = wf > np.percentile(wf, 0.2)
        # closed operation
        # mask = ndimage.binary_closing(mask, iterations=2)
        # lightfield = lightfield * mask
        
        # res_list = np.concatenate([image_in, reconstruciton, ssl_sim_deconv, lightfield, background], axis=1)
        save_emitters_dir = os.path.join(target_dir, "test")
        os.makedirs(save_emitters_dir, exist_ok=True)
        save_meta_dir = os.path.join(target_dir, "test_meta")
        os.makedirs(save_meta_dir, exist_ok=True)
        
        file_name, _ = os.path.splitext(os.path.split(file)[-1])

        save_tiff_imagej_compatible(os.path.join(save_emitters_dir, file_name + ".tif"), test_set.deconv.astype(np.float32).squeeze(), "YX")
        save_tiff_imagej_compatible(os.path.join(save_meta_dir, file_name + "_lp.tif"), test_set.light_pattern.astype(np.float32).squeeze(), "ZYX")
        save_tiff_imagej_compatible(os.path.join(save_meta_dir, file_name + "_bg.tif"), test_set.background.astype(np.float32).squeeze(), "YX")
        
        if i == 0:
            save_tiff_imagej_compatible(os.path.join(args.save_dir, "psf.tif"), result["psf"][0,0, ...].astype(np.float32), "ZYX")