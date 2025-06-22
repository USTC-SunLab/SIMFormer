# Simplified model for SIMFormer distillation
# Focuses only on emitter reconstruction without physics simulation

import os
import jax
from typing import Any
import jax.numpy as jnp
import orbax
import orbax.checkpoint as ocp
import flax
from flax.training import train_state
from flax import jax_utils
from flax import traverse_util
from torch.utils.data import DataLoader
from network_distillation import PiMAE_Distillation
from utils_data import dataset_3d, get_sample, dataset_3d_infer
from utils_imageJ import save_tiff_imagej_compatible
from utils_metrics import ms_ssim_3d
import optax
import pickle
import numpy as np
import tqdm
import functools
import glob
from skimage.io import imread
from utils_eval_biosr import eval_nrmse
from utils_eval_biosr import percentile_norm
from utils_data import min_max_norm

class TrainState(train_state.TrainState):
  batch_stats: Any


def rec_loss(x, rec, mask=None):
    """Reconstruction loss with L1 and MS-SSIM components"""
    if mask is None:
        mask = jnp.ones_like(x)
    l1_loss = jnp.abs((rec - x))
    l1_loss = (l1_loss * mask).sum() / mask.sum()
    
    # Normalize for MS-SSIM
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    rec_norm = (rec - x.min()) / (x.max() - x.min() + 1e-8)
    ms_ssim_loss = jnp.mean(1 - ms_ssim_3d(x_norm, rec_norm, win_size=5))
    
    loss = 0.875 * l1_loss + 0.125 * ms_ssim_loss
    return loss


def emitter_lasso_loss(x):
    """L1 regularization on emitter output"""
    return jnp.abs(x).mean()


def compute_metrics(data, state, params, args, rng, train=True):
    """Compute distillation metrics - simplified without physics"""
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    
    # Forward pass
    result, updates = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats}, 
        data['img'], args, train, 
        rngs={'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, 
        mutable=['batch_stats']
    )
    
    # Distillation loss - compare with SIMFormer output
    emitter_rec = rec_loss(result["deconv"], data["emitter_gt"]).mean()
    
    # Optional L1 regularization
    emitter_lasso = emitter_lasso_loss(result["deconv"]).mean()
    
    # Total loss
    loss = emitter_rec + args.emitter_lasso * emitter_lasso
    
    metrics = {
        "loss": loss, 
        "emitter_rec_loss": emitter_rec, 
        "emitter_lasso": emitter_lasso
    }
    
    return metrics, result, updates


def pipeline(args, writer):
    """Main training pipeline for distillation"""
    
    # Create datasets with SIMFormer outputs as targets
    train_set = dataset_3d(
        args.trainset, 
        args.crop_size, 
        minimum_number=args.min_datasize,
        simformer_infer_save_dir=args.simformer_infer_save_dir  # Key parameter
    )
    trainloader = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True)

    test_set = dataset_3d(
        args.testset, 
        args.crop_size, 
        minimum_number=args.batchsize, 
        sampling_rate=1,
        simformer_infer_save_dir=args.simformer_infer_save_dir
    )
    testloader = DataLoader(test_set, batch_size=args.batchsize, shuffle=True, num_workers=0, drop_last=True)

    print(f"Training set size: {len(train_set)}, Test set size: {len(test_set)}")
    
    # Get sample for model initialization
    x_exp = get_sample(trainloader)
    print("Input tensor shape:", x_exp.shape)

    def net_model():
        image_size = [x_exp.shape[2], args.crop_size[0] * args.rescale[0], args.crop_size[1] * args.rescale[1]]
        return PiMAE_Distillation(image_size, args.patch_size)
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Define training step
    @functools.partial(jax.pmap, axis_name='Parallelism')
    def apply_model(state, data, rng):
        def loss_fn(params):
            metrics, res, updates = compute_metrics(data, state, params, args, rng, train=True)
            return metrics['loss'], (metrics, res, updates)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)
        metrics, res, updates = aux
        grads = jax.lax.pmean(grads, 'Parallelism')
        loss = jax.lax.pmean(loss, 'Parallelism')
        batch_stats = jax.lax.pmean(updates["batch_stats"], "Parallelism")
        new_state = state.apply_gradients(grads=grads, batch_stats=batch_stats)
        return metrics, res, new_state
    
    # Define inference step
    @functools.partial(jax.pmap, axis_name='Parallelism')
    def infer_model(state, data, rng):
        metrics, res, updates = compute_metrics(data, state, state.params, args, rng, train=False)
        return metrics, res, state
    
    # Learning rate schedule
    warmup_steps = 7 * len(trainloader)
    lr_schedule = optax.linear_schedule(init_value=0, end_value=args.lr, transition_steps=warmup_steps)
    
    # Create training state
    @jax.jit
    def create_train_state(rng):
        net = net_model()
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        params = net.init({'params': rng1, 'dropout': rng2, 'random_masking': rng3}, 
                         get_sample(trainloader), args, True)
        
        if args.resume_pretrain:
            print("Loading pre-trained parameters...")
            with open('pretrain_params.pkl', 'rb') as f:
                pretrain_params = pickle.load(f)
            params = flax.core.frozen_dict.unfreeze(params)
            params['params']['pt_predictor']['MAE'] = pretrain_params
            params = flax.core.frozen_dict.freeze(params)
        
        opt = optax.adam(lr_schedule)
        return TrainState.create(
            apply_fn=net.apply, params=params['params'], tx=opt, batch_stats=params.get('batch_stats', {})
        )
    
    state = create_train_state(init_rng)
    state = jax_utils.replicate(state, devices=jax.devices())
    
    # Setup checkpoint manager
    options = ocp.CheckpointManagerOptions(max_to_keep=10, save_interval_steps=100)
    mngr = ocp.CheckpointManager(
        os.path.join(args.save_dir, 'state'), 
        options=options
    )
    
    # Resume logic
    if args.resume_s1_path is not None:
        print(f"Resuming from stage 1: {args.resume_s1_path}")
        
        s1_checkpoint_dir = os.path.join(args.resume_s1_path, 'state')
        s1_mngr = ocp.CheckpointManager(s1_checkpoint_dir)
        
        if args.resume_s1_iter is not None:
            restored = s1_mngr.restore(int(args.resume_s1_iter))
        else:
            restored = s1_mngr.restore(s1_mngr.latest_step())
        
        state = jax_utils.unreplicate(state)
        state = state.replace(params=restored['params'], batch_stats=restored['batch_stats'])
        
        if not args.not_resume_s1_opt:
            state = state.replace(opt_state=restored['opt_state'], step=restored['step'])
        
        state = jax_utils.replicate(state, devices=jax.devices())
        print(f"Resumed successfully from iteration {restored.get('step', 0)}")
    
    elif args.resume:
        print("Resuming from checkpoint...")
        restored = mngr.restore(mngr.latest_step())
        state = jax_utils.unreplicate(state)
        state = state.replace(
            params=restored['params'], 
            batch_stats=restored['batch_stats'],
            opt_state=restored['opt_state'],
            step=restored['step']
        )
        state = jax_utils.replicate(state, devices=jax.devices())
        print(f"Resumed from iteration {restored['step']}")
    
    # Training loop
    rng = jax.random.PRNGKey(0)
    tr_metrics_cpu = []
    batch_x = next(iter(trainloader))
    img_gt = batch_x["emitter_gt"][0, 0, 0]
    
    # Calculate training iterations
    total_epochs = args.epoch
    total_iter = total_epochs * len(trainloader)
    
    # Main training loop
    pbar = tqdm.tqdm(range(total_iter))
    for train_iter in pbar:
        epoch = train_iter // len(trainloader)
        iter_in_epoch = train_iter % len(trainloader)
        
        if iter_in_epoch == 0:
            trainloader_iter = iter(trainloader)
        
        # Get batch
        batch = next(trainloader_iter)
        batch = jax.tree_map(lambda x: x.numpy(), batch)
        batch = jax_utils.replicate(batch, devices=jax.devices())
        
        # Train step
        rng, step_rng = jax.random.split(rng)
        rngs = jax.random.split(step_rng, jax.device_count())
        tr_metrics, res, state = apply_model(state, batch, rngs)
        
        # Log metrics
        tr_metrics_cpu.append(jax.tree_map(lambda x: np.mean(jax_utils.unreplicate(x)), tr_metrics))
        
        if train_iter % 10 == 0:
            avg_loss = np.mean([m['loss'] for m in tr_metrics_cpu[-10:]])
            avg_emitter_rec = np.mean([m['emitter_rec_loss'] for m in tr_metrics_cpu[-10:]])
            pbar.set_description(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Emitter Rec: {avg_emitter_rec:.4f}")
        
        # Tensorboard logging
        if train_iter % 50 == 0:
            for key, value in tr_metrics_cpu[-1].items():
                writer.add_scalar(f'train/{key}', value, train_iter)
        
        # Validation and checkpointing
        if train_iter % 100 == 0:
            # Run validation
            val_metrics_list = []
            for val_batch in testloader:
                val_batch = jax.tree_map(lambda x: x.numpy(), val_batch)
                val_batch = jax_utils.replicate(val_batch, devices=jax.devices())
                rng, val_rng = jax.random.split(rng)
                val_rngs = jax.random.split(val_rng, jax.device_count())
                val_metrics, val_res, _ = infer_model(state, val_batch, val_rngs)
                val_metrics_list.append(jax.tree_map(lambda x: np.mean(jax_utils.unreplicate(x)), val_metrics))
            
            # Log validation metrics
            for key in val_metrics_list[0].keys():
                avg_val = np.mean([m[key] for m in val_metrics_list])
                writer.add_scalar(f'val/{key}', avg_val, train_iter)
            
            # Save checkpoint
            state_cpu = jax_utils.unreplicate(state)
            mngr.save(
                train_iter,
                args={
                    'params': state_cpu.params,
                    'batch_stats': state_cpu.batch_stats,
                    'opt_state': state_cpu.opt_state,
                    'step': train_iter
                }
            )
            
            # Visualize results
            res_cpu = jax.tree_map(lambda x: jax_utils.unreplicate(x)[0], res)
            deconv = res_cpu["deconv"][0, 0]
            writer.add_image('deconv', deconv[None, ...], train_iter)
            writer.add_image('emitter_gt', img_gt[None, ...], train_iter)
    
    print("Training completed!")
    return state


def pipeline_infer_distillation(args):
    """Inference pipeline for distillation models"""
    
    def net_model():
        # Determine image size from input
        image_size = [9, args.crop_size[0] * args.rescale[0], args.crop_size[1] * args.rescale[1]]
        return PiMAE_Distillation(image_size, args.patch_size)

    # Create training state for loading
    def create_train_state(rng):
        net = net_model()
        x_init = jax.random.normal(rng, (1, 1, 9, *args.crop_size))
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        variables = net.init({"params": rng1, 'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, 
                           x_init, args, False)
        return TrainState.create(
            apply_fn=net.apply, params=variables['params'], 
            batch_stats=variables.get('batch_stats', {}), tx=optax.adamw(0.0))
    
    # Initialize state
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.resume_path}")
    options = ocp.CheckpointManagerOptions()
    checkpoint_dir = os.path.abspath(os.path.join(args.resume_path, "state"))
    
    if os.path.exists(checkpoint_dir):
        with ocp.CheckpointManager(checkpoint_dir, options=options) as mngr:
            if args.resume_iter is not None:
                step = args.resume_iter
            else:
                step = mngr.latest_step()
            print(f"Loading iteration: {step}")
            resume_state = mngr.restore(step, args=ocp.args.StandardRestore())
        
        state = state.replace(params=resume_state['params'])
        if 'batch_stats' in resume_state:
            state = state.replace(batch_stats=resume_state['batch_stats'])
        print(f"\033[34mSuccessfully loaded checkpoint from iteration {step}\033[0m")
    else:
        raise ValueError(f"No checkpoint found at {checkpoint_dir}")
    
    # Get file list
    if any(pattern in args.data_dir for pattern in ['*.tif', '*.png', '*.jpg']):
        file_names = glob.glob(args.data_dir)
        print(f"Found {len(file_names)} images")
    else:
        file_names = [args.data_dir]
    
    # JIT compile evaluation function
    @jax.jit
    def eval_model(x, rng):
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        result, _ = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats}, 
            x, args, False, 
            rngs={'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, 
            mutable=['batch_stats']
        )
        return result
    
    file_names.sort()
    
    # Process each file
    for i, file in enumerate(file_names):
        print(f"\nProcessing [{i+1}/{len(file_names)}]: {file}")
        
        # Determine output directory structure
        if any(pattern in args.data_dir for pattern in ['*.tif', '*.png', '*.jpg']):
            relative_path = os.path.relpath(file, os.path.commonpath([args.data_dir.replace('*', ''), file]))
            relative_path_star = os.path.relpath(args.data_dir, os.path.commonpath([args.data_dir.replace('*', ''), file]))
            star_indices = [i for i, part in enumerate(relative_path_star.split(os.sep)) if "*" in part]
            relative_parts = relative_path.split(os.sep)
            selected_parts = [relative_parts[i] for i in star_indices]
            target_dir = os.path.join(args.save_dir, *selected_parts[:-1])
        else:
            target_dir = args.save_dir
        
        # Read image
        img = imread(file).astype(np.float32)
        if len(img.shape) == 2:
            img = img[np.newaxis, np.newaxis, :, :]
        elif len(img.shape) == 3:
            img = img[np.newaxis, :, :, :]
        print(f"Image shape: {img.shape}")
        
        # Create dataset for tiling if needed
        patch_size_z = args.patch_size[0] if len(args.patch_size) > 2 else 1
        test_set = dataset_3d_infer(img, args.crop_size, args.rescale, patch_size_z=patch_size_z)
        test_dataloader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False, 
                                   num_workers=0, pin_memory=False, drop_last=False)
        
        # Process patches
        pbar = tqdm.tqdm(test_dataloader, desc="Processing patches")
        for x, patch_index in pbar:
            x = jnp.array(x)
            result = eval_model(x, rng)
            result = {key: np.array(value) for key, value in result.items()}
            patch_index = [np.array(i) for i in patch_index]
            test_set.assemble_patch(result, patch_index)
        
        # Save results
        save_dir = os.path.join(target_dir, "emitter")
        os.makedirs(save_dir, exist_ok=True)
        
        file_name, _ = os.path.splitext(os.path.basename(file))
        output_path = os.path.join(save_dir, f"{file_name}.{args.save_format}")
        
        # Get assembled emitter prediction
        emitter = test_set.deconv.astype(np.float32).squeeze()
        
        if args.save_format == 'tif':
            save_tiff_imagej_compatible(output_path, emitter, "YX" if emitter.ndim == 2 else "ZYX")
        else:
            # For PNG, normalize to 0-255
            emitter_norm = percentile_norm(emitter)
            emitter_uint8 = (emitter_norm * 255).astype(np.uint8)
            from skimage.io import imsave
            imsave(output_path, emitter_uint8)
        
        print(f"Saved: {output_path}")
    
    print(f"\nInference completed! Results saved to: {args.save_dir}")