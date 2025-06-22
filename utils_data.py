import numpy as np
import os
import torch
from skimage.io import imread
from skimage.transform import resize
import jax.numpy as jnp
import tqdm
import glob
import pdb



def adapt_pattern_dimension_frames(data, target_pattern_frames=9, random=False):
    """Adapt pattern dimension to match model's expected input.
    
    SIMFormer expects 9-frame input (3 angles × 3 phases), but datasets
    may have varying pattern dimensions. This function adapts the pattern dimension
    to ensure model compatibility.
    
    Args:
        data: Input with shape (c, patterns, y, x) where patterns may vary
        target_pattern_frames: Target number of patterns (default: 9 for standard SIM)
        random: If True, randomly sample patterns; if False, uniformly sample
    
    Returns:
        Data with shape (c, target_pattern_frames, y, x)
    """
    assert len(data.shape) == 4, "Input data must have 4 dimensions (c, patterns, y, x)"
    c, patterns, y, x = data.shape
    
    if patterns <= target_pattern_frames:
        return data
    
    if random:
        indices = np.random.choice(patterns, size=target_pattern_frames, replace=False)
        indices.sort()
    else:
        step = (patterns - 1) // (target_pattern_frames - 1) if target_pattern_frames > 1 else 1
        if step * (target_pattern_frames - 1) + 1 <= patterns:
            indices = [i * step for i in range(target_pattern_frames)]
        else:
            indices = np.linspace(0, patterns-1, target_pattern_frames, dtype=int)
    
    extracted_data = data[:, indices, :, :]
    return extracted_data

def multi_channel_random_crop(data, crop_size):
    assert len(data.shape) == 4, "Input data must have 4 dimensions (c, patterns, y, x)"
    assert all(crop_size[i] <= data.shape[i+2] for i in range(2)), "Crop size must be smaller or equal to data dimensions (patterns, y, x)"
    
    c, patterns, y, x = data.shape
    crop_y, crop_x = crop_size

    start_y = np.random.randint(0, y - crop_y + 1)
    start_x = np.random.randint(0, x - crop_x + 1)
    cropped_data = data[:, :, start_y:start_y + crop_y, start_x:start_x + crop_x]
    
    return cropped_data


def multi_image_multi_channel_random_crop(data_list, crop_size):
    for data in data_list:
        assert len(data.shape) == 4, "Input data must have 4 dimensions (c, patterns, y, x)"
    
    c, patterns, y, x = data_list[0].shape
    crop_y, crop_x = crop_size

    start_y = np.random.randint(0, y - crop_y + 1)
    start_x = np.random.randint(0, x - crop_x + 1)
    cropped_data_list = []
    for data in data_list:
        zoom_level_y = data.shape[2] / y
        zoom_level_x = data.shape[3] / x
        start_y_zoomed = int(start_y * zoom_level_y)
        start_x_zoomed = int(start_x * zoom_level_x)
        crop_y_zoomed = int(crop_y * zoom_level_y)
        crop_x_zoomed = int(crop_x * zoom_level_x)
        cropped_data = data[:, :, start_y_zoomed:start_y_zoomed + crop_y_zoomed, start_x_zoomed:start_x_zoomed + crop_x_zoomed]
        cropped_data_list.append(cropped_data)
    return cropped_data_list



class dataset_3d(torch.utils.data.Dataset):
    """Dataset for 3D data with Z-dimension adaptation.
    
    Args:
        dataset: Path pattern for dataset files
        crop_size: Spatial crop size (y, x)
        minimum_number: Minimum dataset size (will repeat if needed)
        use_gt: Whether to load ground truth
        sampling_rate: Fraction of files to use
        patch_size_z: Z-dimension patch size
        adapt_pattern_dimension: Enable pattern dimension adaptation for model compatibility
        target_pattern_frames: Target pattern dimension size (default: 9 for standard SIM)
        random_pattern_sampling: Use random sampling instead of uniform for pattern dimension
    """
    def __init__(self, dataset, crop_size, minimum_number=None, use_gt=False, sampling_rate=1, patch_size_z=1, adapt_pattern_dimension=False, target_pattern_frames=9, random_pattern_sampling=False):
        super().__init__()
        files_path = glob.glob(dataset)
        files_path = np.random.permutation(files_path)
        files_path = files_path[:int(len(files_path)*sampling_rate)]
        self.crop_size = crop_size
        self.files_pool = []
        self.use_gt = use_gt
        self.patch_size_z = patch_size_z
        self.adapt_pattern_dimension = adapt_pattern_dimension
        self.target_pattern_frames = target_pattern_frames
        self.random_pattern_sampling = random_pattern_sampling

        
        for file in tqdm.tqdm(files_path):
            if not use_gt:
                img = min_max_norm(imread(file).astype(np.float32))
                if len(img.shape) == 2:
                    img = img[np.newaxis, np.newaxis, :, :]
                elif len(img.shape) == 3:
                    img = img[np.newaxis, :, :, :]
                
                self.files_pool.append([img])
            else:
                folder = os.path.dirname(file)
                filename = os.path.basename(file)
                filename, ext = os.path.splitext(filename)
                img = min_max_norm(imread(file).astype(np.float32))
                if len(img.shape) == 2:
                    img = img[np.newaxis, np.newaxis, :, :]
                elif len(img.shape) == 3:
                    img = img[np.newaxis, :, :, :]
                
                
                emitter_gt = imread(os.path.join(folder+"_gt", filename+ext)).astype(np.float32)[np.newaxis, np.newaxis, :, :]
                lp_gt = imread(os.path.join(folder+"_gt", filename+"_lp"+ext)).astype(np.float32)[np.newaxis, :, :, :]
                emitter_gt = resize(emitter_gt, [emitter_gt.shape[0], emitter_gt.shape[1], emitter_gt.shape[2]//2, emitter_gt.shape[3]//2])
                lp_gt = resize(lp_gt, [lp_gt.shape[0], lp_gt.shape[1], lp_gt.shape[2]//2, lp_gt.shape[3]//2])
                self.files_pool.append([img, emitter_gt, lp_gt])


        assert len(self.files_pool) > 0
        if minimum_number is None:
            minimum_number = len(self.files_pool)
        
        while len(self.files_pool) < minimum_number:
            self.files_pool.extend(self.files_pool)
        self.files_pool = self.files_pool[:minimum_number]
        print(dataset, len(self.files_pool))
    
        
    
    def __getitem__(self, idx):
        files = self.files_pool[idx]
        
        processed_files = []
        for file_item in files:
            if self.adapt_pattern_dimension and file_item.shape[1] > self.target_pattern_frames:
                processed_item = adapt_pattern_dimension_frames(file_item, self.target_pattern_frames, random=self.random_pattern_sampling)
                processed_files.append(processed_item)
            else:
                processed_files.append(file_item)
        
        if self.use_gt and len(processed_files) > 2:
            img, emitter_gt, lp_gt = processed_files
            processed_files = [img, emitter_gt, lp_gt]
        else:
            processed_files = processed_files
        
        processed_files = multi_image_multi_channel_random_crop(processed_files, self.crop_size)
        if self.use_gt:
            img, emitter_gt, lp_gt = processed_files
            return {"img": img, "emitter_gt": emitter_gt, "lp_gt": lp_gt}
        else:
            return {"img": processed_files[0]}

    def __len__(self):
        return len(self.files_pool)
    

class dataset_3d_supervised(torch.utils.data.Dataset):
    # Supervised dataset for 3D data
    def __init__(self, dataset, crop_size, minimum_number=None, use_gt=False, sampling_rate=1, adapt_pattern_dimension=False, target_pattern_frames=9, random_pattern_sampling=False):
        super().__init__()
        files_path = glob.glob(dataset)
        files_path = np.random.permutation(files_path)
        files_path = files_path[:int(len(files_path)*sampling_rate)]
        self.crop_size = crop_size
        self.files_pool = []
        self.use_gt = use_gt
        self.adapt_pattern_dimension = adapt_pattern_dimension
        self.target_pattern_frames = target_pattern_frames
        self.random_pattern_sampling = random_pattern_sampling

        
        for file in tqdm.tqdm(files_path):
            if not use_gt:
                img = min_max_norm(imread(file).astype(np.float32))
                if len(img.shape) == 2:
                    img = img[np.newaxis, np.newaxis, :, :]
                elif len(img.shape) == 3:
                    img = img[np.newaxis, :, :, :]
                
                self.files_pool.append([img])
            else:
                folder = os.path.dirname(file)
                filename = os.path.basename(file)
                filename, ext = os.path.splitext(filename)
                img = min_max_norm(imread(file).astype(np.float32))
                if len(img.shape) == 2:
                    img = img[np.newaxis, np.newaxis, :, :]
                elif len(img.shape) == 3:
                    img = img[np.newaxis, :, :, :]
                
                
                emitter_gt = imread(os.path.join(folder+"_gt", filename+ext)).astype(np.float32)[np.newaxis, np.newaxis, :, :]
                lp_gt = imread(os.path.join(folder+"_gt", filename+"_lp"+ext)).astype(np.float32)[np.newaxis, :, :, :]
                self.files_pool.append([img, emitter_gt, lp_gt])


        assert len(self.files_pool) > 0
        if minimum_number is None:
            minimum_number = len(self.files_pool)
        
        while len(self.files_pool) < minimum_number:
            self.files_pool.extend(self.files_pool)
        self.files_pool = self.files_pool[:minimum_number]
        print(dataset, len(self.files_pool))
    
        
    
    def __getitem__(self, idx):
        files = self.files_pool[idx]
        
        processed_files = []
        for file_item in files:
            if self.adapt_pattern_dimension and file_item.shape[1] > self.target_pattern_frames:
                processed_item = adapt_pattern_dimension_frames(file_item, self.target_pattern_frames, random=self.random_pattern_sampling)
                processed_files.append(processed_item)
            else:
                processed_files.append(file_item)
        
        if self.use_gt and len(processed_files) > 2:
            img, emitter_gt, lp_gt = processed_files
            processed_files = [img, emitter_gt, lp_gt]
        else:
            processed_files = processed_files
        
        processed_files = multi_image_multi_channel_random_crop(processed_files, self.crop_size)
        if self.use_gt:
            img, emitter_gt, lp_gt = processed_files
            return {"img": img, "emitter_gt": emitter_gt, "lp_gt": lp_gt}
        else:
            return {"img": processed_files[0]}

    def __len__(self):
        return len(self.files_pool)
        

class dataset_3d_infer(torch.utils.data.Dataset):
    def __init__(self, image, crop_size, rescale, patch_size_z=1, adapt_pattern_dimension=False, target_pattern_frames=9, random_pattern_sampling=False):
        super().__init__()
        self.overlap = [16, 16]
        self.crop_size = [crop_size[i] - self.overlap[i] * 2 for i in range(len(crop_size))]
        self.rescale = rescale
        self.patch_size_z = patch_size_z
        self.adapt_pattern_dimension = adapt_pattern_dimension
        self.target_pattern_frames = target_pattern_frames
        self.random_pattern_sampling = random_pattern_sampling
        if self.adapt_pattern_dimension and image.shape[1] > self.target_pattern_frames:
            self.image = adapt_pattern_dimension_frames(image, self.target_pattern_frames, random=self.random_pattern_sampling)
        else:
            self.image = image
            
        self.init_infer_tiff()
        self.init_patch_index()
    
    def init_infer_tiff(self):
        new_tiff = min_max_norm(self.image)
        self.image = new_tiff.astype(np.float32).copy()
        self.image_in = new_tiff.astype(np.float32).copy()
        self.input_shape = self.image.shape
        c, z, y, x = self.image.shape
        pad_y = (self.crop_size[0] - y%self.crop_size[0]) % self.crop_size[0]
        pad_x = (self.crop_size[1] - x%self.crop_size[1]) % self.crop_size[1]
        if pad_y != 0 or pad_x != 0:
            self.image = np.pad(self.image, ((0, 0), (0, 0), (0, pad_y), (0, pad_x)), "reflect")
        self.deconv_image = np.zeros([c, 1, self.image.shape[2]*self.rescale[0], self.image.shape[3]*self.rescale[1]])
        self.rec_image = np.zeros([c, z, self.image.shape[2]*self.rescale[0], self.image.shape[3]*self.rescale[1]])
        self.lp_image = np.zeros_like(self.rec_image)
        self.bg_image = np.zeros_like(self.deconv_image)


    def init_patch_index(self):
        c, z, y, x = self.image.shape
        ystart_index = list(range(0, y - 1, self.crop_size[0]))
        xstart_index = list(range(0, x - 1, self.crop_size[1]))
        self.patch_index = [(ys, ys + self.crop_size[0], xs, xs + self.crop_size[1]) for ys in ystart_index for xs in xstart_index]
        self.image = np.pad(self.image, ((0, 0), (0, 0), (self.overlap[0], self.overlap[0]), (self.overlap[1], self.overlap[1])), "reflect")

    def assemble_patch(self, batch_patch_result, batch_patch_index):
        batch_patch_deconv = batch_patch_result["deconv"]
        batch_patch_rec = batch_patch_result["rec_up"]
        batch_patch_lp = batch_patch_result["light_pattern"]
        batch_patch_bg = batch_patch_result["background"]

        ys, ye, xs, xe = batch_patch_index
        for b in range(batch_patch_deconv.shape[0]):
            patch_deconv = batch_patch_deconv[b]
            patch_rec = batch_patch_rec[b]
            patch_lp = batch_patch_lp[b]
            patch_bg = batch_patch_bg[b]

            overlap = [self.overlap[i] * self.rescale[i] for i in range(2)]
            patch_deconv = patch_deconv[..., overlap[0]:-overlap[0], overlap[1]:-overlap[1]]
            patch_rec = patch_rec[..., overlap[0]:-overlap[0], overlap[1]:-overlap[1]]
            patch_lp = patch_lp[..., overlap[0]:-overlap[0], overlap[1]:-overlap[1]]
            patch_bg = patch_bg[..., overlap[0]:-overlap[0], overlap[1]:-overlap[1]]

            y_start = ys[b] * self.rescale[0]
            y_end = y_start + self.crop_size[0] * self.rescale[0]
            x_start = xs[b] * self.rescale[1]
            x_end = x_start + self.crop_size[1] * self.rescale[1]

            self.deconv_image[:, :, y_start:y_end, x_start:x_end] = patch_deconv
            self.rec_image[:, :, y_start:y_end, x_start:x_end] = patch_rec
            self.lp_image[:, :, y_start:y_end, x_start:x_end] = patch_lp
            self.bg_image[:, :, y_start:y_end, x_start:x_end] = patch_bg

    
    @property
    def deconv(self):
        img = self.deconv_image
        c, z, y, x = self.input_shape
        y = y * self.rescale[0]
        x = x * self.rescale[1]
        img = img[..., :y, :x]
        return img
    
    @property
    def reconstruction(self):
        img = self.rec_image
        c, z, y, x = self.input_shape
        y = y * self.rescale[0]
        x = x * self.rescale[1]
        img = img[..., :y, :x]
        return img

    @property
    def light_pattern(self):
        img = self.lp_image
        c, z, y, x = self.input_shape
        y = y * self.rescale[0]
        x = x * self.rescale[1]
        img = img[..., :y, :x]
        return img
    
    @property
    def background(self):
        img = self.bg_image
        c, z, y, x = self.input_shape
        y = y * self.rescale[0]
        x = x * self.rescale[1]
        img = img[..., :y, :x]
        return img

    def __getitem__(self, index):
        ys, ye, xs, xe = self.patch_index[index]
        ys, ye, xs, xe  = ys, ye + self.overlap[0]*2, xs, xe + self.overlap[1]*2
        img = self.image[:, :, ys:ye, xs:xe]
        index = [ys, ys + self.crop_size[0], xs, xs + self.crop_size[1]]
        return img, index

    def __len__(self):
        return len(self.patch_index)


def std_norm(im):
    # Standard normalization
    im = im - im.mean()
    im_std = im.std()
    if im_std > 0:
        im /= im_std
    return im

def min_max_norm(im):
    # Min-max normalization with percentiles
    im_min = np.percentile(im, 0.01)
    im = im - im_min
    im_max = np.percentile(im, 99)
    if im_max > 0:
        im /= im_max
    return im

def get_sample(dataloader):
    # Get single sample from dataloader
    for data in dataloader:
        return jnp.array(data['img'][0:1, ...])