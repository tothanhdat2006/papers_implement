import numpy as np
from PIL import Image
from pathlib import Path
from os import listdir
from os.path import splitext, isfile, join

import logging
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

def load_image(filename):
    ext = splitext(filename)[1] # Extract extension name

    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', 'pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)        

def load_mask(idx, dir_masks, mask_suffix):
    tmp = list(dir_masks.glob(idx + mask_suffix + '.*'))
    mask_file = list(dir_masks.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))

    if mask.ndim == 2:
        return np.unique(mask) # Black and white mask of image
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1]) # Color mask of image
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')

class ImageCustomDataset(Dataset):
    def __init__(self, dir_images: str, dir_masks: str, scale: float = 1.0, mask_suffix: str = ''):
        self.dir_images = Path(dir_images)
        self.dir_masks = Path(dir_masks)
        self.scale = scale
        self.mask_suffix = mask_suffix

        # Check for input file
        self.ids = [splitext(file)[0] for file in listdir(dir_images) if isfile(join(dir_images, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {dir_images}, make sure you put your images there')

        print(len(self.ids))
        # ???
        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(load_mask, dir_masks=self.dir_masks, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    @staticmethod
    def preprocess(mask_values, img, scale, is_mask):
        w, h = img.size

        # Scaling
        newW, newH = int(scale * w), int(scale * h) 
        if newW == 0 or newH == 0:
            print(f'Image is too small: {w} x {h}')
            return
        
        # Resize
        img = img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(img)

        # Add/reshape channel for image and Standardize image
        if not is_mask:
            if img.ndim == 2:
                img = img[np.newaxis, ...] # Add channel for black and white image 
            else:
                img = img.transpose((2, 0, 1)) # Convert BGR to RGB
            
            if (img > 1).any():
                img = img / 255.0 # Standardize
            
            return img
        # ???
        else:
            mask = np.zeros((newH, newW), dtype=np.int64) # H x W
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask
            

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.dir_images.glob(name + '.*'))
        mask_file = list(self.dir_masks.glob(name + self.mask_suffix + '.*'))

        # Check for non-exist or duplicate name
        if len(img_file) != 1:
            print(f'No or multiple images found for ID {name}: {img_file}')
        if len(mask_file) != 1:
            print(f'No or multiple images found for ID {name}: {mask_file}')

        # Check size
        tmpImg = load_image(img_file[0])
        tmpMask = load_image(mask_file[0])
        if tmpImg.size != tmpMask.size:
            print(f'Image and mask {name} should be the same size, but are {tmpImg.size} and {tmpMask.size}')
            return
        
        # Preprocess
        img_preprocessed = self.preprocess(self.mask_values, tmpImg, self.scale, is_mask = False)
        mask_preprocessed = self.preprocess(self.mask_values, tmpMask, self.scale, is_mask = True)

        # Return
        return {
            'image': torch.as_tensor(img_preprocessed.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask_preprocessed.copy()).long().contiguous()
        }
        

class MaskCustomDataset(ImageCustomDataset):
    def __init__(self, dir_images: str, dir_masks: str, scale: float = 1.0, mask_suffix: str = ''):
        super().__init__(dir_images, dir_masks, scale, mask_suffix='_mask')
