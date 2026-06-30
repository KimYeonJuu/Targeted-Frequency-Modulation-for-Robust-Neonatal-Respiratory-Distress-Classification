import os
import cv2
import torch
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sci.constants import *
from sci.data_preprocessing import apply_blur_with_annotation
import torchvision.transforms.functional as TF

class ImageBaseDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_path=None,
        transform=None,
    ):
        self.data_path = data_path
        self.transform = transform
        self.split = split
        self.image_size = 256  # Default image size, can be overridden

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def read_from_jpg(self, img_path, img_size):
        x = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if x is None:
            print(f"Warning: Failed to read image at {img_path}. Skipping.")
            return None  # Return None when the image cannot be loaded so the caller can skip it.

        # transform images
        x = self._resize_img(x, img_size)
        img = Image.fromarray(x).convert("RGB")
        if self.transform is not None:
            # Check whether the transform is an Albumentations transform.
            if 'albumentations' in str(type(self.transform)).lower():
                img_np = np.array(img)
                img = self.transform(image=img_np)['image']
            else:
                img = self.transform(img)
        if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
            img = img.float() / 255.0
        return img

    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img


class CheXpertImageDataset(ImageBaseDataset):
    def __init__(self, split="train", data_path=None, transform=None, image_size=256):
        
        self.data_path = data_path
        self.image_size = image_size
        # read in csv file
        if data_path is not None:
            self.df = pd.read_csv(data_path)
        else:
            if split == "train":
                self.df = pd.read_csv(CHEXPERT_TRAIN_CSV)
            elif split == "valid":
                self.df = pd.read_csv(CHEXPERT_VALID_CSV)
            else:
                self.df = pd.read_csv(CHEXPERT_TEST_CSV)

        # sample data
        # if cfg.data.frac != 1 and split == "train":
        #     self.df = self.df.sample(frac=cfg.data.frac)

        # # filter image type
        # if img_type != "All":
        #     self.df = self.df[self.df[CHEXPERT_VIEW_COL] == img_type]

        # Resolve paths from absolute paths, paths relative to the current
        # directory, or paths relative to CHEXPERT_DATA_DIR.
        def _resolve_chexpert_path(path_value):
            path_value = str(path_value)
            if os.path.isabs(path_value) or os.path.exists(path_value):
                return path_value

            candidates = [
                os.path.join(CHEXPERT_DATA_DIR, path_value),
                os.path.join(CHEXPERT_DATA_DIR, "/".join(path_value.split("/")[1:])),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
            return candidates[0]

        self.df[CHEXPERT_PATH_COL] = self.df[CHEXPERT_PATH_COL].apply(_resolve_chexpert_path)

        # fill na with 0s
        self.df = self.df.fillna(0)

        # replace uncertains
        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        self.df = self.df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
        super(CheXpertImageDataset, self).__init__(split, data_path, transform)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = row["Path"]
        x = self.read_from_jpg(img_path, self.image_size)

        y = list(row[CHEXPERT_COMPETITION_TASKS])
        y = torch.tensor(y)

        dummy_mask = torch.ones((self.image_size, self.image_size), dtype=torch.uint8)
        return x, y, dummy_mask

    def __len__(self):
        return len(self.df)

    
class RDSImageDataset(ImageBaseDataset):
    def __init__(self, split="test", data_path=None, transform=None, image_size=256, use_blur=False, ksize=21, sigma=0):
        self.data_path = data_path
        self.image_size = image_size
        self.use_blur = use_blur
        self.ksize = ksize
        self.sigma = sigma
        # Read the CSV file.
        if data_path is not None:
            self.df = pd.read_csv(data_path)
        else:
            if split == "train":
                self.df = pd.read_csv(RDS_TRAIN_CSV)
            elif split == "valid":
                self.df = pd.read_csv(RDS_VALID_CSV)
            else:
                self.df = pd.read_csv(RDS_TEST_CSV)
        super(RDSImageDataset, self).__init__(split, data_path, transform)

    def __getitem__(self, index):
        # Retry until a valid sample is found.
        row = self.df.iloc[index]
        # img_path = row["filepath"]
        img_path = row["Path"]
        mask_path = img_path.replace('_seg_crop.png', '_seg_crop_mask.png')
        
        # Apply blur only when enabled and the path is a virtual crop.
        if self.use_blur and "_crop_virtual.png" in img_path:
            try:
                # Apply blur; the helper reads the image as grayscale.
                blurred_gray = apply_blur_with_annotation(img_path, ksize=(self.ksize, self.ksize), sigma=self.sigma)
                
                if blurred_gray is None:
                    print(f"Warning: Failed to apply blur at {img_path}")
                    return self.__getitem__((index + 1) % len(self.df))
                
                # Resize the image using the same 2D-array path.
                blurred_gray = self._resize_img(blurred_gray, self.image_size)
                
                # Convert to PIL and RGB.
                img = Image.fromarray(blurred_gray).convert("RGB")
                
                # Apply transform.
                if self.transform is not None:
                    # Check whether the transform is an Albumentations transform.
                    if 'albumentations' in str(type(self.transform)).lower():
                        img_np = np.array(img)
                        img = self.transform(image=img_np)['image']
                    else:
                        img = self.transform(img)
                
                # Handle tensor dtype.
                if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
                    img = img.float() / 255.0
                    
                x = img
                    
            except Exception as e:
                print(f"Error applying blur to {img_path}: {e}")
                # Fall back to ordinary image loading if blur fails.
                x = self.read_from_jpg(img_path, self.image_size)
        else:
            # Use ordinary image loading when blur is disabled or the file is not a virtual crop.
            x = self.read_from_jpg(img_path, self.image_size)
            
        if x is None:
            return self.__getitem__((index + 1) % len(self.df))
        
        # y = float(row["rds"])
        # y = torch.tensor([y])
        y = list(row[RDS_TASKS])
        y = torch.tensor(y, dtype=torch.float)

        # Read and resize the mask, then convert it to a tensor (H, W).
        mask_img = Image.open(mask_path).convert("L")
        mask_img = mask_img.resize((self.image_size, self.image_size), Image.NEAREST)
        mask = TF.to_tensor(mask_img).squeeze(0)  # (H, W)
        mask = (mask > 0).to(torch.uint8)
        
        return x, y, mask

    def __len__(self):
        return len(self.df)

class RSNAImageDataset(ImageBaseDataset):
    def __init__(self, split="train", data_path=None, transform=None, image_size=256):
        self.data_path = data_path
        self.image_size = image_size
        

        # read in csv file
        if split == "train":
            self.df = pd.read_csv(RSNA_TRAIN_CSV)
        elif split == "valid":
            self.df = pd.read_csv(RSNA_VALID_CSV)
        else:
            self.df = pd.read_csv(RSNA_TEST_CSV)

        # if cfg.phase == "detection":
        #     self.df = self.df[self.df["Target"] == 1]

        # sample data
        # if self.data_frac != 1 and split == "train":
        #     self.df = self.df.sample(frac=self.data_frac)
            
        super(RSNAImageDataset, self).__init__(split, data_path, transform)
    
    def __getitem__(self, index):

        row = self.df.iloc[index]

        # get image
        img_path = row["Path"]
        x = self.read_from_jpg(img_path, self.image_size)

        # get labels
        y = list(row[RSNA_COMPETITION_TASKS])
        y = torch.tensor(y)

        return x, y

    def __len__(self):
        return len(self.df)
