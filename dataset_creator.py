import os
from glob import glob

import numpy as np
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """
    Dataset class for segmentation training and validation data.
    
    Args:
        image_paths: List of paths to input images
        mask_paths: List of paths to corresponding masks
        transform: Optional transform to be applied to images and masks
    """
    
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single image-mask pair.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, mask) tensors
        """
        image = Image.open(self.image_paths[idx]).convert("L")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        if self.transform:
            image, mask = self.transform(image, mask)
            
        image = image.float()
        mask = mask.float() / 255.0
        
        return image, mask


class DICOMTestDataset(Dataset):
    """
    Dataset class for DICOM test data.
    
    Args:
        image_paths: List of paths to DICOM files
        transform: Optional transform to be applied to images
    """
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a single DICOM image.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image_tensor, image_path)
        """
        img_path = self.image_paths[idx]

        # Read DICOM file
        dcm_image = pydicom.dcmread(img_path)
        image_array = dcm_image.pixel_array

        # Normalize to 0-255 range
        img_norm = (image_array - image_array.min()) / (image_array.max() - image_array.min() + 1e-6)
        image_array_uint8 = (img_norm * 255.0).astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(image_array_uint8).convert("L")

        if self.transform:
            image = self.transform(image)

        image = image.float()
        return image, img_path


def load_data(path, test_path=None, val_split=0.2, train_transform=None, val_transform=None):
    """
    Load and split dataset into train, validation, and test sets.
    
    Args:
        path: Path to training data directory
        test_path: Optional path to test data directory
        val_split: Fraction of data to use for validation (default: 0.2)
        train_transform: Transform for training data
        val_transform: Transform for validation/test data
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Load training images and masks
    images = sorted(glob(os.path.join(path, "*", "image", "*.png")))
    masks = sorted(glob(os.path.join(path, "*", "mask", "*.png")))

    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=val_split, random_state=42
    )
    
    # Create datasets
    train_dataset = SegmentationDataset(X_train, y_train, train_transform)
    val_dataset = SegmentationDataset(X_val, y_val, val_transform)
    
    # Load test data (DICOM files) if path is provided
    test_dataset = None
    if test_path:
        X_test = sorted(glob(os.path.join(test_path, "*", "*", "*.dcm")))
        if X_test:
            test_dataset = DICOMTestDataset(X_test, val_transform)

    return train_dataset, val_dataset, test_dataset