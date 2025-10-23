"""
Data augmentation and transformation classes for medical image segmentation.
"""

import random

import torchvision.transforms.functional as F


class SegmentationTrainTransform:
    """
    Training data augmentation transform for segmentation tasks.
    
    Applies random augmentations including:
    - Resizing to target size
    - Random horizontal and vertical flips
    - Random rotation (-20 to +20 degrees)
    - Normalization
    
    Args:
        image_size: Target image size (height, width) (default: (512, 512))
    """
    
    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size
        
    def __call__(self, image, mask):
        """
        Apply training augmentations to image and mask.
        
        Args:
            image: PIL Image
            mask: PIL Image mask
            
        Returns:
            Tuple of (augmented_image, augmented_mask) tensors
        """
        # Resize to target size
        image = F.resize(image, self.image_size)
        mask = F.resize(mask, self.image_size)
        
        # Random horizontal flip (50% probability)
        if random.random() < 0.5:
            image, mask = F.hflip(image), F.hflip(mask)
            
        # Random vertical flip (50% probability)
        if random.random() < 0.5:
            image, mask = F.vflip(image), F.vflip(mask)
            
        # Random rotation (-20 to +20 degrees)
        angle = random.uniform(-20, 20)
        image, mask = F.rotate(image, angle), F.rotate(mask, angle)
        
        # Convert to tensors and normalize
        image = F.to_tensor(image)
        mask = F.pil_to_tensor(mask)
        image = F.normalize(image, mean=[0.5], std=[0.5])
        
        return image, mask


class SegmentationValTransform:
    """
    Validation/test data transform for segmentation tasks.
    
    Applies only basic preprocessing:
    - Resizing to target size
    - Normalization
    - No random augmentations
    
    Args:
        image_size: Target image size (height, width) (default: (512, 512))
    """
    
    def __init__(self, image_size=(512, 512)):
        self.image_size = image_size
        
    def __call__(self, image, mask=None):
        """
        Apply validation preprocessing to image and optional mask.
        
        Args:
            image: PIL Image
            mask: Optional PIL Image mask
            
        Returns:
            Processed image tensor, or tuple of (image, mask) if mask provided
        """
        # Resize to target size
        image = F.resize(image, self.image_size)
        
        # Convert to tensor and normalize
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.5], std=[0.5])

        if mask is not None:
            # Process mask if provided
            mask = F.resize(mask, self.image_size)
            mask = F.pil_to_tensor(mask)
            return image, mask

        return image