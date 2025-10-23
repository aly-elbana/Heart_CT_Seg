import torch


def dice_coefficient(preds, targets, smooth=1e-6):
    """
    Calculate Dice coefficient for segmentation evaluation.
    
    Args:
        preds: Predicted segmentation masks
        targets: Ground truth segmentation masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient score
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    intersection = (preds * targets).sum()
    return (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def iou_score(preds, targets, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for segmentation evaluation.
    
    Args:
        preds: Predicted segmentation masks
        targets: Ground truth segmentation masks
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        IoU score
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(preds, targets):
    """
    Calculate pixel-wise accuracy for segmentation evaluation.
    
    Args:
        preds: Predicted segmentation masks
        targets: Ground truth segmentation masks
        
    Returns:
        Pixel accuracy score
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    correct = (preds == targets).float().sum()
    total = torch.numel(targets)
    return correct / total
