import os
import argparse

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import UNet
from transformations import SegmentationValTransform

# Set environment variable for OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def predict_single_image(model, transform, image_path, device):
    """
    Predict segmentation for a single image.
    
    Args:
        model: Trained UNet model
        transform: The validation transform object
        image_path: Path to input image
        device: Device to run inference on
        
    Returns:
        Segmentation mask as numpy array, or None if failed
    """
    # Load image with OpenCV to get original dimensions and for visualization
    image_cv = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image_cv is None:
        print(f"Error: Cannot read image from {image_path}")
        return None
    
    # Get original dimensions
    original_h, original_w = image_cv.shape
    print(f"Original image size: {original_w}x{original_h}")
    
    # Load image with PIL for correct transformation
    try:
        pil_image = Image.open(image_path).convert("L")
    except Exception as e:
        print(f"Error: Cannot read image with PIL from {image_path}: {e}")
        return None

    # Apply the *exact* validation transform
    # This handles resizing to 512x512 and normalization (mean=0.5, std=0.5)
    image_tensor = transform(pil_image)
    
    # Add batch dimension and send to device
    image_tensor = image_tensor.unsqueeze(0).to(device, dtype=torch.float32)
    
    # Run inference
    with torch.no_grad():
        print("Running prediction...")
        outputs = model(image_tensor)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        
        # Convert to numpy. This mask is (512, 512)
        pred_np = preds[0, 0].cpu().numpy().astype(np.uint8)
        
        # Resize mask back to original dimensions
        print(f"Resizing predicted mask from {pred_np.shape} to {(original_w, original_h)}")
        pred_resized = cv.resize(pred_np, (original_w, original_h), interpolation=cv.INTER_NEAREST)
        
        # Create visualization
        img_rgb = cv.cvtColor(image_cv, cv.COLOR_GRAY2RGB)
        contours, _ = cv.findContours(pred_resized, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        overlay_img = cv.drawContours(img_rgb.copy(), contours, -1, (255, 0, 0), 2)
        
        # Display results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_cv, cmap='gray')
        axes[0].set_title(f"Original Image\n{os.path.basename(image_path)}")
        axes[0].axis('off')
        
        axes[1].imshow(pred_resized, cmap='gray')
        axes[1].set_title("Prediction (Mask)")
        axes[1].axis('off')
        
        axes[2].imshow(overlay_img)
        axes[2].set_title("Prediction Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("Prediction completed successfully!")
        return pred_resized


def main(args):
    """Main inference function."""
    
    # Load model
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded successfully from {args.model_path}")
    model.eval()

    # Initialize the *correct* validation transform
    # Assumes model was trained with 512x512 images
    transform = SegmentationValTransform((512, 512))

    print(f"Processing image: {args.image_path}")
    prediction = predict_single_image(model, transform, args.image_path, device)
    
    if prediction is not None:
        print("✓ Segmentation completed successfully!")
    else:
        print("✗ Segmentation failed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UNet inference on a single image")
    
    parser.add_argument('--model_path', type=str, 
                        default="models/best_unet_model.pth",
                        help="Path to the trained model weights")
    parser.add_argument('--image_path', type=str, 
                        required=True,
                        help="Path to the input image")

    args = parser.parse_args()
    main(args)