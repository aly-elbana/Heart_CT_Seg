import os
import argparse
from typing import Tuple

import cv2 as cv
import gradio as gr
import numpy as np
import torch
from PIL import Image

from model import UNet
from transformations import SegmentationValTransform

# Configuration
DEFAULT_MODEL_PATH = "models/best_unet_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variables
model = None
transform = None


def load_model(model_path: str):
    """Load the trained UNet model."""
    global model, transform
    
    try:
        # Load model
        model = UNet(in_channels=1, out_channels=1).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Initialize the correct validation transform
        # This MUST match the validation transform from training (e.g., 512x512)
        transform = SegmentationValTransform((512, 512))
        
        print(f"Model loaded successfully from {model_path}")
        print(f"Using device: {DEVICE}")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def preprocess_image(image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Preprocess image for model inference using the validation transform.
    
    Args:
        image: PIL Image
        
    Returns:
        Tuple of (processed_tensor, original_size)
    """
    global transform
    if transform is None:
        raise RuntimeError("Transform is not initialized. Model may not be loaded.")

    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Get original dimensions
    original_w, original_h = image.size
    
    # Apply the validation transform (handles resizing and normalization)
    image_tensor = transform(image)
    
    # Add batch dimension and send to device
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    return image_tensor, (original_w, original_h)


def predict_segmentation(image: Image.Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict heart segmentation for the input image.
    
    Args:
        image: Input PIL Image
        
    Returns:
        Tuple of (original_image, mask, overlay_image)
    """
    if model is None:
        raise RuntimeError("Model not loaded. Please restart the application.")
    
    try:
        # Preprocess image
        image_tensor, original_size = preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            # Convert to numpy (this will be 512x512, or training size)
            pred_np = preds[0, 0].cpu().numpy()
            
            # Resize back to original dimensions using INTER_NEAREST
            pred_resized = cv.resize(pred_np, original_size, interpolation=cv.INTER_NEAREST)
            
            # Convert original image to numpy
            original_np = np.array(image.convert('RGB'))
            
            # Create overlay
            overlay = original_np.copy()
            mask_uint8 = (pred_resized * 255).astype(np.uint8)
            contours, _ = cv.findContours(mask_uint8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cv.drawContours(overlay, contours, -1, (255, 0, 0), 2)
            
            return original_np, pred_resized, overlay
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise RuntimeError(f"Prediction failed: {str(e)}")


def create_interface(model_path: str):
    """Create the Gradio interface."""
    
    # Load model
    if not load_model(model_path):
        raise RuntimeError(f"Failed to load model from {model_path}. Please check the file path.")
    
    # Define the interface
    with gr.Blocks(
        title="Heart CT Segmentation",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>Heart CT Segmentation</h1>
            <p>Upload a CT scan image to automatically segment the heart using our trained UNet model.</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Upload Image")
                input_image = gr.Image(
                    label="CT Scan Image",
                    type="pil",
                    height=300
                )
                
                # Controls
                gr.Markdown("### Controls")
                predict_btn = gr.Button(
                    "Segment Heart",
                    variant="primary",
                    size="lg"
                )
                
                # Model info
                gr.Markdown("### Model Information")
                gr.Markdown(f"""
                - **Device**: {DEVICE}
                - **Model**: UNet Architecture
                - **Input**: Grayscale CT Images
                - **Output**: Binary Heart Segmentation
                """)
            
            with gr.Column(scale=2):
                # Output section
                gr.Markdown("### Results")
                
                with gr.Tabs():
                    with gr.TabItem("Original Image"):
                        original_output = gr.Image(
                            label="Original CT Scan",
                            height=300
                        )
                    
                    with gr.TabItem("Segmentation Mask"):
                        mask_output = gr.Image(
                            label="Heart Segmentation Mask (White = Heart Region)",
                            height=300,
                            image_mode="L" # Display as grayscale
                        )
                    
                    with gr.TabItem("Overlay"):
                        overlay_output = gr.Image(
                            label="Segmentation Overlay",
                            height=300
                        )
        
        # Event handlers
        def process_image(image):
            """Process the uploaded image and return results."""
            if image is None:
                gr.Warning("Please upload an image first.")
                return None, None, None
            
            try:
                # Run prediction
                original, mask, overlay = predict_segmentation(image)
                
                # Convert mask to 0-255 uint8 for Gradio display
                mask_display = (mask * 255).astype(np.uint8)
                
                if np.sum(mask_display) == 0:
                    gr.Info("No heart region was detected in the image.")
                
                return original, mask_display, overlay
                
            except Exception as e:
                gr.Error(f"Prediction failed: {str(e)}")
                return None, None, None
        
        # Connect button to processing function
        predict_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[original_output, mask_output, overlay_output]
        )
        
        # Auto-process when image is uploaded (optional, can be noisy)
        # input_image.change(
        #     fn=process_image,
        #     inputs=[input_image],
        #     outputs=[original_output, mask_output, overlay_output]
        # )
    
    return interface


def main(args):
    """Main function to launch the Gradio app."""
    print("Starting Heart CT Segmentation Web Application...")
    
    try:
        # Create and launch interface
        interface = create_interface(args.model_path)
        interface.launch()
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio Web App for Heart Segmentation")
    
    parser.add_argument('--model_path', type=str, 
                        default=DEFAULT_MODEL_PATH,
                        help="Path to the trained model weights")

    args = parser.parse_args()
    main(args)