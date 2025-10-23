import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset_creator import load_data
from transformations import SegmentationTrainTransform, SegmentationValTransform
from model import UNet, BCEDiceLoss
from rating_metrices import dice_coefficient, iou_score, pixel_accuracy


def create_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    # Create directories
    create_dir(os.path.dirname(args.save_path))

    # Data transforms
    image_size_tuple = (args.image_size, args.image_size)
    train_transform = SegmentationTrainTransform(image_size_tuple)
    val_transform = SegmentationValTransform(image_size_tuple)

    # Load datasets
    # We pass test_path=None because we are not using the unlabeled test set during training.
    train_dataset, val_dataset, _ = load_data(
        args.dataset_path, test_path=None, 
        val_split=0.2, 
        train_transform=train_transform, 
        val_transform=val_transform
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model setup
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, 
        verbose=True, min_lr=1e-6
    )

    # Training variables
    best_val_loss = float("inf")

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("-" * 60)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_train_loss = 0.0
        
        train_loop = tqdm(
            train_loader, 
            desc=f"Epoch [{epoch}/{args.epochs}] Training", 
            leave=False
        )
        
        for images, masks in train_loop:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            masks = masks.to(device, dtype=torch.float32, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            val_loop = tqdm(
                val_loader, 
                desc=f"Epoch [{epoch}/{args.epochs}] Validation", 
                leave=False
            )
            
            for images, masks in val_loop:
                images = images.to(device, dtype=torch.float32, non_blocking=True)
                masks = masks.to(device, dtype=torch.float32, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                running_val_loss += loss.item()
                total_dice += dice_coefficient(outputs, masks).item()
                total_iou += iou_score(outputs, masks).item()
                total_acc += pixel_accuracy(outputs, masks).item()
        
        # Calculate metrics
        avg_val_loss = running_val_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        avg_acc = total_acc / len(val_loader)
        epoch_time = time.time() - start_time
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss  : {avg_val_loss:.4f}")
        print(f"Dice Score: {avg_dice:.6f}")
        print(f"IoU Score : {avg_iou:.6f}")
        print(f"Accuracy  : {avg_acc:.6f}")
        print(f"LR        : {current_lr:.6f}")
        print(f"Time      : {epoch_time:.2f}s")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"✓ New best model saved to '{args.save_path}'")
        
        print("-" * 60)

    print("Training completed!")
    
    # --- Final Evaluation ---
    print("\n" + "=" * 60)
    print("Starting final evaluation on validation set with best model...")
    
    # Load best model
    model.load_state_dict(torch.load(args.save_path))
    model.eval()
    
    running_val_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        final_val_loop = tqdm(
            val_loader, 
            desc="Final Evaluation", 
            leave=False
        )
        
        for images, masks in final_val_loop:
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            masks = masks.to(device, dtype=torch.float32, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_val_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks).item()
            total_iou += iou_score(outputs, masks).item()
            total_acc += pixel_accuracy(outputs, masks).item()

    # Calculate final metrics
    avg_val_loss = running_val_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_acc = total_acc / len(val_loader)

    print("\n--- Final Validation Metrics ---")
    print(f"Val Loss  : {avg_val_loss:.4f}")
    print(f"Dice Score: {avg_dice:.6f}")
    print(f"IoU Score : {avg_iou:.6f}")
    print(f"Accuracy  : {avg_acc:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet for Heart Segmentation")
    
    parser.add_argument('--dataset_path', type=str, default=os.path.join("data", "train"),
                        help="Path to the training data directory")
    parser.add_argument('--save_path', type=str, default=os.path.join("models", "best_unet_model.pth"),
                        help="Path to save the best model weights")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Training batch size")
    parser.add_argument('--epochs', type=int, default=30,
                        help="Number of training epochs")
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help="Optimizer learning rate")
    parser.add_argument('--image_size', type=int, default=512,
                        help="Target image size (height and width)")

    args = parser.parse_args()
    main(args)