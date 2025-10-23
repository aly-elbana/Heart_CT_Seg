"""
UNet model architecture for medical image segmentation.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class UNet(nn.Module):
    """
    UNet architecture for medical image segmentation.
    
    Args:
        in_channels: Number of input channels (default: 1 for grayscale)
        out_channels: Number of output channels (default: 1 for binary segmentation)
        init_features: Number of initial features (default: 64)
    """
    
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(UNet, self).__init__()
        features = init_features

        # Encoder path
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16)

        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, stride=2)
        self.decoder4 = self._block(features * 16, features * 8)

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, stride=2)
        self.decoder3 = self._block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, stride=2)
        self.decoder2 = self._block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        # Final output layer
        self.conv_final = nn.Conv2d(features, out_channels, kernel_size=1)

    def _block(self, in_channels, out_channels):
        """
        Create a double convolution block with BatchNorm and ReLU.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            
        Returns:
            Sequential block with two convolutions
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the UNet architecture.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Segmentation output tensor
        """
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv_final(dec1)


class BCEDiceLoss(nn.Module):
    """
    Combined BCE and Dice loss for segmentation tasks.
    
    Args:
        smooth: Smoothing factor for Dice loss (default: 1.0)
    """
    
    def __init__(self, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, inputs, targets):
        """
        Calculate combined BCE and Dice loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth masks
            
        Returns:
            Combined loss value
        """
        # Handle size mismatch
        if inputs.size() != targets.size():
            targets = F.interpolate(targets, size=inputs.shape[2:], mode="nearest")
            
        # BCE loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice loss
        probs = torch.sigmoid(inputs)
        targets = targets.float()
        
        probs_flat = probs.contiguous().view(probs.size(0), -1)
        targets_flat = targets.contiguous().view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice_score = (2. * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice_score.mean()

        return bce_loss + dice_loss