# Heart CT Segmentation

This project provides a complete deep learning pipeline for automatic heart segmentation in CT scans using a U-Net architecture. It includes scripts for training, command-line inference, and an interactive web demo.

> (This is a placeholder. You should add a screenshot or GIF of your Gradio app in action!)

## Features

- U-Net Model: A robust U-Net implementation tailored for medical image segmentation.
- Web Interface: An easy-to-use Gradio web application for interactive segmentation.
- End-to-End Pipeline: Includes data loading, augmentation, training, and evaluation.
- Advanced Training: Features a combined BCE + Dice Loss, AdamW optimizer, and ReduceLROnPlateau learning rate scheduling.
- Medical Format Support: The data loader is built to handle standard PNGs for training and can be adapted for DICOM files.
- Evaluation Metrics: Calculates Dice Coefficient, IoU (Jaccard Index), and Pixel Accuracy for model validation.

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aly-elbana/Heart_CT_Seg.git
cd heart-ct-seg
```

### 2. Create a Virtual Environment (Recommended)

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all required packages using the fixed requirements.txt.

```bash
pip install -r requirements.txt
```

### 4. Download Data

This repository does not include training or testing data. You must provide your own dataset. The training script expects data to be organized in the following structure:

```
data/
└── train/
    ├── patient_001/
    │   ├── image/
    │   │   └── scan_001.png
    │   └── mask/
    │       └── mask_001.png
    ├── patient_002/
    │   ...
    └── ...
```

## Usage

### 1. Train a New Model

To train the U-Net from scratch, run `model_train.py`. All hyperparameters can be passed as command-line arguments.

```bash
python model_train.py \
    --dataset_path "path/to/your/data/train" \
    --save_path "models/my_best_model.pth" \
    --epochs 50 \
    --batch_size 4 \
    --learning_rate 0.0001
```

Training progress, validation loss, and metrics will be printed to the console, and the best-performing model will be saved to the path specified by `--save_path`.

### 2. Run the Web Interface (Gradio)

To launch the interactive demo, run `app.py`. You can (and should) point it to your newly trained model.

```bash
python app.py --model_path "models/my_best_model.pth"
```

Navigate to the local URL (e.g., `http://127.0.0.1:7860`) in your browser to upload a CT scan and see the segmentation in real-time.

### 3. Run Inference from Command Line

For a single-image test, use `main.py`. This script will load an image, perform segmentation, and display the results using matplotlib.

```bash
python main.py \
    --image_path "path/to/your/image.png" \
    --model_path "models/my_best_model.pth"
```

## Project Structure

```
heart-ct-seg/
├── app.py                # Gradio web application
├── main.py               # Command-line inference script
├── model.py              # U-Net architecture & BCEDiceLoss
├── model_train.py        # Model training script
├── dataset_creator.py    # PyTorch Dataset/Dataloader classes
├── transformations.py    # Data augmentation transforms
├── rating_metrices.py    # Dice, IoU, Pixel Accuracy metrics
├── requirements.txt      # Project dependencies
├── notebooks/            # Jupyter notebooks
│   └── heart_seg.ipynb
├── models/               # Model weights and architecture
│   ├── model_skeleton.py
│   └── best_unet_model.pth
└── data/                 # Dataset directory
    └── train/
        ├── patient_001/
        │   ├── image/
        │   └── mask/
        └── patient_002/
```

## Model Details

- Architecture: A standard U-Net with a 4-level encoder-decoder path. Skip connections concatenate features from the encoder to the decoder at each level.
- Training Augmentations: `SegmentationTrainTransform` applies resizing, random horizontal flips, random vertical flips, and random rotation (-20 to +20 degrees).
- Validation Transforms: `SegmentationValTransform` applies only resizing and normalization to ensure consistent evaluation.
- Loss Function: `BCEDiceLoss`, a compound loss that combines `BCEWithLogitsLoss` (for pixel-wise stability) and Dice (to combat class imbalance).
- Optimizer: AdamW.
- Scheduler: ReduceLROnPlateau, which reduces the learning rate when the validation loss stops improving.

## Hardware Requirements

- Python 3.8+
- PyTorch 1.9.0+
- NVIDIA CUDA GPU (Recommended): Training is computationally intensive and will be extremely slow on a CPU.
- 8GB+ RAM: Required for loading and augmenting 512x512 images.
