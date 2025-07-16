# Computer Vision with Deep Learning

A comprehensive collection of computer vision examples and tutorials covering the concepts included in our "Computer Vision with Deep Learning" introduction course!

## 📁 Repository Structure

### 📓 Jupyter Notebooks

#### `CLIP_example.ipynb`
**CLIP (Contrastive Language-Image Pre-training) Demonstration**

This notebook demonstrates how to use OpenAI's CLIP model for understanding relationships between images and text. CLIP is a multimodal model that can:

- **Compare images to text descriptions** - Find how well different text prompts describe an image
- **Compare text to multiple images** - Find which images best match a given text query
- **Zero-shot classification** - Classify images without training on specific categories

**Key Features:**
- Uses the `openai/clip-vit-base-patch32` model from Hugging Face
- Demonstrates both image-to-text and text-to-image similarity scoring
- Includes visualization of results with probability scores
- Shows practical applications like content-based image search

**Technologies:** `transformers`, `torch`, `torchvision`, `matplotlib`, `PIL`

#### `Convolution_MNIST.ipynb`
**Convolutional Neural Network for MNIST Digit Classification**

A simple tutorial on building and training a basic Convolutional Neural Network (CNN) from scratch for handwritten digit recognition using the MNIST dataset.

**What You'll Learn:**
- **CNN Architecture Design** - How convolutional layers extract features from images
- **Training Loop Implementation** - Complete training and validation procedures
- **Model Evaluation** - Monitoring training progress and validation performance
- **PyTorch Fundamentals** - Data loading, model definition, and optimization

**Architecture Details:**
- 3 convolutional layers with ReLU activation
- Progressive feature extraction (1→16→32→64 channels)
- Spatial downsampling with stride=2
- Final classification layer for 10 digit classes

**Technologies:** `torch`, `torchvision`, `torch.nn`, `torch.optim`

#### `YOLO_example.ipynb`
**YOLOS Object Detection Tutorial**

Demonstrates modern object detection using YOLOS (You Only Look One-level Series), a Vision Transformer-based approach to object detection.

**Capabilities:**
- **Multi-object Detection** - Identify multiple objects in a single image
- **Bounding Box Prediction** - Precise localization of detected objects
- **COCO Dataset Classes** - Recognition of 80+ common object categories
- **Confidence Scoring** - Reliability assessment of each detection

**Key Features:**
- Uses the `hustvl/yolos-tiny` model for efficient inference
- Includes post-processing for filtering high-confidence detections
- Visualizes results with bounding boxes and class labels
- Demonstrates end-to-end object detection pipeline

**Technologies:** `transformers`, `torch`, `torchvision`, `matplotlib`, `PIL`

## 🚀 Getting Started

### Prerequisites
```bash
pip install transformers torch torchvision matplotlib pillow
```

### Running the Examples

1. **CLIP Example**: Open `CLIP_example.ipynb` to explore image-text similarity
2. **CNN Training**: Run `Convolution_MNIST.ipynb` to train a digit classifier
3. **Object Detection**: Try `YOLO_example.ipynb` for detecting objects in images

### Google Colab Support
All notebooks include Colab badges for easy cloud execution:
- Click the "Open in Colab" button at the top of each notebook
- No local setup required - runs entirely in the browser

## 📄 License

This project is released into the public domain under the Unlicense. See the `LICENSE` file for details.