# Skin Lesion Classification using Deep Learning

An AI-powered skin lesion classification system that uses a custom Convolutional Neural Network (CNN) to identify 7 different types of skin lesions from dermatoscopic images. Built with TensorFlow/Keras and deployed using Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

## Overview

This project implements a deep learning solution for automated skin lesion classification to assist dermatologists in early detection of skin cancer. The system achieves **92%+ accuracy** on the HAM10000 dataset and provides a user-friendly web interface for real-time predictions.

### Key Highlights
- Custom CNN architecture built from scratch
- Trained on 10,015 dermatoscopic images
- Handles class imbalance using oversampling and class weights
- Real-time predictions via Streamlit web app
- Comprehensive model evaluation with multiple metrics

## Features

- **Multi-class Classification**: Identifies 7 types of skin lesions
  - Melanocytic nevi (nv)
  - Melanoma (mel)
  - Benign keratosis-like lesions (bkl)
  - Basal cell carcinoma (bcc)
  - Actinic keratoses (akiec)
  - Vascular lesions (vasc)
  - Dermatofibroma (df)

- **Advanced Data Processing**:
  - Data augmentation (rotation, flip, zoom, shift)
  - Class balancing techniques
  - Image normalization and preprocessing

- **Interactive Web Interface**:
  - Easy image upload
  - Real-time predictions
  - Confidence scores
  - Probability distribution visualization

- **Comprehensive Evaluation**:
  - Confusion matrix
  - Classification report
  - Per-class metrics (Precision, Recall, F1-Score)
  - AUC score

## Dataset

**HAM10000 (Human Against Machine with 10000 training images)**

- **Total Images**: 10,015 dermatoscopic images
- **Classes**: 7 different types of skin lesions
- **Source**: [HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- **Format**: JPG images of varying sizes
- **Split**: 70% Training, 15% Validation, 15% Testing

### Class Distribution (Before Balancing)
| Class | Name | Count |
|-------|------|-------|
| nv | Melanocytic nevi | 6,705 |
| mel | Melanoma | 1,113 |
| bkl | Benign keratosis | 1,099 |
| bcc | Basal cell carcinoma | 514 |
| akiec | Actinic keratoses | 327 |
| vasc | Vascular lesions | 142 |
| df | Dermatofibroma | 115 |

## Model Architecture

### Custom CNN Architecture

```
Input (224×224×3)
    ↓
[Conv Block 1] 32 filters  → 112×112×32
    ↓
[Conv Block 2] 64 filters  → 56×56×64
    ↓
[Conv Block 3] 128 filters → 28×28×128
    ↓
[Conv Block 4] 256 filters → 14×14×256
    ↓
Flatten → 50,176 features
    ↓
[Dense Layer 1] 512 neurons
    ↓
[Dense Layer 2] 256 neurons
    ↓
[Dense Layer 3] 128 neurons
    ↓
Output: 7 classes (Softmax)
```

### Each Convolutional Block Contains:
- 2× Conv2D layers (3×3 kernel)
- Batch Normalization
- MaxPooling2D (2×2)
- Dropout (0.25)

### Dense Layers Include:
- Dense layer with ReLU activation
- Batch Normalization
- Dropout (0.5)

**Total Parameters**: ~26 million trainable parameters

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) GPU with CUDA support for faster training

### Step 1: Clone the Repository
```bash
git clone https://github.com/Kunal-2301/Skin-Lesion-Classification.git
cd skin-lesion-classification
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
1. Download the HAM10000 dataset from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Extract the files to the project directory
3. Ensure you have:
   - `HAM10000_metadata.csv`
   - `HAM10000_images_part_1/` folder
   - `HAM10000_images_part_2/` folder

### Step 5 (Optional): Already Trained Model
1. To Download the Pre-Trained Model of this Project, visit [this link](https://drive.google.com/drive/folders/1QHJkjk5WVS-Ss3aYrkodBJjb5Frg9S0Z?usp=share_link).
2. Download both files
3. Add it to your Project Directory.


## Usage

### Training the Model

1. Open the Jupyter Notebook:
```bash
jupyter notebook skin_lesion_classification.ipynb
```

2. Update the dataset paths in the notebook:
```python
metadata_path = 'HAM10000_metadata.csv'
images_path_1 = 'HAM10000_images_part_1/'
images_path_2 = 'HAM10000_images_part_2/'
```

3. Run all cells to:
   - Load and preprocess data
   - Train the model
   - Evaluate performance
   - Save the trained model

**Training Time**: 2-4 hours with GPU, 8-12 hours with CPU

### Running the Streamlit App

1. Ensure you have the trained model files:
   - `skin_lesion_classifier_final.h5`
   - `label_encoder.pkl`
   - `lesion_type_dict.pkl`

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`

4. Upload a dermatoscopic image and click "Classify Lesion"

### Using Pre-trained Model (If Available)

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 92% |
| **Test AUC** | 0.99 |
| **Training Time** | ~3 hours (GPU) |
| **Inference Time** | <1 second |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| akiec | 0.99 | 0.96 | 0.97 | 1005 |
| bcc | 0.99 | 1.00 | 0.99 | 1006 |
| bkl | 0.91 | 0.82 | 0.86 | 1006 |
| df | 0.98 | 0.99 | 0.99 | 1006 |
| mel | 0.81 | 0.80 | 0.80 | 1006 |
| nv | 0.77 | 0.86 | 0.81 | 1006 |
| vasc | 1.00 | 1.00 | 1.00 | 1006 |

### Key Insights
- **Outstanding performance** on vasc (100% precision & recall)
- **Excellent results** on rare classes (bcc, df, akiec - all >96%)
- **Strong overall accuracy** at 91.8%
- Moderate performance on common classes (nv, mel - 77-81% precision)
- Some confusion between melanocytic lesions (mel ↔ nv) - expected due to visual similarity
- Best suited for dermatoscopic images (not regular smartphone photos)

## Project Structure

```
skin-lesion-classification/
│
├── HAM10000_images_part_1/          # Dataset images (part 1)
├── HAM10000_images_part_2/          # Dataset images (part 2)
├── HAM10000_metadata.csv            # Dataset metadata
│
├── skin_lesion_classification.ipynb # Training notebook
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
│
├── skin_lesion_classifier_final.h5  # Trained model (generated)
├── label_encoder.pkl                # Label encoder (generated)
├── lesion_type_dict.pkl             # Class dictionary (generated)
└── best_skin_lesion_model.h5        # Best checkpoint (generated)
```

## Technologies Used

### Core Frameworks
- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **Python 3.8+**: Programming language

### Data Processing
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **OpenCV**: Image processing
- **Pillow (PIL)**: Image handling

### Machine Learning
- **Scikit-learn**: ML utilities (train-test split, metrics, label encoding)
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical plotting

### Development Tools
- **Jupyter Notebook**: Interactive development
- **Git**: Version control
- **pip**: Package management

## Future Improvements

### Short-term Enhancements
- [ ] Add Grad-CAM visualization to highlight important regions
- [ ] Implement data augmentation for test-time augmentation
- [ ] Add model confidence calibration
- [ ] Create API endpoint using FastAPI
- [ ] Add batch prediction capability

### Long-term Enhancements
- [ ] Implement transfer learning (EfficientNet, ResNet)
- [ ] Create ensemble model for better accuracy
- [ ] Add explainability features (LIME, SHAP)
- [ ] Develop mobile application (TensorFlow Lite)
- [ ] Incorporate patient metadata (age, gender, location)
- [ ] Multi-modal learning (dermoscopic + clinical images)
- [ ] Active learning for continuous improvement

### Performance Optimization
- [ ] Model quantization for faster inference
- [ ] Edge deployment optimization
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] Docker containerization
- [ ] CI/CD pipeline setup

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Write clear commit messages
- Add tests for new features
- Update documentation as needed
- Follow PEP 8 style guide for Python code

## Important Disclaimer

**This project is for educational and research purposes only.**

- Not intended for actual medical diagnosis
- Not a replacement for professional medical advice
- Should not be used for clinical decisions
- Demonstrates AI/ML applications in healthcare
- Learning tool for deep learning concepts


