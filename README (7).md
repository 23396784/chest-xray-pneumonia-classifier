# Chest X-Ray Pneumonia Classifier

AI-powered chest X-ray analysis using Computer Vision to detect pneumonia.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Roboflow](https://img.shields.io/badge/Roboflow-Trained-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

This project uses a **Vision Transformer (ViT)** model trained on chest X-ray images to classify them as either **Normal** or **Pneumonia**. The model was trained using Roboflow's AutoML platform and achieves high accuracy in detecting pneumonia from chest radiographs.

### Key Features
- 🔬 Medical image classification
- 🤖 Vision Transformer (ViT) architecture
- ☁️ Cloud-deployed API for inference
- 🌐 Interactive Gradio web interface
- 📊 Real-time predictions with confidence scores

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Architecture | Vision Transformer (ViT) |
| Input Size | 224 × 224 pixels |
| Classes | Normal, Pneumonia |
| Training Platform | Roboflow |

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/chest-xray-pneumonia-classifier.git
cd chest-xray-pneumonia-classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Create a `.env` file or set environment variable:
```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

Get your API key from: https://app.roboflow.com/settings/api

### 4. Run Inference
```bash
python chest_xray_classifier.py --image path/to/xray.jpg
```

### 5. Launch Web Interface
```bash
python gradio_app.py
```

## 📁 Project Structure

```
chest-xray-pneumonia-classifier/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── chest_xray_classifier.py     # Main inference script
├── gradio_app.py               # Interactive web interface
├── batch_inference.py          # Batch processing script
├── test_images/                # Sample test images
│   ├── normal/
│   └── pneumonia/
└── results/                    # Output predictions
```

## 🔧 Usage

### Single Image Prediction
```python
from chest_xray_classifier import ChestXrayClassifier

# Initialize classifier
classifier = ChestXrayClassifier(api_key="your_api_key")

# Predict
result = classifier.predict("chest_xray.jpg")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Batch Processing
```python
# Process entire folder
results = classifier.predict_batch("xray_folder/")
classifier.generate_report(results)
```

### Web Interface
```bash
python gradio_app.py
# Opens browser at http://localhost:7860
```

## 🏥 Clinical Disclaimer

⚠️ **IMPORTANT**: This model is for **educational and research purposes only**. 

- NOT approved for clinical diagnosis
- NOT a replacement for professional medical advice
- Always consult qualified healthcare professionals
- Model predictions should be verified by radiologists

## 📈 Model Details

### Training Data
- **Source**: Kaggle Chest X-Ray Pneumonia Dataset
- **Classes**: Normal, Pneumonia (bacterial/viral)
- **Preprocessing**: 
  - Resize to 224×224
  - Auto-orientation
  - Normalization

### Architecture
- **Model**: Vision Transformer (ViT)
- **Platform**: Roboflow
- **Model ID**: `chest-xray-pneumonia-s5ed5/1`

## 🛠️ Technologies Used

- **Python 3.8+**
- **Roboflow** - Model training and deployment
- **Gradio** - Web interface
- **Inference SDK** - API client
- **PIL/Pillow** - Image processing

## 👨‍⚕️ Author

**Victor Prefa**
- Medical Doctor & Data Scientist
- MSc Data Science & Business Analytics
- 17+ years clinical experience

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Roboflow](https://roboflow.com) for the training platform
- [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) for the dataset
- Paul Mooney for curating the original dataset

## 📚 References

1. Kermany, D. S., et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell, 172(5), 1122-1131.
2. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv:2010.11929.
