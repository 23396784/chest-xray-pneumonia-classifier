# ============================================================================
# CHEST X-RAY PNEUMONIA CLASSIFIER - GOOGLE COLAB NOTEBOOK
# ============================================================================
#
# This notebook demonstrates how to use the trained chest X-ray classifier.
# Run each cell in order.
#
# Author: Victor Prefa
# Model: Vision Transformer (ViT) on Roboflow
# ============================================================================

# -----------------------------------------------------------------------------
# CELL 1: Install Dependencies
# -----------------------------------------------------------------------------
# !pip install inference-sdk gradio pillow

# -----------------------------------------------------------------------------
# CELL 2: Configuration
# -----------------------------------------------------------------------------
"""
IMPORTANT: Replace 'YOUR_API_KEY_HERE' with your actual Roboflow API key.
Get your API key from: https://app.roboflow.com/settings/api
"""

API_KEY = "YOUR_API_KEY_HERE"  # <-- REPLACE THIS
MODEL_ID = "chest-xray-pneumonia-s5ed5/1"

# -----------------------------------------------------------------------------
# CELL 3: Initialize the Classifier
# -----------------------------------------------------------------------------
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

print("✓ Classifier initialized!")
print(f"  Model: {MODEL_ID}")

# -----------------------------------------------------------------------------
# CELL 4: Single Image Prediction Function
# -----------------------------------------------------------------------------
def predict_xray(image_path):
    """Predict if a chest X-ray shows pneumonia."""
    result = CLIENT.infer(image_path, model_id=MODEL_ID)
    
    prediction = result.get("top", "Unknown")
    confidence = result.get("confidence", 0.0)
    
    print(f"\n{'='*50}")
    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.1%}")
    print(f"{'='*50}\n")
    
    return result

# -----------------------------------------------------------------------------
# CELL 5: Upload and Test Your Own Image
# -----------------------------------------------------------------------------
from google.colab import files

print("Select a chest X-ray image to upload:")
uploaded = files.upload()

# Get filename and predict
filename = list(uploaded.keys())[0]
result = predict_xray(filename)

# -----------------------------------------------------------------------------
# CELL 6: Batch Prediction (Multiple Images)
# -----------------------------------------------------------------------------
def predict_batch(image_list):
    """Predict multiple images."""
    print("=" * 60)
    print("CHEST X-RAY CLASSIFICATION RESULTS")
    print("=" * 60)
    
    results = []
    for image_path in image_list:
        result = CLIENT.infer(image_path, model_id=MODEL_ID)
        prediction = result.get("top", "Unknown")
        confidence = result.get("confidence", 0.0)
        
        print(f"{image_path}: {prediction} ({confidence:.1%})")
        results.append({
            "image": image_path,
            "prediction": prediction,
            "confidence": confidence
        })
    
    # Summary
    pneumonia_count = sum(1 for r in results if r["prediction"] == "Pneumonia")
    normal_count = sum(1 for r in results if r["prediction"] == "Normal")
    
    print("\n" + "-" * 60)
    print(f"Summary: {pneumonia_count} Pneumonia, {normal_count} Normal")
    print("-" * 60)
    
    return results

# Upload multiple images
print("Select multiple chest X-ray images:")
uploaded = files.upload()
image_files = list(uploaded.keys())
results = predict_batch(image_files)

# -----------------------------------------------------------------------------
# CELL 7: Launch Gradio Web Interface (Optional)
# -----------------------------------------------------------------------------
# Uncomment and run to launch interactive web interface

"""
!pip install gradio

import gradio as gr
from PIL import Image
import os

def classify_for_gradio(image):
    # Save temp image
    temp_path = "/tmp/temp_xray.jpg"
    if isinstance(image, Image.Image):
        image.save(temp_path)
    else:
        Image.fromarray(image).save(temp_path)
    
    # Predict
    result = CLIENT.infer(temp_path, model_id=MODEL_ID)
    prediction = result.get("top", "Unknown")
    confidence = result.get("confidence", 0.0)
    
    # Return label dict for Gradio
    if prediction == "Pneumonia":
        return {"Pneumonia": confidence, "Normal": 1-confidence}
    else:
        return {"Normal": confidence, "Pneumonia": 1-confidence}

# Create interface
demo = gr.Interface(
    fn=classify_for_gradio,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=gr.Label(num_top_classes=2, label="Prediction"),
    title="🫁 Chest X-Ray Pneumonia Classifier",
    description="Upload a chest X-ray to detect pneumonia. **Disclaimer: For educational purposes only.**"
)

demo.launch(share=True)
"""

# -----------------------------------------------------------------------------
# CELL 8: Test with Sample Images from URL
# -----------------------------------------------------------------------------
# Download and test sample images

"""
# Download sample X-ray
!wget -O sample_xray.jpg "https://example.com/sample_xray.jpg"

# Predict
result = predict_xray("sample_xray.jpg")
"""

print("\n✓ Notebook ready!")
print("Run each cell in order to use the classifier.")
