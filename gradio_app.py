"""
================================================================================
CHEST X-RAY PNEUMONIA CLASSIFIER - GRADIO WEB INTERFACE
================================================================================

Interactive web application for chest X-ray analysis.
Upload an X-ray image and get instant pneumonia detection results.

Author: Victor Prefa
Model: Vision Transformer (ViT) trained on Roboflow

Usage:
    python gradio_app.py
    
    Then open http://localhost:7860 in your browser

================================================================================
"""

import os
import gradio as gr
from inference_sdk import InferenceHTTPClient
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Configuration
MODEL_ID = "chest-xray-pneumonia-s5ed5/1"
API_URL = "https://serverless.roboflow.com"

# Get API key from environment variable
# Set it with: export ROBOFLOW_API_KEY="your_key"
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")

# Initialize client (will be done when app starts)
CLIENT = None


# ============================================================================
# CLASSIFICATION FUNCTION
# ============================================================================

def initialize_client():
    """Initialize the Roboflow client."""
    global CLIENT
    if CLIENT is None:
        if API_KEY == "YOUR_API_KEY_HERE":
            raise ValueError("Please set ROBOFLOW_API_KEY environment variable")
        CLIENT = InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)
    return CLIENT


def classify_xray(image):
    """
    Classify a chest X-ray image.
    
    Args:
        image: PIL Image or numpy array from Gradio
        
    Returns:
        tuple: (label_dict, status_message)
    """
    if image is None:
        return None, "⚠️ Please upload an image"
    
    try:
        # Initialize client
        client = initialize_client()
        
        # Save image temporarily
        temp_path = "/tmp/temp_xray.jpg"
        if isinstance(image, Image.Image):
            image.save(temp_path)
        else:
            Image.fromarray(image).save(temp_path)
        
        # Run inference
        result = client.infer(temp_path, model_id=MODEL_ID)
        
        # Parse results
        prediction = result.get("top", "Unknown")
        confidence = result.get("confidence", 0.0)
        
        # Get all class probabilities
        predictions = result.get("predictions", [])
        label_dict = {}
        
        for pred in predictions:
            class_name = pred.get("class", "Unknown")
            class_conf = pred.get("confidence", 0.0)
            label_dict[class_name] = class_conf
        
        # If only one prediction, add the other class
        if "Normal" not in label_dict:
            label_dict["Normal"] = 1.0 - confidence if prediction == "Pneumonia" else confidence
        if "Pneumonia" not in label_dict:
            label_dict["Pneumonia"] = 1.0 - confidence if prediction == "Normal" else confidence
        
        # Generate status message
        if prediction == "Pneumonia":
            if confidence > 0.8:
                status = f"🔴 HIGH PROBABILITY: Pneumonia detected ({confidence:.1%} confidence)"
            elif confidence > 0.6:
                status = f"🟠 MODERATE PROBABILITY: Possible pneumonia ({confidence:.1%} confidence)"
            else:
                status = f"🟡 LOW PROBABILITY: Uncertain - review recommended ({confidence:.1%} confidence)"
        else:
            if confidence > 0.8:
                status = f"🟢 Normal chest X-ray ({confidence:.1%} confidence)"
            else:
                status = f"🟡 Likely normal, but review recommended ({confidence:.1%} confidence)"
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return label_dict, status
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS for better appearance
custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
"""

# Create the interface
def create_interface():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(css=custom_css, title="Chest X-Ray Classifier") as demo:
        
        # Header
        gr.Markdown("""
        # 🫁 Chest X-Ray Pneumonia Classifier
        
        Upload a chest X-ray image to detect pneumonia using AI.
        
        **Model:** Vision Transformer (ViT) | **Platform:** Roboflow
        """)
        
        # Disclaimer
        gr.Markdown("""
        <div class="disclaimer">
        ⚠️ <strong>MEDICAL DISCLAIMER:</strong> This tool is for educational and research purposes only. 
        It is NOT approved for clinical diagnosis. Always consult a qualified healthcare professional 
        for medical advice and diagnosis.
        </div>
        """)
        
        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Chest X-Ray",
                    type="pil",
                    height=400
                )
                
                submit_btn = gr.Button(
                    "🔍 Analyze X-Ray",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### 📋 Instructions:
                1. Upload a chest X-ray image (JPG, PNG)
                2. Click "Analyze X-Ray"
                3. View the prediction results
                
                ### ✅ Supported Formats:
                - JPEG / JPG
                - PNG
                - BMP
                - WebP
                """)
            
            # Right column - Output
            with gr.Column(scale=1):
                output_label = gr.Label(
                    label="Classification Results",
                    num_top_classes=2
                )
                
                output_status = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False
                )
                
                gr.Markdown("""
                ### 📊 Understanding Results:
                
                | Indicator | Meaning |
                |-----------|---------|
                | 🟢 Green | Normal - No pneumonia detected |
                | 🟡 Yellow | Uncertain - Review recommended |
                | 🟠 Orange | Moderate probability |
                | 🔴 Red | High probability of pneumonia |
                
                **Confidence Score:** Higher = more certain prediction
                """)
        
        # Example images (if available)
        gr.Markdown("---")
        gr.Markdown("### 📁 Example Images")
        gr.Markdown("*Upload your own chest X-ray images to test the classifier*")
        
        # Footer
        gr.Markdown("""
        ---
        
        ### 🔬 About This Model
        
        - **Architecture:** Vision Transformer (ViT)
        - **Training Data:** Chest X-Ray Images (Pneumonia)
        - **Classes:** Normal, Pneumonia
        - **Platform:** Roboflow
        
        ### 👨‍⚕️ Developer
        
        **Victor Prefa** - Medical Doctor & Data Scientist
        
        ---
        
        *This project is part of the AI in Healthcare portfolio demonstrating 
        the application of computer vision in medical imaging.*
        """)
        
        # Connect the button to the function
        submit_btn.click(
            fn=classify_xray,
            inputs=[input_image],
            outputs=[output_label, output_status]
        )
        
        # Also trigger on image upload
        input_image.change(
            fn=classify_xray,
            inputs=[input_image],
            outputs=[output_label, output_status]
        )
    
    return demo


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Launch the Gradio application."""
    
    print("=" * 60)
    print("   CHEST X-RAY PNEUMONIA CLASSIFIER")
    print("   Web Interface")
    print("=" * 60)
    
    # Check API key
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️  WARNING: API key not set!")
        print("   Set it with: export ROBOFLOW_API_KEY='your_key'")
        print("   Or edit the API_KEY variable in this file.\n")
    
    # Create and launch interface
    demo = create_interface()
    
    print("\n🚀 Launching web interface...")
    print("   Open http://localhost:7860 in your browser\n")
    
    # Launch with share=True to get a public URL
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates a public URL
        show_error=True
    )


if __name__ == "__main__":
    main()
