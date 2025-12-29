"""
================================================================================
CHEST X-RAY PNEUMONIA CLASSIFIER
================================================================================

A Vision Transformer (ViT) based classifier for detecting pneumonia 
from chest X-ray images.

Author: Victor Prefa
Model: Trained on Roboflow
Model ID: chest-xray-pneumonia-s5ed5/1

Usage:
    python chest_xray_classifier.py --image path/to/xray.jpg
    python chest_xray_classifier.py --folder path/to/xrays/
    
================================================================================
"""

import os
import argparse
from datetime import datetime
from inference_sdk import InferenceHTTPClient

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model Configuration
MODEL_ID = "chest-xray-pneumonia-s5ed5/1"
API_URL = "https://serverless.roboflow.com"

# Get API key from environment variable or set directly
# For security, use environment variable: export ROBOFLOW_API_KEY="your_key"
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")


# ============================================================================
# CHEST X-RAY CLASSIFIER CLASS
# ============================================================================

class ChestXrayClassifier:
    """
    Chest X-Ray Pneumonia Classifier using Roboflow-trained Vision Transformer.
    
    Attributes:
        api_key (str): Roboflow API key
        model_id (str): Model identifier
        client: Inference HTTP client
    """
    
    def __init__(self, api_key=None, model_id=MODEL_ID):
        """
        Initialize the classifier.
        
        Args:
            api_key (str): Roboflow API key. If None, uses environment variable.
            model_id (str): Model ID for inference.
        """
        self.api_key = api_key or API_KEY
        self.model_id = model_id
        
        # Validate API key
        if self.api_key == "YOUR_API_KEY_HERE":
            raise ValueError(
                "Please set your Roboflow API key!\n"
                "Option 1: Set environment variable ROBOFLOW_API_KEY\n"
                "Option 2: Pass api_key parameter to ChestXrayClassifier()"
            )
        
        # Initialize client
        self.client = InferenceHTTPClient(
            api_url=API_URL,
            api_key=self.api_key
        )
        
        print("✓ Chest X-Ray Classifier initialized")
        print(f"  Model: {self.model_id}")
    
    def predict(self, image_path):
        """
        Predict whether a chest X-ray shows pneumonia or is normal.
        
        Args:
            image_path (str): Path to the chest X-ray image
            
        Returns:
            dict: Prediction results with keys:
                - image: Path to image
                - prediction: "Normal" or "Pneumonia"
                - confidence: Confidence score (0-1)
                - inference_time: Time taken for inference
        """
        # Validate file exists
        if not os.path.exists(image_path):
            return {
                "image": image_path,
                "prediction": "Error",
                "confidence": 0.0,
                "error": f"File not found: {image_path}"
            }
        
        try:
            # Run inference
            result = self.client.infer(image_path, model_id=self.model_id)
            
            return {
                "image": image_path,
                "prediction": result.get("top", "Unknown"),
                "confidence": result.get("confidence", 0.0),
                "inference_time": result.get("time", 0.0),
                "all_predictions": result.get("predictions", [])
            }
            
        except Exception as e:
            return {
                "image": image_path,
                "prediction": "Error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def predict_batch(self, folder_path):
        """
        Predict multiple chest X-ray images in a folder.
        
        Args:
            folder_path (str): Path to folder containing X-ray images
            
        Returns:
            list: List of prediction results
        """
        results = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')
        
        # Get all image files
        if not os.path.isdir(folder_path):
            print(f"Error: {folder_path} is not a valid directory")
            return results
        
        image_files = [
            f for f in os.listdir(folder_path) 
            if f.lower().endswith(valid_extensions)
        ]
        
        print(f"\nProcessing {len(image_files)} images...")
        print("-" * 50)
        
        for filename in image_files:
            image_path = os.path.join(folder_path, filename)
            result = self.predict(image_path)
            results.append(result)
            
            # Print progress
            pred = result.get('prediction', 'Error')
            conf = result.get('confidence', 0)
            status = "✓" if pred != "Error" else "✗"
            print(f"  {status} {filename}: {pred} ({conf:.1%})")
        
        return results
    
    def generate_report(self, results, save_path=None):
        """
        Generate a summary report from batch predictions.
        
        Args:
            results (list): List of prediction results
            save_path (str): Optional path to save report
            
        Returns:
            dict: Summary statistics
        """
        total = len(results)
        if total == 0:
            print("No results to report")
            return {}
        
        # Count predictions
        normal_count = sum(1 for r in results if r.get('prediction') == 'Normal')
        pneumonia_count = sum(1 for r in results if r.get('prediction') == 'Pneumonia')
        error_count = sum(1 for r in results if r.get('prediction') == 'Error')
        
        # Calculate average confidence
        confidences = [r.get('confidence', 0) for r in results if r.get('prediction') != 'Error']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_images": total,
            "normal_count": normal_count,
            "pneumonia_count": pneumonia_count,
            "error_count": error_count,
            "average_confidence": avg_confidence,
            "pneumonia_rate": pneumonia_count / (total - error_count) if (total - error_count) > 0 else 0
        }
        
        # Print report
        print("\n" + "=" * 60)
        print("           CHEST X-RAY ANALYSIS REPORT")
        print("=" * 60)
        print(f"""
    Timestamp: {report['timestamp']}
    
    Total Images Analyzed: {report['total_images']}
    
    Results:
    ├── Normal:     {report['normal_count']} ({report['normal_count']/total*100:.1f}%)
    ├── Pneumonia:  {report['pneumonia_count']} ({report['pneumonia_count']/total*100:.1f}%)
    └── Errors:     {report['error_count']} ({report['error_count']/total*100:.1f}%)
    
    Average Confidence: {report['average_confidence']:.1%}
    Pneumonia Detection Rate: {report['pneumonia_rate']:.1%}
        """)
        print("=" * 60)
        
        # Clinical warning
        if pneumonia_count > 0:
            print("\n⚠️  CLINICAL NOTE:")
            print(f"    {pneumonia_count} image(s) flagged for potential pneumonia.")
            print("    Please review flagged images with a qualified radiologist.")
            print("    This AI tool is for screening assistance only.\n")
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write("CHEST X-RAY ANALYSIS REPORT\n")
                f.write("=" * 60 + "\n\n")
                for key, value in report.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n\nDETAILED RESULTS:\n")
                f.write("-" * 60 + "\n")
                for r in results:
                    f.write(f"{r.get('image', 'Unknown')}: {r.get('prediction', 'Error')} ({r.get('confidence', 0):.1%})\n")
            print(f"Report saved to: {save_path}")
        
        return report


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main function for command line usage."""
    
    parser = argparse.ArgumentParser(
        description="Chest X-Ray Pneumonia Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Predict single image
    python chest_xray_classifier.py --image chest_xray.jpg
    
    # Process folder of images
    python chest_xray_classifier.py --folder ./xrays/
    
    # Save report to file
    python chest_xray_classifier.py --folder ./xrays/ --output report.txt
        """
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Path to a single chest X-ray image'
    )
    
    parser.add_argument(
        '--folder', '-f',
        type=str,
        help='Path to folder containing chest X-ray images'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Path to save the report (for batch processing)'
    )
    
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        help='Roboflow API key (or set ROBOFLOW_API_KEY environment variable)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.folder:
        parser.print_help()
        print("\nError: Please provide --image or --folder argument")
        return
    
    # Initialize classifier
    try:
        classifier = ChestXrayClassifier(api_key=args.api_key)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("    CHEST X-RAY PNEUMONIA CLASSIFIER")
    print("    Powered by Vision Transformer (ViT)")
    print("=" * 60)
    
    # Process single image
    if args.image:
        print(f"\nAnalyzing: {args.image}")
        result = classifier.predict(args.image)
        
        if result.get('prediction') != 'Error':
            print(f"\n┌─────────────────────────────────────┐")
            print(f"│ Prediction: {result['prediction']:>20s}  │")
            print(f"│ Confidence: {result['confidence']:>19.1%}  │")
            print(f"└─────────────────────────────────────┘")
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")
    
    # Process folder
    if args.folder:
        results = classifier.predict_batch(args.folder)
        classifier.generate_report(results, save_path=args.output)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
