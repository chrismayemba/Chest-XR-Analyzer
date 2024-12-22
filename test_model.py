import argparse
from models.swin_llava import SwinLLaVA
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_xray(image_path: str, prompt: str = None):
    """
    Analyze a chest X-ray image using the SwinLLaVA model
    Args:
        image_path: Path to the X-ray image
        prompt: Optional custom prompt for analysis
    """
    try:
        # Initialize model
        logger.info("Initializing model...")
        model = SwinLLaVA(config_path="config/model_config.yaml")
        
        # Use default prompt if none provided
        if not prompt:
            prompt = "Analyze this chest X-ray and describe any findings in detail."
        
        # Analyze image
        logger.info(f"Analyzing image: {image_path}")
        results = model.analyze_image(image_path, prompt=prompt)
        
        if results["success"]:
            # Print results in a formatted way
            print("\n=== Analysis Results ===")
            print("\nFull Report:")
            print("-" * 50)
            print(results["full_report"])
            print("\nKey Findings:")
            print("-" * 50)
            for finding in results["findings"]["key_findings"]:
                print(f"• {finding}")
            
            print("\nClass Predictions:")
            print("-" * 50)
            for class_name, prob in results["class_predictions"].items():
                print(f"{class_name}: {prob:.2%}")
            
            # Save results to file
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_analysis.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_path}")
            
        else:
            logger.error(f"Analysis failed: {results['error']}")
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze chest X-ray images using SwinLLaVA")
    parser.add_argument("image_path", type=str, help="Path to the X-ray image")
    parser.add_argument("--prompt", type=str, help="Custom prompt for analysis", default=None)
    
    args = parser.parse_args()
    analyze_xray(args.image_path, args.prompt)
