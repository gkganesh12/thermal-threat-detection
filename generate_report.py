import pandas as pd
import torch
from datetime import datetime
import os
from train_thermal import ThermalClassifier
from predict_thermal import load_model, predict
import glob

def get_threat_level(predicted_class, confidence):
    """Convert model prediction to threat level"""
    if predicted_class == 2:  # Assuming class 2 is highest threat
        return "High" if confidence > 0.6 else "Medium"
    elif predicted_class == 1:  # Assuming class 1 is medium threat
        return "Medium" if confidence > 0.6 else "Low"
    else:
        return "Low"

def generate_threat_report(model, image_directory, output_file="threat_report.csv"):
    """Generate a comprehensive threat report from thermal images"""
    # Initialize lists to store report data
    timestamps = []
    locations = []
    threat_levels = []
    confidences = []
    image_paths = []
    predicted_classes = []
    
    # Get current timestamp
    current_time = datetime.now()
    
    # Process all images in the directory
    for image_path in glob.glob(os.path.join(image_directory, "**/*.jpg"), recursive=True):
        try:
            # Get prediction
            pred_class, confidence = predict(model, image_path)
            
            # Extract location from image path
            location = os.path.basename(os.path.dirname(image_path))
            
            # Generate report data
            timestamps.append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
            locations.append(location)
            predicted_classes.append(pred_class)
            confidences.append(f"{confidence:.2%}")
            threat_level = get_threat_level(pred_class, confidence)
            threat_levels.append(threat_level)
            image_paths.append(image_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    # Create DataFrame
    data = {
        'Timestamp': timestamps,
        'Location': locations,
        'Predicted Class': predicted_classes,
        'Confidence': confidences,
        'Threat Level': threat_levels,
        'Image Path': image_paths
    }
    
    df = pd.DataFrame(data)
    
    # Sort by threat level (High > Medium > Low)
    threat_level_order = {'High': 0, 'Medium': 1, 'Low': 2}
    df['Threat Level Rank'] = df['Threat Level'].map(threat_level_order)
    df = df.sort_values('Threat Level Rank').drop('Threat Level Rank', axis=1)
    
    # Save report
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\nThreat Report Summary:")
    print("-" * 50)
    print(f"Total images processed: {len(df)}")
    print("\nThreat Level Distribution:")
    print(df['Threat Level'].value_counts())
    print("\nLocation Distribution:")
    print(df['Location'].value_counts())
    print(f"\nReport saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    model_path = "best_thermal_model.pth"
    num_classes = 3
    model = load_model(model_path, num_classes=num_classes, device=device)
    print("Model loaded successfully!")
    
    # Generate report
    image_directory = "example_dataset/val"  # Using validation set for demonstration
    df = generate_threat_report(model, image_directory)
