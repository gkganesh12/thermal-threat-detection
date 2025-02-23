import pandas as pd
import torch
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from train_thermal import ThermalClassifier
from predict_thermal import load_model, predict

class ThreatAnalyzer:
    def __init__(self):
        # Load the thermal model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.thermal_model = load_model("best_thermal_model.pth", num_classes=3, device=self.device)
        
        # Load NLP model for threat analysis
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
        
        # Define threat levels
        self.threat_levels = ['LOW', 'MEDIUM', 'HIGH']
    
    def generate_context_description(self, thermal_class, location, confidence):
        """Generate contextual description based on thermal detection"""
        time_of_day = datetime.now().strftime("%H:%M")
        
        contexts = {
            0: f"Thermal signature detected at {location} at {time_of_day}. Low heat pattern observed.",
            1: f"Elevated thermal signature detected at {location} at {time_of_day}. Medium heat pattern observed.",
            2: f"Critical thermal signature detected at {location} at {time_of_day}. High heat pattern observed."
        }
        
        return contexts.get(thermal_class, "Unknown thermal pattern detected.")
    
    def analyze_thermal_image(self, image_path):
        """Analyze thermal image and return prediction"""
        pred_class, confidence = predict(self.thermal_model, image_path, self.device)
        return pred_class, confidence
    
    def analyze_text(self, text):
        """Analyze text for threat context"""
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.nlp_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0][pred_class].item()
        
        return pred_class, confidence
    
    def combine_analyses(self, thermal_class, thermal_conf, text_class, text_conf):
        """Combine thermal and text analyses for final threat assessment"""
        # Weight the predictions (can be adjusted based on reliability of each model)
        thermal_weight = 0.6
        text_weight = 0.4
        
        # Combine scores
        combined_score = (thermal_class * thermal_conf * thermal_weight + 
                         text_class * text_conf * text_weight)
        
        # Normalize to threat levels
        if combined_score > 1.5:
            return 'HIGH'
        elif combined_score > 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def analyze_threat(self, image_path, location):
        """Complete threat analysis combining thermal and contextual information"""
        # Analyze thermal image
        thermal_class, thermal_conf = self.analyze_thermal_image(image_path)
        
        # Generate and analyze context description
        context = self.generate_context_description(thermal_class, location, thermal_conf)
        text_class, text_conf = self.analyze_text(context)
        
        # Combine analyses
        final_threat_level = self.combine_analyses(thermal_class, thermal_conf,
                                                 text_class, text_conf)
        
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'location': location,
            'thermal_class': thermal_class,
            'thermal_confidence': f"{thermal_conf:.2%}",
            'context_description': context,
            'text_class': text_class,
            'text_confidence': f"{text_conf:.2%}",
            'final_threat_level': final_threat_level
        }

def generate_threat_report(analyzer, image_paths, output_file="enhanced_threat_report.csv"):
    """Generate comprehensive threat report with both thermal and NLP analysis"""
    reports = []
    
    for image_path in image_paths:
        location = image_path.split('/')[-2]  # Extract location from path
        try:
            analysis = analyzer.analyze_threat(image_path, location)
            reports.append(analysis)
        except Exception as e:
            print(f"Error analyzing {image_path}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(reports)
    
    # Sort by threat level
    threat_level_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    df['threat_level_rank'] = df['final_threat_level'].map(threat_level_order)
    df = df.sort_values('threat_level_rank').drop('threat_level_rank', axis=1)
    
    # Save report
    df.to_csv(output_file, index=False)
    
    # Print summary
    print("\nEnhanced Threat Report Summary:")
    print("-" * 50)
    print(f"Total incidents analyzed: {len(df)}")
    print("\nThreat Level Distribution:")
    print(df['final_threat_level'].value_counts())
    print("\nLocation Distribution:")
    print(df['location'].value_counts())
    print(f"\nReport saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    # Initialize analyzer
    print("Initializing Threat Analyzer...")
    analyzer = ThreatAnalyzer()
    
    # Get list of images to analyze
    import glob
    image_paths = glob.glob("example_dataset/val/**/*.jpg", recursive=True)
    
    # Generate enhanced threat report
    print("\nGenerating Enhanced Threat Report...")
    df = generate_threat_report(analyzer, image_paths)
    
    # Display example of detailed analysis
    print("\nExample Detailed Analysis:")
    print("-" * 50)
    example = df.iloc[0]
    for key, value in example.items():
        print(f"{key}: {value}")
