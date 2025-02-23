import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from train_thermal import ThermalClassifier

def load_model(model_path, num_classes=2, device='cpu'):
    """Load the trained model"""
    model = ThermalClassifier(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load state dict based on the checkpoint format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess an image for model inference"""
    # Define the same transforms used during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

def predict(model, image_path, device='cpu'):
    """Make a prediction on a single image"""
    # Preprocess the image
    image = preprocess_image(image_path)
    image = image.to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    model_path = "best_thermal_model.pth"  # Use the best model from training
    num_classes = 3  # We have 3 classes in our dataset
    model = load_model(model_path, num_classes=num_classes, device=device)
    print("Model loaded successfully!")
    
    # Example usage with a test image
    test_image_path = "example_dataset/val/class_0/img_0.jpg"  # Using a validation image
    try:
        predicted_class, confidence = predict(model, test_image_path, device)
        print(f"\nPrediction Results:")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print("Please make sure to provide a valid image path.")
