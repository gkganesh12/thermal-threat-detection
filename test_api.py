import requests
import json
from pathlib import Path

def test_threat_detection(
    api_url: str,
    image_path: str,
    message: str,
    location: str,
    additional_details: dict = None
):
    """Test the threat detection API"""
    
    # Prepare the multipart form data
    files = {
        'file': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')
    }
    
    # Prepare the request data
    request_data = {
        'message': message,
        'location': location,
        'additional_details': additional_details or {},
        'alert_platforms': ['email', 'slack']
    }
    
    try:
        # Send request to API
        response = requests.post(
            f"{api_url}/detect/",
            files=files,
            data={'request': json.dumps(request_data)}
        )
        
        # Check for errors
        if response.status_code == 422:
            print("Validation Error:")
            print(response.json())
            return None
            
        response.raise_for_status()
        
        # Print results
        result = response.json()
        print("\nThreat Detection Results:")
        print("-" * 50)
        print(f"Request ID: {result['request_id']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Location: {result['location']}")
        print(f"Threat Level: {result['threat_level']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Message: {result['message']}")
        print("\nAlerts Sent:")
        for platform, status in result['alerts_sent'].items():
            print(f"- {platform}: {'Success' if status else 'Failed'}")
        
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {str(e)}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None

if __name__ == "__main__":
    # Test parameters
    API_URL = "http://localhost:8080"
    IMAGE_PATH = "test_image.jpg"  # Update this path to an existing image file
    MESSAGE = "Suspicious thermal signature detected near secure area"
    LOCATION = "Building A, Secure Zone"
    ADDITIONAL_DETAILS = {
        "Security Level": "High",
        "Zone ID": "SZ-123",
        "Camera ID": "CAM-456"
    }
    
    # Run test
    result = test_threat_detection(
        API_URL,
        IMAGE_PATH,
        MESSAGE,
        LOCATION,
        ADDITIONAL_DETAILS
    )
