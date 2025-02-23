import pytest
import os
import tempfile
from pathlib import Path

@pytest.fixture(scope="session")
def test_dir():
    """Create a temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture(scope="session")
def env_setup():
    """Set up environment variables for testing"""
    os.environ["TESTING"] = "true"
    os.environ["API_KEY"] = "test_key"
    os.environ["EMAIL_HOST"] = "localhost"
    os.environ["EMAIL_PORT"] = "587"
    os.environ["SLACK_API_TOKEN"] = "test_token"
    yield
    # Clean up
    for key in ["TESTING", "API_KEY", "EMAIL_HOST", "EMAIL_PORT", "SLACK_API_TOKEN"]:
        os.environ.pop(key, None)

@pytest.fixture
def sample_image(test_dir):
    """Create a sample image file for testing"""
    import numpy as np
    from PIL import Image
    
    # Create a simple gradient image
    arr = np.linspace(0, 255, 224*224).reshape(224, 224).astype('uint8')
    img = Image.fromarray(arr)
    
    # Save to temporary file
    img_path = test_dir / "test_image.jpg"
    img.save(img_path)
    
    yield img_path
    
    # Cleanup
    if img_path.exists():
        img_path.unlink()
