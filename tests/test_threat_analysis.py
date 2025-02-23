import pytest
import numpy as np
from threat_analysis import ThreatAnalyzer
from unittest.mock import patch, MagicMock

@pytest.fixture
def threat_analyzer():
    return ThreatAnalyzer()

def test_threat_analyzer_initialization(threat_analyzer):
    assert isinstance(threat_analyzer, ThreatAnalyzer)

@patch('threat_analysis.torch.load')
def test_load_model(mock_load, threat_analyzer):
    mock_load.return_value = MagicMock()
    threat_analyzer.load_model("dummy_path")
    mock_load.assert_called_once_with("dummy_path")

def test_preprocess_image():
    # Create a dummy image array
    image = np.random.rand(100, 100, 3).astype(np.uint8)
    
    with patch('threat_analysis.cv2.resize') as mock_resize:
        mock_resize.return_value = np.zeros((224, 224, 3))
        
        analyzer = ThreatAnalyzer()
        processed = analyzer.preprocess_image(image)
        
        assert processed.shape == (1, 3, 224, 224)  # Standard input shape
        mock_resize.assert_called_once()

@patch('threat_analysis.torch.no_grad')
def test_predict_threat_level(mock_no_grad, threat_analyzer):
    # Mock the model prediction
    threat_analyzer.model = MagicMock()
    threat_analyzer.model.return_value = MagicMock(detach=lambda: np.array([0.1, 0.8, 0.1]))
    
    # Create dummy input tensor
    dummy_input = np.zeros((1, 3, 224, 224))
    
    level, confidence = threat_analyzer.predict_threat_level(dummy_input)
    
    assert isinstance(level, str)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1
