import pytest
from alert_system import AlertSystem
from unittest.mock import patch, MagicMock

@pytest.fixture
def alert_system():
    return AlertSystem()

def test_alert_initialization(alert_system):
    assert isinstance(alert_system, AlertSystem)

@patch('alert_system.requests.post')
def test_slack_notification(mock_post, alert_system):
    mock_post.return_value = MagicMock(status_code=200)
    result = alert_system.send_slack_alert("Test message", "Test Location")
    assert result is True
    mock_post.assert_called_once()

@patch('alert_system.smtplib.SMTP')
def test_email_notification(mock_smtp, alert_system):
    mock_smtp_instance = MagicMock()
    mock_smtp.return_value = mock_smtp_instance
    
    result = alert_system.send_email_alert("Test message", "Test Location")
    assert result is True
    mock_smtp_instance.send_message.assert_called_once()
