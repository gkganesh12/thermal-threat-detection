# Thermal Threat Detection System

A comprehensive system for detecting and analyzing thermal threats using deep learning and natural language processing.

## Features

- **Thermal Image Analysis**: CNN-based model for classifying thermal images into threat levels
- **NLP Context Analysis**: BERT-based model for analyzing textual descriptions
- **Multi-Platform Alerts**: Integrated notification system (Email, Slack, Telegram)
- **Real-time API**: FastAPI-based endpoint for threat detection
- **Interactive UI**: Streamlit-based chatbot interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/thermal-threat-detection.git
cd thermal-threat-detection
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Starting the API Server

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8080
```

### Starting the Chatbot UI

```bash
streamlit run chatbot_ui.py
```

### Running Tests

```bash
python test_api.py
```

## Project Structure

- `api_server.py`: FastAPI application for threat detection
- `chatbot_ui.py`: Streamlit-based user interface
- `threat_analysis.py`: Core threat analysis logic
- `alert_system.py`: Multi-platform notification system
- `data_preparation.py`: Data preprocessing utilities
- `train.py`: Model training scripts
- `predict.py`: Inference utilities
- `generate_report.py`: Report generation tools

## API Documentation

### POST /detect/

Analyzes a thermal image for potential threats.

**Request Body:**
- `file`: Thermal image file
- `message`: Description of the observation
- `location`: Location where the image was captured
- `additional_details`: Optional metadata
- `alert_platforms`: List of platforms to notify

**Response:**
```json
{
    "request_id": "string",
    "timestamp": "string",
    "threat_level": "string",
    "confidence": "float",
    "location": "string",
    "message": "string",
    "alerts_sent": {
        "email": "boolean",
        "slack": "boolean"
    }
}
```

## Configuration

The system can be configured using environment variables:

- `API_KEY`: Authentication key for the API
- `EMAIL_*`: Email notification settings
- `SLACK_*`: Slack integration settings
- `TELEGRAM_*`: Telegram bot settings
- `MODEL_PATH`: Path to trained model weights

## Development

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security

- API authentication using API keys
- Rate limiting to prevent abuse
- Secure file handling
- Environment-based configuration

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Authors

- Ganesh Khetawat(GK)

## Acknowledgments

- Thanks to all contributors
- Special thanks to the open-source community
