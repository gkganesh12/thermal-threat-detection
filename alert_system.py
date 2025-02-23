import smtplib
import os
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import requests
from typing import Dict, List, Optional
import logging

class AlertSystem:
    def __init__(self, config_path: str = "alert_config.json"):
        """Initialize the alert system with configuration"""
        self.config_path = config_path
        self.config = self._load_config()
        
        # Setup logging
        logging.basicConfig(
            filename='alert_system.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> dict:
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Create default config
                default_config = {
                    "email": {
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "use_tls": True,
                        "sender_email": "",
                        "sender_password": "",
                        "default_recipients": []
                    },
                    "slack": {
                        "webhook_url": "",
                        "default_channel": "#security-alerts"
                    },
                    "telegram": {
                        "bot_token": "",
                        "chat_ids": []
                    },
                    "alert_levels": {
                        "LOW": {"color": "green", "icon": "ℹ️"},
                        "MEDIUM": {"color": "yellow", "icon": "⚠️"},
                        "HIGH": {"color": "red", "icon": "🚨"},
                        "CRITICAL": {"color": "purple", "icon": "⛔"}
                    }
                }
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            raise

    def format_alert_message(self, 
                           threat_level: str, 
                           location: str, 
                           message: str, 
                           details: Optional[Dict] = None) -> Dict[str, str]:
        """Format alert message for different platforms"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_config = self.config["alert_levels"].get(threat_level.upper(), 
                                                     {"color": "blue", "icon": "ℹ️"})
        
        # Basic message
        basic_msg = (
            f"{alert_config['icon']} *{threat_level.upper()} Alert*\n"
            f"Location: {location}\n"
            f"Time: {timestamp}\n"
            f"Message: {message}\n"
        )
        
        # Add details if provided
        if details:
            basic_msg += "\nDetails:\n" + "\n".join(
                f"- {k}: {v}" for k, v in details.items()
            )
        
        # HTML version for email
        html_msg = basic_msg.replace('\n', '<br>')
        html_msg = f'<div style="color: {alert_config["color"]};">{html_msg}</div>'
        
        return {
            "plain": basic_msg,
            "html": html_msg,
            "slack": basic_msg,  # Slack supports basic markdown
            "telegram": basic_msg
        }

    def send_email_alert(self, 
                        subject: str, 
                        message_dict: Dict[str, str], 
                        recipients: List[str] = None) -> bool:
        """Send email alert"""
        if not recipients:
            recipients = self.config["email"]["default_recipients"]
        
        if not recipients or not self.config["email"]["sender_email"]:
            self.logger.warning("Email configuration incomplete")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.config["email"]["sender_email"]
            msg['To'] = ", ".join(recipients)
            
            # Attach both plain text and HTML versions
            msg.attach(MIMEText(message_dict["plain"], 'plain'))
            msg.attach(MIMEText(message_dict["html"], 'html'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(
                self.config["email"]["smtp_server"],
                self.config["email"]["smtp_port"]
            )
            
            if self.config["email"]["use_tls"]:
                server.starttls()
            
            server.login(
                self.config["email"]["sender_email"],
                self.config["email"]["sender_password"]
            )
            
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email alert sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email: {str(e)}")
            return False

    def send_slack_alert(self, message_dict: Dict[str, str], channel: str = None) -> bool:
        """Send Slack alert"""
        if not self.config["slack"]["webhook_url"]:
            self.logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            webhook_url = self.config["slack"]["webhook_url"]
            channel = channel or self.config["slack"]["default_channel"]
            
            payload = {
                "channel": channel,
                "text": message_dict["slack"]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent to {channel}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {str(e)}")
            return False

    def send_telegram_alert(self, message_dict: Dict[str, str]) -> bool:
        """Send Telegram alert"""
        if not self.config["telegram"]["bot_token"] or not self.config["telegram"]["chat_ids"]:
            self.logger.warning("Telegram configuration incomplete")
            return False
        
        try:
            bot_token = self.config["telegram"]["bot_token"]
            base_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            success = True
            for chat_id in self.config["telegram"]["chat_ids"]:
                payload = {
                    "chat_id": chat_id,
                    "text": message_dict["telegram"],
                    "parse_mode": "Markdown"
                }
                
                response = requests.post(base_url, json=payload)
                if not response.ok:
                    self.logger.error(f"Error sending Telegram alert to {chat_id}: {response.text}")
                    success = False
            
            if success:
                self.logger.info(f"Telegram alerts sent to {len(self.config['telegram']['chat_ids'])} chats")
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {str(e)}")
            return False

    def send_alert(self, 
                  threat_level: str, 
                  location: str, 
                  message: str, 
                  details: Optional[Dict] = None,
                  platforms: List[str] = None,
                  email_recipients: List[str] = None,
                  slack_channel: str = None) -> Dict[str, bool]:
        """
        Send alert to all specified platforms
        
        Args:
            threat_level: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            location: Where the threat was detected
            message: Alert message
            details: Additional details as key-value pairs
            platforms: List of platforms to send alert to (email, slack, telegram)
            email_recipients: Optional list of email recipients
            slack_channel: Optional Slack channel
            
        Returns:
            Dictionary with status of each platform's alert
        """
        if not platforms:
            platforms = ["email", "slack", "telegram"]
        
        # Format message for all platforms
        message_dict = self.format_alert_message(threat_level, location, message, details)
        
        results = {}
        
        # Send to each platform
        if "email" in platforms:
            results["email"] = self.send_email_alert(
                f"{threat_level.upper()} Alert: {message[:50]}...",
                message_dict,
                email_recipients
            )
        
        if "slack" in platforms:
            results["slack"] = self.send_slack_alert(message_dict, slack_channel)
        
        if "telegram" in platforms:
            results["telegram"] = self.send_telegram_alert(message_dict)
        
        # Log overall results
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        self.logger.info(
            f"Alert sent to {success_count}/{total_count} platforms successfully"
        )
        
        return results

if __name__ == "__main__":
    # Example usage
    alert_system = AlertSystem()
    
    # Example alert
    alert_details = {
        "Thermal Reading": "45.2°C",
        "Confidence": "89%",
        "Movement Direction": "North-East",
        "Camera ID": "CAM-123"
    }
    
    results = alert_system.send_alert(
        threat_level="HIGH",
        location="Building A, Gate 3",
        message="Unauthorized thermal signature detected with suspicious movement pattern",
        details=alert_details,
        platforms=["email", "slack", "telegram"]
    )
    
    print("Alert Status:", results)
