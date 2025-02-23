import streamlit as st
import requests
import json
from datetime import datetime
import io
from PIL import Image
import os

# Configure page settings
st.set_page_config(
    page_title="Thermal Detection Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8080"

def send_to_api(image_file, message, location):
    """Send the image and data to the thermal detection API"""
    try:
        files = {
            'file': ('image.jpg', image_file, 'image/jpeg')
        }
        
        request_data = {
            'message': message,
            'location': location,
            'additional_details': {
                'timestamp': datetime.now().isoformat(),
                'source': 'chatbot_ui'
            },
            'alert_platforms': ['email', 'slack']
        }
        
        response = requests.post(
            f"{st.session_state.api_url}/detect/",
            files=files,
            data={'request': json.dumps(request_data)}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                'error': f"API Error: {response.status_code}",
                'details': response.text
            }
    except Exception as e:
        return {'error': f"Request failed: {str(e)}"}

def format_api_response(response):
    """Format the API response for display"""
    if 'error' in response:
        return f"❌ Error: {response['error']}"
    
    result = "🔍 Analysis Results:\n\n"
    result += f"🎯 Threat Level: {response['threat_level']}\n"
    result += f"📊 Confidence: {response['confidence']:.2%}\n"
    result += f"📍 Location: {response['location']}\n"
    result += f"💬 Message: {response['message']}\n\n"
    
    result += "📢 Alerts sent to:\n"
    for platform, status in response['alerts_sent'].items():
        icon = "✅" if status else "❌"
        result += f"{icon} {platform}\n"
    
    return result

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API URL", value=st.session_state.api_url)
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
    
    st.markdown("---")
    st.markdown("""
    ### 🤖 How to use:
    1. Upload a thermal image
    2. Enter location details
    3. Describe what you see
    4. Get instant analysis!
    """)

# Main chat interface
st.title("🔥 Thermal Detection Chatbot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"], caption="Uploaded thermal image")
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What do you see in the thermal image?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # File uploader in the chat
    with st.chat_message("assistant"):
        st.markdown("Please upload a thermal image for analysis:")
        image_file = st.file_uploader("Choose a thermal image", type=['jpg', 'jpeg', 'png'])
        location = st.text_input("Location:", placeholder="e.g., Building A, Secure Zone")
        
        if image_file and location:
            # Process the image
            image = Image.open(image_file)
            
            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Send to API
            with st.spinner("Analyzing thermal image..."):
                response = send_to_api(img_byte_arr, prompt, location)
            
            # Format and display response
            result = format_api_response(response)
            st.markdown(result)
            
            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result,
                "image": image
            })

# Footer
st.markdown("---")
st.markdown("Powered by Thermal Detection API | Created with Streamlit")
