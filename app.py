import streamlit as st
import requests
import json
import os
import tempfile
from datetime import datetime
from google import genai
from google.genai import types
from config_azure import GEMINI_API_KEY, TRANSCRIPT_GEMINI_API_KEY

# Configure Streamlit page
st.set_page_config(
    page_title="Gemini Audio & Text Processor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark/light mode support
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .option-card {
        background: var(--background-color);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def download_audio_from_url(audio_url):
    """Download audio file from URL and return temporary file path"""
    try:
        import requests
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        return tmp_path
    except Exception as e:
        st.error(f"Error downloading audio from URL: {str(e)}")
        return None

def upload_audio_to_gemini(audio_path, display_name):
    """Upload audio file to Gemini and return mime_type and file_uri"""
    try:
        # Determine MIME type based on file extension
        file_extension = os.path.splitext(audio_path)[1].lower()
        mime_type_map = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.m4a': 'audio/mp4',
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac'
        }
        mime_type = mime_type_map.get(file_extension, 'audio/wav')
        
        # Upload file to Gemini
        upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={TRANSCRIPT_GEMINI_API_KEY}"
        
        with open(audio_path, 'rb') as audio_file:
            files = {'file': (display_name, audio_file, mime_type)}
            response = requests.post(upload_url, files=files)
        
        if response.status_code != 200:
            raise Exception(f"Failed to upload file. Response: {response.text}")
        
        response_json = response.json()
        file_uri = response_json.get('file', {}).get('uri')
        
        if not file_uri:
            raise Exception("No file URI returned from upload")
        
        return mime_type, file_uri
    
    except Exception as e:
        st.error(f"Error uploading audio file: {str(e)}")
        return None, None

def generate_transcript_from_audio(audio_path, display_name, user_prompt):
    """Generate transcript from audio using Gemini"""
    try:
        prompt = user_prompt
        
        mime_type, file_uri = upload_audio_to_gemini(audio_path, display_name)
        if not file_uri:
            return None
        
        # Generate content using Gemini
        generate_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={TRANSCRIPT_GEMINI_API_KEY}"
        
        generate_headers = {
            "Content-Type": "application/json"
        }
        
        generate_body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}
                    ]
                }
            ]
        }
        
        generate_response = requests.post(generate_url, headers=generate_headers, data=json.dumps(generate_body))
        
        if generate_response.status_code != 200:
            raise Exception(f"Failed to generate content. Response: {generate_response.text}")
        
        response_json = generate_response.json()
        
        # Extract transcript from response
        for candidate in response_json.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    return part["text"]
        
        return None
    
    except Exception as e:
        st.error(f"Error generating transcript: {str(e)}")
        return None

def process_text_with_gemini(prompt, transcript):
    """Process text using Gemini with prompt and transcript"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        model = "gemini-2.0-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f"Give Proper JSON output as mentioned in the format, that will make it easier to parse.\n{prompt}\n{transcript}")
                ]
            )
        ]
        
        config = types.GenerateContentConfig(
            safety_settings=[
                {
                    "category": types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": types.HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": types.HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": types.HarmBlockThreshold.BLOCK_NONE
                },
                {
                    "category": types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": types.HarmBlockThreshold.BLOCK_NONE
                }
            ]
        )
        
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        
        return response.text if response.text else None
    
    except Exception as e:
        st.error(f"Error processing with Gemini: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🎵 Gemini Audio & Text Processor</h1>', unsafe_allow_html=True)
    
    # Sidebar for mode selection
    st.sidebar.title("Choose Processing Mode")
    mode = st.sidebar.radio(
        "Select an option:",
        ["🎤 Audio Transcription", "📝 Prompt & Transcript Processing"],
        index=0
    )
    
    if mode == "🎤 Audio Transcription":
        st.markdown('<div class="option-card">', unsafe_allow_html=True)
        st.header("🎤 Audio Transcription")
        st.write("Upload an audio file and provide a prompt to generate a transcript using Gemini AI")
        
        # Prompt input for audio transcription
        st.subheader("📋 Enter Your Transcription Prompt")
        audio_prompt = st.text_area(
            "Transcription Prompt:",
            placeholder="Enter your transcription prompt here...",
            height=150,
            help="This prompt will guide how Gemini transcribes the audio file"
        )
        
        # Choose input method
        input_method = st.radio(
            "Choose input method:",
            ["📁 Upload Audio File", "🔗 Audio URL Link"],
            horizontal=True
        )
        
        uploaded_file = None
        audio_url = None
        
        if input_method == "📁 Upload Audio File":
            # File upload
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
                help="Supported formats: WAV, MP3, M4A, OGG, FLAC"
            )
        else:
            # URL input
            audio_url = st.text_input(
                "Enter Audio URL:",
                placeholder="https://example.com/audio.wav",
                help="Enter a direct link to an audio file"
            )
        
        if uploaded_file is not None or audio_url:
            if uploaded_file is not None:
                # Display file details
                st.write(f"**File:** {uploaded_file.name}")
                st.write(f"**Size:** {uploaded_file.size} bytes")
                st.write(f"**Type:** {uploaded_file.type}")
                display_name = uploaded_file.name
            else:
                # Display URL details
                st.write(f"**URL:** {audio_url}")
                display_name = audio_url.split('/')[-1].split('?')[0] or "audio_file"
            
            # Process button
            if st.button("🚀 Generate Transcript", type="primary"):
                if not audio_prompt.strip():
                    st.warning("⚠️ Please enter a transcription prompt")
                else:
                    with st.spinner("Processing audio..."):
                        try:
                            if uploaded_file is not None:
                                # Save uploaded file temporarily
                                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    tmp_path = tmp_file.name
                            else:
                                # Download from URL
                                tmp_path = download_audio_from_url(audio_url)
                                if not tmp_path:
                                    st.error("❌ Failed to download audio from URL")
                                    return
                            
                            # Generate transcript
                            transcript = generate_transcript_from_audio(tmp_path, display_name, audio_prompt)
                            
                            if transcript:
                                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                                st.success("✅ Transcript generated successfully!")
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Display transcript
                                st.subheader("📄 Generated Transcript")
                                st.text_area("Transcript", transcript, height=400, disabled=True)
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Transcript",
                                    data=transcript,
                                    file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                    mime="text/plain"
                                )
                            else:
                                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                                st.error("❌ Failed to generate transcript. Please try again.")
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        finally:
                            # Clean up temporary file
                            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                                os.unlink(tmp_path)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:  # Prompt & Transcript Processing
        st.markdown('<div class="option-card">', unsafe_allow_html=True)
        st.header("📝 Prompt & Transcript Processing")
        st.write("Provide a prompt and transcript to get processed output using Gemini AI")
        
        # Prompt input
        st.subheader("📋 Enter Your Prompt")
        prompt = st.text_area(
            "Prompt:",
            placeholder="Enter your processing prompt here...",
            height=150,
            help="This prompt will guide how Gemini processes the transcript"
        )
        
        # Transcript input
        st.subheader("📄 Enter Transcript")
        transcript = st.text_area(
            "Transcript:",
            placeholder="Paste your transcript here...",
            height=300,
            help="The transcript to be processed according to your prompt"
        )
        
        # Process button
        if st.button("🚀 Process with Gemini", type="primary"):
            if not prompt.strip():
                st.warning("⚠️ Please enter a prompt")
            elif not transcript.strip():
                st.warning("⚠️ Please enter a transcript")
            else:
                with st.spinner("Processing with Gemini..."):
                    result = process_text_with_gemini(prompt, transcript)
                    
                    if result:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("✅ Processing completed successfully!")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display result
                        st.subheader("📊 Processed Result")
                        st.text_area("Result", result, height=400, disabled=True)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Result",
                            data=result,
                            file_name=f"processed_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.markdown('<div class="error-box">', unsafe_allow_html=True)
                        st.error("❌ Failed to process. Please try again.")
                        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Powered by Google Gemini AI | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
