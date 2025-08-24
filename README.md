# StudyMate AI - Voice & Chat Assistant 🎓

A powerful AI-powered study assistant that combines document-based Q&A with voice interaction capabilities. Upload your study materials and interact with them through both text and voice!

## Features ✨

- **📚 Document Processing**: Upload PDF, TXT, and DOCX files
- **🎤 Voice Interaction**: Speak your questions and get audio responses
- **💬 Text Chat**: Traditional text-based conversation
- **🧠 Multiple Modes**: 
  - Chat mode for conversational learning
  - Q&A mode for direct answers
  - Quiz mode for generating practice questions
- **📝 Conversation History**: Persistent conversation storage and export
- **🔍 RAG Technology**: Advanced retrieval-augmented generation for accurate responses

## Prerequisites 📋

Before running the application, ensure you have:

1. **Python 3.8+** installed
2. **Audio drivers** properly configured on your system
3. **Microphone access** enabled for your browser/system
4. **Internet connection** for speech recognition and AI processing

### Windows-specific Requirements

For Windows users, you may need to install additional audio libraries:

```bash
# Install PyAudio dependencies (if needed)
pip install pipwin
pipwin install pyaudio
```

## Installation 🚀

1. **Clone or download** this repository
2. **Navigate** to the project directory
3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On macOS/Linux
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure your API key** in `config.py`:
   - Update the `GROQ_API_KEY` with your actual Groq API key
   - You can get a free API key from [Groq](https://groq.com/)

## Usage 🎯

### Starting the Application

1. **Activate your virtual environment** (if using one)
2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
3. **Open your browser** to the provided URL (usually `http://localhost:8501`)

### Using the Voice Bot

1. **Upload Documents**:
   - Use the sidebar to upload your study materials (PDF, TXT, DOCX)
   - Click "Process Documents" to build the knowledge base

2. **Voice Interaction**:
   - Click the "🎤 Start Voice Chat" button
   - Speak your question clearly when prompted
   - Wait for the AI to process and respond with audio

3. **Text Interaction**:
   - Type your questions in the text area
   - Click "Send" to get text responses

4. **Mode Selection**:
   - **Chat Mode**: Conversational, educational responses
   - **Q&A Mode**: Direct, factual answers
   - **Quiz Mode**: Generates practice questions

### Voice Features

- **Speech Recognition**: Uses Google Speech Recognition for accurate transcription
- **Text-to-Speech**: Converts AI responses to natural-sounding speech
- **Conversation History**: All voice interactions are transcribed and stored
- **Audio Feedback**: Visual indicators during recording and processing

## Configuration ⚙️

### Audio Settings

The voice bot automatically adjusts for ambient noise, but you can optimize performance by:

- Using a **good quality microphone**
- Speaking in a **quiet environment**
- Speaking **clearly and at normal pace**
- Ensuring **stable internet connection**

### API Configuration

In `config.py`, you can modify:

- `GROQ_API_KEY`: Your Groq API key
- `LLM_MODEL`: The language model to use
- `EMBEDDING_MODEL`: The embedding model for document processing
- `CHUNK_SIZE` and `CHUNK_OVERLAP`: Document processing parameters

## Troubleshooting 🔧

### Common Issues

1. **Microphone not working**:
   - Check browser permissions for microphone access
   - Ensure your microphone is properly connected
   - Try refreshing the page

2. **Audio playback issues**:
   - Check system volume settings
   - Ensure audio drivers are up to date
   - Try using headphones

3. **Speech recognition errors**:
   - Speak more clearly and slowly
   - Check internet connection
   - Try in a quieter environment

4. **Installation issues**:
   - Ensure Python 3.8+ is installed
   - Try installing dependencies one by one
   - Check for system-specific requirements

### Error Messages

- **"Could not understand the audio"**: Speak more clearly or try again
- **"Error with speech recognition service"**: Check internet connection
- **"Please upload and process documents first"**: Upload documents before chatting

## File Structure 📁

```
StudyMate_AI/
├── app.py                 # Main Streamlit application
├── voice_utils.py         # Voice processing utilities
├── rag_engine.py         # RAG implementation
├── memory.py             # Conversation memory management
├── utils.py              # Document loading utilities
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dependencies 📦

Key dependencies include:

- **Streamlit**: Web interface
- **SpeechRecognition**: Speech-to-text conversion
- **gTTS**: Google Text-to-Speech
- **PyGame**: Audio playback
- **LangChain**: Document processing and RAG
- **Groq**: AI language model API
- **ChromaDB**: Vector database for document storage

## Contributing 🤝

Feel free to contribute by:

1. Reporting bugs
2. Suggesting new features
3. Improving documentation
4. Submitting pull requests

## License 📄

This project is open source and available under the MIT License.

## Support 💬

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the error messages for specific guidance
3. Ensure all dependencies are properly installed
4. Verify your API key is correctly configured

---

**Happy studying with StudyMate AI! 🎓✨**

