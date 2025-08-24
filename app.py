#!/usr/bin/env python3
"""
StudyMate AI - Intelligent Study Assistant with Voice Integration
A comprehensive AI-powered study companion that supports document analysis, Q&A, quizzes, and voice interaction.
"""

import streamlit as st
import os
import tempfile
import time
from typing import List, Optional
import pandas as pd

# Import our custom modules
from config import GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL
from utils import (
    save_uploaded_files, load_documents, allowed_file, 
    cleanup_previous_session, ensure_directory_permissions
)
from rag_engine import (
    build_vector_db, get_relevant_context, generate_answer, 
    close_vector_db, generate_quiz_questions
)
from memory import ConversationMemory, add_to_history, get_chat_history, clear_history
from voice_utils import get_voice_bot, streamlit_audio_recorder, streamlit_text_to_speech, create_audio_interface

# Page configuration
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        background-color: #f0f2f6;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .quiz-question {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
    }
    .mode-selector {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "qa"
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory()
    if "voice_enabled" not in st.session_state:
        st.session_state.voice_enabled = False

def display_header():
    """Display the application header"""
    st.title("🎓 StudyMate AI")
    st.markdown("### Your Intelligent Study Companion with Voice Integration")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("📚 **Upload documents** • ❓ **Ask questions** • 📝 **Generate quizzes** • 🗣️ **Voice interaction**")
    with col2:
        if st.button("🔄 New Session", type="primary", use_container_width=True):
            cleanup_session()
            st.rerun()

def reset_chromadb_database():
    """Reset ChromaDB database to fix corruption issues"""
    status_placeholder = st.sidebar.empty()
    
    with st.spinner("Resetting ChromaDB database..."):
        try:
            from utils import reset_chromadb
            from rag_engine import close_vector_db
            
            status_placeholder.info("🔄 Closing existing database connections...")
            
            # Close any existing vector database connections
            if st.session_state.get("vector_db"):
                close_vector_db(st.session_state.vector_db)
                st.session_state.vector_db = None
                st.session_state.documents_loaded = False
            
            status_placeholder.info("🗄️ Resetting ChromaDB...")
            
            # Reset the ChromaDB
            success = reset_chromadb()
            
            if success:
                status_placeholder.success("✅ Database reset successfully!")
                st.sidebar.info("💡 Please upload your documents again")
            else:
                status_placeholder.error("❌ Database reset failed")
                
        except Exception as e:
            status_placeholder.error(f"❌ Reset error: {str(e)[:50]}...")

def test_api_connection():
    """Test API connectivity and display results"""
    # Create a placeholder for the status message
    status_placeholder = st.sidebar.empty()
    
    with st.spinner("Testing API connection..."):
        try:
            from rag_engine import create_groq_client
            from config import LLM_MODEL
            
            status_placeholder.info("🔄 Connecting to API...")
            
            client = create_groq_client()
            
            # Simple test query
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            if response.choices[0].message.content:
                status_placeholder.success("✅ API Connection OK")
            else:
                status_placeholder.warning("⚠️ API responded but with empty content")
                
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower():
                status_placeholder.error("❌ Connection failed - Check internet")
            elif "auth" in error_msg.lower() or "api" in error_msg.lower():
                status_placeholder.error("❌ Authentication failed - Check API key")
            elif "rate" in error_msg.lower():
                status_placeholder.error("❌ Rate limited - Try later")
            else:
                status_placeholder.error(f"❌ API Error: {error_msg[:50]}...")

def reset_session():
    """Reset session with comprehensive cleanup"""
    try:
        # Close vector database connection first
        if st.session_state.get("vector_db"):
            close_vector_db(st.session_state.vector_db)
        
        # Force cleanup of files and directories
        cleanup_previous_session()
        
        # Clear conversation memory
        if hasattr(st.session_state, 'conversation_memory'):
            st.session_state.conversation_memory.clear_memory()
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Reinitialize
        initialize_session_state()
        
        st.success("✅ Session reset successfully! All files and data cleared.")
        
    except Exception as e:
        st.error(f"Error during reset: {e}")
        # Force reinitialize even if cleanup fails
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        initialize_session_state()

def cleanup_session():
    """Clean up current session and reset state"""
    # Close vector database
    if st.session_state.get("vector_db"):
        close_vector_db(st.session_state.vector_db)
    
    # Clean up files and reset state
    cleanup_previous_session()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    initialize_session_state()

def handle_file_upload():
    """Handle document upload and processing"""
    st.sidebar.header("📂 Document Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload your study materials",
        type=['pdf', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT"
    )
    
    if uploaded_files:
        if st.sidebar.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing documents..."):
                try:
                    # Save uploaded files
                    file_paths = save_uploaded_files(uploaded_files)
                    
                    # Load documents
                    progress_bar = st.progress(0)
                    st.info("Loading documents...")
                    progress_bar.progress(25)
                    
                    documents = load_documents(file_paths)
                    if not documents:
                        st.error("No documents could be loaded. Please check your files.")
                        return False
                    
                    progress_bar.progress(50)
                    st.info("Building knowledge base...")
                    
                    # Build vector database
                    vector_db = build_vector_db(documents)
                    progress_bar.progress(100)
                    
                    # Update session state
                    st.session_state.vector_db = vector_db
                    st.session_state.documents_loaded = True
                    
                    st.success(f"✅ Successfully processed {len(uploaded_files)} file(s) with {len(documents)} pages!")
                    return True
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    return False
    
    return st.session_state.documents_loaded

def display_mode_selector():
    """Display mode selection interface"""
    st.markdown('<div class="mode-selector">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        pass  # Empty column for spacing
    
    with col2:
        if st.button("❓ Q&A Mode", use_container_width=True,
                    type="primary" if st.session_state.current_mode == "qa" else "secondary"):
            st.session_state.current_mode = "qa"
    
    with col3:
        if st.button("📝 Quiz Mode", use_container_width=True,
                    type="primary" if st.session_state.current_mode == "quiz" else "secondary"):
            st.session_state.current_mode = "quiz"
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display mode description
    mode_descriptions = {
        "qa": "❓ **Q&A Mode**: Ask specific questions for precise answers",
        "quiz": "📝 **Quiz Mode**: Generate practice questions and quizzes"
    }
    st.info(mode_descriptions[st.session_state.current_mode])

def handle_voice_input():
    """Handle voice input if enabled"""
    if st.session_state.voice_enabled:
        voice_bot = get_voice_bot()
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🎤 Voice Input"):
                with st.spinner("🗣️ Ready! Speak your complete question... (I'll wait for you to finish)"):
                    text = voice_bot.speech_to_text(timeout=20)
                    if text:
                        st.session_state.voice_input = text
                        st.success(f"✅ Heard: {text}")
                        return text
                    else:
                        st.error("❌ Could not understand speech. Please speak clearly and try again.")
                        st.info("💡 Tip: Speak clearly and wait for 1.5 seconds of silence when you're done.")
        return st.session_state.get("voice_input", "")
    return ""

def handle_user_input():
    """Handle user input (text or voice)"""
    voice_input = ""
    
    # Voice input (if enabled)
    if st.session_state.voice_enabled:
        voice_input = handle_voice_input()
    
    # Text input
    text_input = st.text_area(
        "Ask me anything about your documents:",
        value=voice_input,
        height=100,
        placeholder="Type your question here or use voice input..."
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        submit_button = st.button("Submit", type="primary", use_container_width=True)
    
    with col2:
        if st.session_state.current_mode != "quiz":
            voice_response = st.checkbox("🔊 Voice Response", value=False)
        else:
            voice_response = False
    
    with col3:
        if st.button("🗑️ Clear Chat"):
            clear_history()
            st.rerun()
    
    return text_input, submit_button, voice_response

def process_user_query(user_input: str, voice_response: bool = False):
    """Process user query and generate response"""
    if not user_input.strip():
        return
    
    try:
        with st.spinner("Thinking..."):
            # Get relevant context
            context = get_relevant_context(user_input, st.session_state.vector_db, top_k=5)
            
            # Generate response based on mode
            if st.session_state.current_mode == "quiz":
                if "generate quiz" in user_input.lower() or "create quiz" in user_input.lower():
                    response = generate_quiz_questions(context, num_questions=5)
                else:
                    response = generate_answer(user_input, context, mode="qa")
            else:  # Q&A mode
                response = generate_answer(user_input, context, mode="qa")
            
            # Add to history
            add_to_history(user_input, response)
            st.session_state.conversation_memory.add_conversation(
                user_input, response, st.session_state.current_mode
            )
            
            # Voice response if enabled
            if voice_response and st.session_state.voice_enabled:
                voice_bot = get_voice_bot()
                voice_bot.text_to_speech(response)
            
            return response
            
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        st.error(error_msg)
        return error_msg

def display_chat_history():
    """Display chat history in an organized manner"""
    chat_history = get_chat_history()
    
    if not chat_history:
        st.info("No conversation history yet. Start by asking a question!")
        return
    
    st.subheader("💭 Conversation History")
    
    for i, exchange in enumerate(reversed(chat_history[-10:])):  # Show last 10 exchanges
        with st.expander(f"Exchange {len(chat_history) - i}", expanded=(i == 0)):
            # User message
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {exchange["user"]}</div>', 
                       unsafe_allow_html=True)
            
            # Bot response
            st.markdown(f'<div class="chat-message bot-message"><strong>StudyMate:</strong> {exchange["bot"]}</div>', 
                       unsafe_allow_html=True)
            
            # Voice response option for past messages
            if st.session_state.voice_enabled:
                if st.button(f"🔊 Replay Response", key=f"replay_{len(chat_history) - i}"):
                    voice_bot = get_voice_bot()
                    with st.spinner("Speaking..."):
                        voice_bot.text_to_speech(exchange["bot"])

def display_quiz_mode():
    """Special handling for quiz mode"""
    if st.session_state.current_mode == "quiz":
        st.subheader("📝 Quiz Generator")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("Ask me to generate a quiz or practice questions based on your documents!")
        with col2:
            num_questions = st.selectbox("Questions:", [3, 5, 10], index=1)
        
        if st.button("🎯 Generate Random Quiz"):
            with st.spinner("Generating quiz questions..."):
                # Get a sample of context for quiz generation
                sample_query = "Generate quiz questions covering the main topics"
                context = get_relevant_context(sample_query, st.session_state.vector_db, top_k=10)
                quiz_response = generate_quiz_questions(context, num_questions)
                
                # Display quiz
                st.markdown("### 📋 Your Quiz")
                st.markdown(f'<div class="quiz-question">{quiz_response}</div>', unsafe_allow_html=True)
                
                # Add to history
                add_to_history(f"Generate {num_questions} quiz questions", quiz_response)

def display_sidebar_stats():
    """Display statistics and controls in sidebar"""
    # Session controls
    st.sidebar.header("🔧 Session Controls")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Reset All", type="secondary", use_container_width=True, help="Clear all files and start fresh"):
            reset_session()
            st.rerun()
    with col2:
        if st.button("🔄 Refresh", type="primary", use_container_width=True, help="Restart current session"):
            cleanup_session()
            st.rerun()
    
    # Connection test
    if st.sidebar.button("🌐 Test Connection", use_container_width=True, help="Test API connectivity"):
        test_api_connection()
    
    # ChromaDB reset for database errors
    if st.sidebar.button("🗄️ Reset Database", use_container_width=True, help="Fix ChromaDB database errors"):
        reset_chromadb_database()
    
    # File status
    if st.session_state.documents_loaded:
        st.sidebar.info("📁 Files in memory - Use Reset to clear")
    
    # Voice controls
    st.sidebar.header("🎤 Voice Controls")
    st.session_state.voice_enabled = st.sidebar.checkbox(
        "Enable Voice Features", 
        value=st.session_state.voice_enabled,
        help="Enable voice input and output"
    )
    
    if st.session_state.voice_enabled:
        st.sidebar.info("🎙️ Voice input enabled - use the microphone button in the input area")
        
        # Simple voice test
        if st.sidebar.button("🔊 Test Voice Output", use_container_width=True):
            try:
                voice_bot = get_voice_bot()
                voice_bot.text_to_speech("Voice system is working correctly!")
                st.sidebar.success("✅ Voice test completed")
            except Exception as e:
                st.sidebar.error(f"❌ Voice error: {e}")
                st.session_state.voice_enabled = False

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Handle file upload
    documents_ready = handle_file_upload()
    
    # Display sidebar stats
    display_sidebar_stats()
    
    if not documents_ready:
        st.warning("👆 Please upload documents to get started!")
        
        # Show demo information
        st.markdown("## 🚀 Get Started")
        st.markdown("""
        **StudyMate AI** helps you study more effectively by:
        - 📖 Analyzing your documents (PDFs, text files)
        - ❓ Answering questions about the content
        - 📝 Generating practice quizzes
        - 🗣️ Supporting voice interaction
        
        **To begin:**
        1. Upload your study materials using the sidebar
        2. Click "Process Documents" 
        3. Start asking questions!
        """)
        
        # Show example usage
        with st.expander("💡 Example Questions"):
            st.markdown("""
            - "What are the main topics covered in this document?"
            - "Explain the concept of [specific topic]"
            - "Generate a quiz about [chapter/topic]"
            - "Summarize the key points"
            """)
        
        return
    
    # Main interface
    display_mode_selector()
    
    # Special handling for quiz mode
    if st.session_state.current_mode == "quiz":
        display_quiz_mode()
    
    # User input section
    user_input, submit_button, voice_response = handle_user_input()
    
    # Process query
    if submit_button and user_input:
        process_user_query(user_input, voice_response)
        st.rerun()
    
    # Display conversation history
    display_chat_history()
    
    # Footer
    st.markdown("---")
    st.markdown("*StudyMate AI - Powered by Groq LLM and ChromaDB*")

if __name__ == "__main__":
    main()
