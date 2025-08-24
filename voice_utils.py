#!/usr/bin/env python3
"""
Voice utilities for StudyMate AI - Text-to-Speech and Speech-to-Text functionality
"""

import io
import os
import tempfile
import threading
import time
from typing import Optional, Union

import streamlit as st
import speech_recognition as sr
import pygame
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

class VoiceBot:
    """
    Voice Bot for handling text-to-speech and speech-to-text operations
    """
    
    def __init__(self, language='en', tld='com'):
        """
        Initialize VoiceBot
        
        Args:
            language (str): Language code for TTS (default: 'en')
            tld (str): Top-level domain for gTTS (default: 'com')
        """
        self.language = language
        self.tld = tld
        self.recognizer = sr.Recognizer()
        self.microphone = None
        self._init_audio_system()
        self._lock = threading.Lock()
        
        # Adjust recognizer settings for better voice detection
        self.recognizer.energy_threshold = 300  # Adjust based on environment
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 1.0  # Wait 1 second of silence before considering phrase complete
    
    def _init_audio_system(self):
        """Initialize audio system"""
        try:
            # Initialize pygame mixer for audio playback
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            
            # Initialize microphone
            try:
                self.microphone = sr.Microphone()
                # Adjust for ambient noise
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            except Exception as e:
                print(f"Warning: Could not initialize microphone: {e}")
                self.microphone = None
                
        except Exception as e:
            print(f"Warning: Could not initialize audio system: {e}")
    
    def text_to_speech(self, text: str, slow: bool = False, play_audio: bool = True) -> bool:
        """
        Convert text to speech and optionally play it
        
        Args:
            text (str): Text to convert to speech
            slow (bool): Whether to speak slowly
            play_audio (bool): Whether to play the audio
            
        Returns:
            bool: Success status
        """
        if not text or not text.strip():
            return False
            
        try:
            with self._lock:
                # Create TTS object
                tts = gTTS(text=text.strip(), lang=self.language, tld=self.tld, slow=slow)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_filename = temp_file.name
                    tts.save(temp_filename)
                
                if play_audio:
                    self._play_audio_file(temp_filename)
                
                # Clean up temporary file
                try:
                    os.unlink(temp_filename)
                except:
                    pass
                    
                return True
                
        except Exception as e:
            print(f"Error in text-to-speech: {e}")
            return False
    
    def _play_audio_file(self, filename: str):
        """Play audio file using pygame"""
        try:
            # Load and play with pygame
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Error playing audio: {e}")
            # Fallback to pydub if pygame fails
            try:
                audio = AudioSegment.from_mp3(filename)
                play(audio)
            except Exception as e2:
                print(f"Fallback audio playback also failed: {e2}")
    
    def speech_to_text(self, timeout: int = 15, phrase_timeout: int = 5) -> Optional[str]:
        """
        Convert speech to text using microphone input
        
        Args:
            timeout (int): Maximum time to wait for speech (increased to 15s)
            phrase_timeout (int): Timeout for phrase completion (increased to 5s)
            
        Returns:
            Optional[str]: Recognized text or None if failed
        """
        if not self.microphone:
            print("No microphone available")
            return None
            
        try:
            with self.microphone as source:
                # Adjust for ambient noise with longer duration
                print("Adjusting for background noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
                
                # Store original settings
                original_energy = self.recognizer.energy_threshold
                original_pause = self.recognizer.pause_threshold
                
                # Adjust for better speech detection
                self.recognizer.energy_threshold = max(original_energy, 200)
                self.recognizer.pause_threshold = 1.5  # Wait 1.5 seconds of silence
                
                print("🎤 Ready! Start speaking now... (I'll wait for you to finish)")
                
                # Listen for audio with improved settings
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout,
                    phrase_time_limit=None  # No limit - wait for natural pause
                )
                
                # Restore original settings
                self.recognizer.energy_threshold = original_energy
                self.recognizer.pause_threshold = original_pause
                
                print("🔄 Processing your speech...")
                
                # Use Google Web Speech API for recognition
                text = self.recognizer.recognize_google(audio, language=self.language)
                print(f"✅ Recognized: {text}")
                return text
                
        except sr.WaitTimeoutError:
            print("Listening timeout - no speech detected")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in speech recognition: {e}")
            return None
    
    def get_microphone_list(self) -> list:
        """Get list of available microphones"""
        try:
            return sr.Microphone.list_microphone_names()
        except Exception as e:
            print(f"Error getting microphone list: {e}")
            return []
    
    def set_microphone(self, device_index: Optional[int] = None):
        """Set microphone device"""
        try:
            if device_index is not None:
                self.microphone = sr.Microphone(device_index=device_index)
            else:
                self.microphone = sr.Microphone()
                
            # Re-adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
            print(f"Microphone set successfully")
            return True
            
        except Exception as e:
            print(f"Error setting microphone: {e}")
            return False
    
    def test_audio_playback(self, test_text: str = "This is a test of the audio system") -> bool:
        """Test audio playback functionality"""
        print("Testing audio playback...")
        return self.text_to_speech(test_text)
    
    def test_speech_recognition(self) -> Optional[str]:
        """Test speech recognition functionality"""
        print("Testing speech recognition...")
        return self.speech_to_text(timeout=10)
    
    def cleanup(self):
        """Clean up audio resources"""
        try:
            pygame.mixer.quit()
        except:
            pass

# Streamlit integration functions
@st.cache_resource
def get_voice_bot():
    """Get cached VoiceBot instance for Streamlit"""
    return VoiceBot()

def streamlit_audio_recorder(voice_bot: VoiceBot, key: str = "audio_input") -> Optional[str]:
    """
    Streamlit component for audio recording
    
    Args:
        voice_bot (VoiceBot): VoiceBot instance
        key (str): Unique key for the component
        
    Returns:
        Optional[str]: Recognized text or None
    """
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("🎤 Record", key=f"record_{key}"):
            with st.spinner("🎤 Adjusting for background noise..."):
                time.sleep(0.5)  # Give user a moment
                
            with st.spinner("🗣️ Ready! Speak your complete question... (I'll wait for you to finish)"):
                text = voice_bot.speech_to_text(timeout=20)  # Give even more time
                if text:
                    st.session_state[f"recorded_text_{key}"] = text
                    st.success(f"✅ Recorded: {text}")
                    return text
                else:
                    st.error("❌ Could not recognize speech. Please speak clearly and try again.")
                    st.info("💡 Tip: Speak clearly, pause briefly between words, and wait for 1.5 seconds of silence when done.")
                    return None
    
    with col2:
        if f"recorded_text_{key}" in st.session_state:
            st.info(f"Last recorded: {st.session_state[f'recorded_text_{key}']}")
    
    return st.session_state.get(f"recorded_text_{key}")

def streamlit_text_to_speech(voice_bot: VoiceBot, text: str, key: str = "tts"):
    """
    Streamlit component for text-to-speech
    
    Args:
        voice_bot (VoiceBot): VoiceBot instance
        text (str): Text to speak
        key (str): Unique key for the component
    """
    if text and st.button("🔊 Speak", key=f"speak_{key}"):
        with st.spinner("Speaking..."):
            success = voice_bot.text_to_speech(text)
            if success:
                st.success("Audio played successfully!")
            else:
                st.error("Failed to play audio.")

# Utility functions for integration
def create_audio_interface(voice_bot: VoiceBot):
    """Create a complete audio interface in Streamlit sidebar"""
    with st.sidebar:
        st.subheader("🎤 Voice Interface")
        
        # Microphone test
        if st.button("Test Microphone"):
            with st.spinner("Testing microphone..."):
                result = voice_bot.test_speech_recognition()
                if result:
                    st.success(f"Microphone working! Heard: '{result}'")
                else:
                    st.error("Microphone test failed")
        
        # Audio test
        if st.button("Test Speakers"):
            with st.spinner("Testing speakers..."):
                success = voice_bot.test_audio_playback()
                if success:
                    st.success("Speaker test completed!")
                else:
                    st.error("Speaker test failed")
        
        # Microphone selection
        mics = voice_bot.get_microphone_list()
        if mics:
            selected_mic = st.selectbox(
                "Select Microphone",
                options=range(len(mics)),
                format_func=lambda x: f"{x}: {mics[x][:50]}...",
                key="mic_selection"
            )
            
            if st.button("Set Microphone"):
                if voice_bot.set_microphone(selected_mic):
                    st.success(f"Microphone set to: {mics[selected_mic]}")
                else:
                    st.error("Failed to set microphone")

if __name__ == "__main__":
    # Test the VoiceBot functionality
    print("Testing VoiceBot...")
    
    voice_bot = VoiceBot()
    
    # Test TTS
    print("Testing text-to-speech...")
    voice_bot.text_to_speech("Hello! This is a test of the voice bot functionality.")
    
    # Test STT
    print("Testing speech-to-text...")
    result = voice_bot.speech_to_text(timeout=5)
    if result:
        print(f"You said: {result}")
    else:
        print("No speech detected or recognition failed")
    
    # Cleanup
    voice_bot.cleanup()
    print("VoiceBot test completed!")
