import streamlit as st
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

class ConversationMemory:
    """
    Enhanced conversation memory management with persistence
    """
    
    def __init__(self, memory_file: str = "conversation_memory.json"):
        self.memory_file = memory_file
        self.conversations = self.load_memory()
    
    def load_memory(self) -> List[Dict[str, Any]]:
        """Load conversation memory from file"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading memory: {e}")
        return []
    
    def save_memory(self):
        """Save conversation memory to file"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def add_conversation(self, user_input: str, bot_response: str, mode: str = "chat"):
        """Add a conversation to memory"""
        conversation = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "bot": bot_response,
            "mode": mode
        }
        self.conversations.append(conversation)
        self.save_memory()
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversations"""
        return self.conversations[-limit:] if self.conversations else []
    
    def search_conversations(self, query: str) -> List[Dict[str, Any]]:
        """Search conversations by query"""
        results = []
        query_lower = query.lower()
        
        for conv in self.conversations:
            if (query_lower in conv.get("user", "").lower() or 
                query_lower in conv.get("bot", "").lower()):
                results.append(conv)
        
        return results
    
    def clear_memory(self):
        """Clear all conversation memory"""
        self.conversations = []
        self.save_memory()
    
    def get_conversation_count(self) -> int:
        """Get total conversation count"""
        return len(self.conversations)
    
    def export_conversations(self) -> str:
        """Export conversations as JSON string"""
        return json.dumps(self.conversations, indent=2, ensure_ascii=False)

# Legacy functions for backward compatibility
def get_chat_history():
    return st.session_state.get("chat_history", [])

def add_to_history(user_query, bot_answer):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    st.session_state["chat_history"].append({"user": user_query, "bot": bot_answer})

def clear_history():
    st.session_state["chat_history"] = []
