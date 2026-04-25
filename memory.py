#!/usr/bin/env python3
"""
Advanced Memory Management System for GPT Agent

This module provides intelligent conversation memory management with automatic
summarization, context windowing, and persistence capabilities.
"""

import json
import os
from typing import List, Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict, field


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryManager:
    """
    Manages conversation memory with intelligent context management.
    
    Features:
    - Automatic summarization of old conversations
    - Context window management to prevent token overflow
    - Conversation history persistence
    - Metadata tracking for conversation turns
    """
    
    def __init__(
        self,
        max_turns: int = 50,
        summary_threshold: int = 20,
        enable_persistence: bool = False,
        persistence_file: Optional[str] = None
    ):
        """
        Initialize the memory manager.
        
        Args:
            max_turns: Maximum number of conversation turns to keep in active memory
            summary_threshold: Number of turns before triggering automatic summarization
            enable_persistence: Whether to save conversation history to disk
            persistence_file: File path for saving conversation history
        """
        self.max_turns = max_turns
        self.summary_threshold = summary_threshold
        self.enable_persistence = enable_persistence
        self.persistence_file = persistence_file or "conversation_history.json"
        
        self.turns: List[ConversationTurn] = []
        self.summaries: List[str] = []
        self.system_message: Optional[str] = None
        
        # Load persisted history if available
        if enable_persistence and os.path.exists(self.persistence_file):
            self.load_from_file()
    
    def add_turn(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation turn to memory.
        
        Args:
            role: The role of the speaker (user, assistant, system)
            content: The content of the message
            metadata: Optional metadata about the turn
        """
        turn = ConversationTurn(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.turns.append(turn)
        
        # Check if we need to summarize
        if len(self.turns) > self.summary_threshold:
            self._trigger_summarization()
        
        # Persist if enabled
        if self.enable_persistence:
            self.save_to_file()
    
    def set_system_message(self, message: str) -> None:
        """Set the system message for the conversation."""
        self.system_message = message
    
    def get_context_window(self, max_tokens: int = 4000) -> List[Dict[str, str]]:
        """
        Get the conversation context formatted for API calls.
        
        This intelligently combines summaries and recent turns to fit
        within the token limit while preserving context.
        
        Args:
            max_tokens: Maximum tokens to allocate for context
        
        Returns:
            List of messages formatted for OpenAI API
        """
        messages = []
        
        # Add system message if present
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        
        # Add summaries as context
        if self.summaries:
            summary_content = "\n\n".join(self.summaries)
            messages.append({
                "role": "system",
                "content": f"Previous conversation summary:\n{summary_content}"
            })
        
        # Add recent turns
        for turn in self.turns:
            messages.append({
                "role": turn.role,
                "content": turn.content
            })
        
        return messages
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history with metadata."""
        return [asdict(turn) for turn in self.turns]
    
    def get_summary_history(self) -> List[str]:
        """Get all conversation summaries."""
        return self.summaries.copy()
    
    def reset(self, keep_system_message: bool = True) -> None:
        """
        Reset the conversation memory.
        
        Args:
            keep_system_message: Whether to keep the system message
        """
        self.turns = []
        self.summaries = []
        if not keep_system_message:
            self.system_message = None
    
    def _trigger_summarization(self) -> None:
        """
        Trigger automatic summarization of older conversation turns.
        
        This keeps the most recent turns in active memory and summarizes
        older ones to preserve context while managing token usage.
        """
        # Keep the most recent turns and summarize the rest
        keep_count = max(5, self.max_turns // 3)
        
        if len(self.turns) > keep_count:
            # Turns to summarize
            turns_to_summarize = self.turns[:-keep_count]
            # Recent turns to keep
            self.turns = self.turns[-keep_count:]
            
            # Create summary
            summary = self._create_summary(turns_to_summarize)
            if summary:
                self.summaries.append(summary)
    
    @staticmethod
    def _create_summary(turns: List[ConversationTurn]) -> str:
        """
        Create a summary of conversation turns.
        
        Args:
            turns: List of conversation turns to summarize
        
        Returns:
            A summary string
        """
        if not turns:
            return ""
        
        # Simple extractive summary - in production, use LLM for abstractive summary
        key_points = []
        for turn in turns:
            if turn.role == "user" and len(turn.content) > 20:
                # Extract first sentence or key phrase
                sentences = turn.content.split('.')
                if sentences:
                    key_points.append(f"User asked: {sentences[0][:100]}")
        
        if key_points:
            return "Earlier in the conversation: " + "; ".join(key_points[:3])
        return ""
    
    def save_to_file(self) -> None:
        """Save conversation history to a JSON file."""
        data = {
            "system_message": self.system_message,
            "turns": [asdict(turn) for turn in self.turns],
            "summaries": self.summaries,
            "saved_at": datetime.now().isoformat()
        }
        
        try:
            with open(self.persistence_file, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save conversation history: {e}")
    
    def load_from_file(self) -> None:
        """Load conversation history from a JSON file."""
        try:
            with open(self.persistence_file, 'r') as f:
                data = json.load(f)
            
            self.system_message = data.get("system_message")
            self.summaries = data.get("summaries", [])
            
            # Reconstruct turns
            self.turns = []
            for turn_data in data.get("turns", []):
                turn = ConversationTurn(
                    role=turn_data["role"],
                    content=turn_data["content"],
                    timestamp=turn_data.get("timestamp", datetime.now().isoformat()),
                    metadata=turn_data.get("metadata", {})
                )
                self.turns.append(turn)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load conversation history: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the conversation."""
        total_turns = len(self.turns)
        user_turns = sum(1 for t in self.turns if t.role == "user")
        assistant_turns = sum(1 for t in self.turns if t.role == "assistant")
        total_chars = sum(len(t.content) for t in self.turns)
        
        return {
            "total_turns": total_turns,
            "user_turns": user_turns,
            "assistant_turns": assistant_turns,
            "total_characters": total_chars,
            "average_turn_length": total_chars // max(total_turns, 1),
            "summaries_count": len(self.summaries),
            "memory_efficient": len(self.turns) <= self.max_turns
        }
