#!/usr/bin/env python3
"""
Dynamic Name Extraction Module
Extracts user names from natural language across multiple languages
"""

import re
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class NameExtractor:
    """Extracts and manages user names from speech input"""
    
    def __init__(self):
        # Name extraction patterns for different languages
        self.english_patterns = [
            r"(?:my name is|i am|i'm|call me|this is)\s+([a-zA-Z][a-zA-Z\s]{1,20})",
            r"(?:hello|hi|hey),?\s+(?:my name is|i am|i'm)\s+([a-zA-Z][a-zA-Z\s]{1,20})",
            r"(?:i go by|they call me|known as)\s+([a-zA-Z][a-zA-Z\s]{1,20})",
            r"(?:i'm called|people call me)\s+([a-zA-Z][a-zA-Z\s]{1,20})"
        ]
        
        self.hindi_patterns = [
            r"(?:mera naam|main|mai)\s+([a-zA-Z][a-zA-Z\s]{1,20})\s+(?:hun|hoon|hai)",
            r"(?:aap mujhe|mujhe)\s+([a-zA-Z][a-zA-Z\s]{1,20})\s+(?:keh sakte hain|bula sakte hain)",
            r"(?:namaste|namaskar),?\s+(?:main|mai)\s+([a-zA-Z][a-zA-Z\s]{1,20})\s+(?:hun|hoon)"
        ]
        
        self.gujarati_patterns = [
            r"(?:maru naam|hu)\s+([a-zA-Z][a-zA-Z\s]{1,20})\s+(?:chhu|chu)",
            r"(?:tame mane|mane)\s+([a-zA-Z][a-zA-Z\s]{1,20})\s+(?:kaho|keho)",
            r"(?:namaste|namaskar),?\s+(?:hu|maru naam)\s+([a-zA-Z][a-zA-Z\s]{1,20})"
        ]
        
        # Common stopwords to avoid (words that are definitely not names)
        self.stopwords = {
            'english': {
                'not', 'very', 'good', 'fine', 'okay', 'yes', 'no', 'the', 'and', 'or',
                'but', 'with', 'from', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under',
                'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most',
                'other', 'some', 'such', 'than', 'too', 'very', 'can', 'will', 'just',
                'should', 'now', 'feeling', 'happy', 'sad', 'angry', 'tired', 'hungry',
                'thirsty', 'cold', 'hot', 'sick', 'well', 'bad', 'good', 'nice', 'great',
                'terrible', 'awful', 'wonderful', 'amazing', 'fantastic', 'horrible'
            },
            'hindi': {
                'nahi', 'haan', 'theek', 'accha', 'bura', 'khushi', 'dukh', 'gussa',
                'thak', 'bhook', 'pyaas', 'thand', 'garam', 'bimar', 'swasth'
            },
            'gujarati': {
                'nathi', 'haan', 'saras', 'bhal', 'kharab', 'khush', 'dukh', 'gusso',
                'thakyo', 'bhukh', 'tiyaas', 'thando', 'garam', 'bimar', 'swasth'
            }
        }
        
        # Name validation patterns
        self.name_validation = {
            'min_length': 2,
            'max_length': 25,
            'allowed_chars': r'^[a-zA-Z\s\'-]+$',
            'max_words': 3
        }
        
        # Previously extracted names for consistency
        self.known_names = []
        
    def extract_name(self, text: str, language: str = 'en') -> Optional[str]:
        """
        Extract name from text input
        
        Args:
            text: Input text containing potential name
            language: Language code ('en', 'hi', 'gu')
            
        Returns:
            Extracted name or None if no valid name found
        """
        if not text or not text.strip():
            return None
        
        text = text.strip().lower()
        logger.debug(f"🔍 Extracting name from [{language}]: {text}")
        
        # Select appropriate patterns based on language
        patterns = self.get_patterns_for_language(language)
        
        # Try to extract name using patterns
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                potential_name = match.group(1).strip()
                
                # Validate the extracted name
                if self.validate_name(potential_name, language):
                    # Clean and format the name
                    cleaned_name = self.clean_name(potential_name)
                    if cleaned_name:
                        logger.info(f"✅ Name extracted: {cleaned_name}")
                        self.known_names.append(cleaned_name)
                        return cleaned_name
        
        # Try fallback extraction for edge cases
        fallback_name = self.fallback_extraction(text, language)
        if fallback_name:
            logger.info(f"✅ Name extracted (fallback): {fallback_name}")
            self.known_names.append(fallback_name)
            return fallback_name
        
        logger.debug("❌ No valid name found in text")
        return None
    
    def get_patterns_for_language(self, language: str) -> List[str]:
        """Get extraction patterns for specific language"""
        if language == 'hi':
            return self.hindi_patterns + self.english_patterns  # Hindi often mixed with English
        elif language == 'gu':
            return self.gujarati_patterns + self.english_patterns  # Gujarati often mixed with English
        else:
            return self.english_patterns
    
    def validate_name(self, name: str, language: str = 'en') -> bool:
        """
        Validate if extracted text is likely a valid name
        
        Args:
            name: Potential name to validate
            language: Language code for context
            
        Returns:
            True if name appears valid, False otherwise
        """
        if not name:
            return False
        
        name = name.strip()
        
        # Length check
        if len(name) < self.name_validation['min_length'] or len(name) > self.name_validation['max_length']:
            logger.debug(f"❌ Name length invalid: {name}")
            return False
        
        # Character validation
        if not re.match(self.name_validation['allowed_chars'], name):
            logger.debug(f"❌ Name contains invalid characters: {name}")
            return False
        
        # Word count check
        words = name.split()
        if len(words) > self.name_validation['max_words']:
            logger.debug(f"❌ Too many words in name: {name}")
            return False
        
        # Check against stopwords
        stopwords = self.stopwords.get(self.get_language_name(language), set())
        if any(word.lower() in stopwords for word in words):
            logger.debug(f"❌ Name contains stopwords: {name}")
            return False
        
        # Additional heuristics
        if self.is_likely_not_name(name):
            logger.debug(f"❌ Name failed heuristic check: {name}")
            return False
        
        return True
    
    def get_language_name(self, language_code: str) -> str:
        """Convert language code to full name"""
        mapping = {'en': 'english', 'hi': 'hindi', 'gu': 'gujarati'}
        return mapping.get(language_code, 'english')
    
    def is_likely_not_name(self, name: str) -> bool:
        """Additional heuristics to filter out non-names"""
        name_lower = name.lower()
        
        # Common phrases that are definitely not names
        not_names = [
            'feeling', 'doing', 'going', 'coming', 'working', 'studying',
            'playing', 'eating', 'sleeping', 'thinking', 'talking',
            'walking', 'running', 'sitting', 'standing', 'looking',
            'listening', 'reading', 'writing', 'cooking', 'cleaning',
            'shopping', 'driving', 'traveling', 'meeting', 'calling',
            'email', 'phone', 'computer', 'internet', 'website',
            'today', 'tomorrow', 'yesterday', 'morning', 'evening',
            'night', 'afternoon', 'weekend', 'monday', 'tuesday',
            'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
        ]
        
        if name_lower in not_names:
            return True
        
        # Names with numbers are suspicious
        if any(char.isdigit() for char in name):
            return True
        
        # Too many repeated characters
        if any(name_lower.count(char) > 3 for char in set(name_lower) if char.isalpha()):
            return True
        
        return False
    
    def clean_name(self, name: str) -> Optional[str]:
        """Clean and format extracted name"""
        if not name:
            return None
        
        # Remove extra whitespace
        name = ' '.join(name.split())
        
        # Capitalize properly
        words = []
        for word in name.split():
            # Handle names with apostrophes (O'Connor, D'Angelo)
            if "'" in word:
                parts = word.split("'")
                cleaned_word = "'".join([part.capitalize() for part in parts])
            else:
                cleaned_word = word.capitalize()
            words.append(cleaned_word)
        
        cleaned = ' '.join(words)
        
        # Final validation
        if self.validate_name(cleaned):
            return cleaned
        
        return None
    
    def fallback_extraction(self, text: str, language: str = 'en') -> Optional[str]:
        """Fallback extraction for edge cases"""
        
        # Look for patterns where name might appear without explicit indicators
        fallback_patterns = [
            r"(?:hello|hi|hey|namaste|namaskar),?\s+([a-zA-Z][a-zA-Z\s]{1,15})(?:\s+(?:here|speaking|this side))?",
            r"(?:myself|i'm|i am)\s+([a-zA-Z][a-zA-Z\s]{1,15})",
            r"^([a-zA-Z][a-zA-Z\s]{2,15})(?:\s+(?:here|speaking|this side))$"
        ]
        
        for pattern in fallback_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                potential_name = match.group(1).strip()
                
                # More strict validation for fallback
                if (self.validate_name(potential_name, language) and 
                    len(potential_name.split()) <= 2 and  # Max 2 words for fallback
                    not self.is_common_phrase(potential_name)):
                    
                    return self.clean_name(potential_name)
        
        return None
    
    def is_common_phrase(self, text: str) -> bool:
        """Check if text is a common phrase rather than a name"""
        common_phrases = [
            'very good', 'very bad', 'very nice', 'very well', 'very happy',
            'very sad', 'thank you', 'how are', 'what is', 'where is',
            'good morning', 'good evening', 'good night', 'good bye',
            'see you', 'talk to', 'listen to', 'look at', 'think about'
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in common_phrases)
    
    def get_greeting_response(self, name: str, language: str = 'en') -> str:
        """Generate appropriate greeting response for extracted name"""
        
        greetings = {
            'en': [
                f"Hello {name}! It's wonderful to meet you.",
                f"Nice to meet you, {name}! I'm so glad you're here.",
                f"Hi {name}! What a lovely name. How are you feeling today?",
                f"Welcome {name}! I'm excited to talk with you.",
                f"Hello there, {name}! Thank you for sharing your name with me."
            ],
            'hi': [
                f"Namaste {name}! Aapse milkar khushi hui.",
                f"Hello {name}! Aapka naam bahut accha hai.",
                f"Namaste {name}! Aap kaise hain?"
            ],
            'gu': [
                f"Namaste {name}! Tamne male ne khushi thay.",
                f"Hello {name}! Tamaru naam saras che.",
                f"Namaste {name}! Tame kem cho?"
            ]
        }
        
        import random
        responses = greetings.get(language, greetings['en'])
        return random.choice(responses)
    
    def is_name_update(self, text: str, current_name: str = None) -> bool:
        """Check if user is trying to update their name"""
        if not current_name:
            return False
        
        update_patterns = [
            r"(?:actually|no|sorry),?\s+(?:my name is|i am|i'm|call me)\s+([a-zA-Z][a-zA-Z\s]{1,20})",
            r"(?:that's wrong|not right|incorrect),?\s+(?:my name is|i am|i'm)\s+([a-zA-Z][a-zA-Z\s]{1,20})",
            r"(?:please change|update|correct)\s+(?:my name|name)\s+(?:to|as)\s+([a-zA-Z][a-zA-Z\s]{1,20})"
        ]
        
        for pattern in update_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def get_known_names(self) -> List[str]:
        """Get list of previously extracted names"""
        return self.known_names.copy()
    
    def clear_known_names(self):
        """Clear the list of known names"""
        self.known_names.clear()
        logger.info("🗑️ Known names cleared")
    
    def get_extraction_stats(self) -> Dict:
        """Get statistics about name extractions"""
        return {
            'total_names_extracted': len(self.known_names),
            'unique_names': len(set(self.known_names)),
            'most_recent': self.known_names[-1] if self.known_names else None,
            'supported_languages': ['English', 'Hindi', 'Gujarati'],
            'patterns_count': {
                'english': len(self.english_patterns),
                'hindi': len(self.hindi_patterns),
                'gujarati': len(self.gujarati_patterns)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    extractor = NameExtractor()
    
    # Test cases
    test_cases = [
        ("My name is John", "en"),
        ("Hi, I'm Sarah Johnson", "en"),
        ("Hello, call me Mike", "en"),
        ("mera naam Priya hai", "hi"),
        ("main Rahul hoon", "hi"),
        ("hu Diya chhu", "gu"),
        ("maru naam Kiran che", "gu"),
        ("I am feeling good", "en"),  # Should not extract
        ("My name is very happy", "en"),  # Should not extract
        ("Call me tomorrow", "en"),  # Should not extract
    ]
    
    print("🧪 Testing Name Extractor...")
    print("=" * 50)
    
    for text, lang in test_cases:
        result = extractor.extract_name(text, lang)
        status = "✅" if result else "❌"
        print(f"{status} [{lang.upper()}] '{text}' -> {result}")
        
        if result:
            greeting = extractor.get_greeting_response(result, lang)
            print(f"   💬 Response: {greeting}")
        
        print()
    
    # Print statistics
    stats = extractor.get_extraction_stats()
    print("📊 Extraction Statistics:")
    print(f"   Total extractions: {stats['total_names_extracted']}")
    print(f"   Unique names: {stats['unique_names']}")
    print(f"   Known names: {extractor.get_known_names()}")