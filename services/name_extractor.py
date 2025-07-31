#!/usr/bin/env python3
"""
Name Extraction Service - Extract user names from speech
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class NameExtractor:
    def __init__(self):
        # Common name introduction patterns
        self.name_patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"name's (\w+)",
            r"i'm called (\w+)",
            r"they call me (\w+)",
        ]
        
        # Words to ignore (not names)
        self.stopwords = {
            'hello', 'hi', 'hey', 'good', 'morning', 'evening', 'afternoon',
            'fine', 'okay', 'well', 'very', 'really', 'pretty', 'quite',
            'feeling', 'doing', 'going', 'coming', 'here', 'there', 'now',
            'today', 'yesterday', 'tomorrow', 'sorry', 'thanks', 'please'
        }
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract name from user speech"""
        text_lower = text.lower().strip()
        
        # Try each pattern
        for pattern in self.name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                potential_name = match.group(1).strip()
                
                # Validate it's not a stopword
                if potential_name not in self.stopwords and len(potential_name) > 1:
                    # Capitalize first letter
                    name = potential_name.capitalize()
                    logger.info(f"✅ Extracted name: {name}")
                    return name
        
        return None
    
    def has_name_introduction(self, text: str) -> bool:
        """Check if text contains name introduction"""
        text_lower = text.lower()
        
        for pattern in self.name_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False

# Global instance
name_extractor = NameExtractor()