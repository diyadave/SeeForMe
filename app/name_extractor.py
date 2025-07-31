#!/usr/bin/env python3
"""
Name Extractor - Extract user names from speech input
Multi-language support for personalizing interactions
"""

import logging
import re
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class NameExtractor:
    """Extract and manage user names from speech input"""
    
    def __init__(self):
        # Name introduction patterns for different languages
        self.name_patterns = {
            'en': [
                r'\bmy name is (\w+)',
                r'\bi am (\w+)',
                r'\bi\'m (\w+)',
                r'\bcall me (\w+)',
                r'\bthis is (\w+)',
                r'\byou can call me (\w+)',
                r'\bit\'s (\w+)',
                r'\bname is (\w+)'
            ],
            'hi': [
                r'\bमेरा नाम (\w+) है',
                r'\bमैं (\w+) हूं',
                r'\bमुझे (\w+) कहते हैं',
                r'\bनाम (\w+) है'
            ],
            'gu': [
                r'\bમારું નામ (\w+) છે',
                r'\bહું (\w+) છું',
                r'\bમને (\w+) કહે છે',
                r'\bનામ (\w+) છે'
            ]
        }
        
        # Common words to filter out (not names)
        self.stopwords = {
            'en': {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
                'her', 'us', 'them', 'my', 'your', 'his', 'hers', 'its', 'our', 'their',
                'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves',
                'yourselves', 'themselves', 'what', 'which', 'who', 'whom', 'whose',
                'where', 'when', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just',
                'should', 'now', 'good', 'bad', 'hello', 'hi', 'hey', 'okay', 'yes',
                'no', 'please', 'thank', 'thanks', 'sorry', 'excuse', 'help', 'here',
                'there', 'today', 'tomorrow', 'yesterday', 'feeling', 'happy', 'sad',
                'angry', 'tired', 'see', 'look', 'around', 'assistant', 'robot'
            },
            'hi': {
                'मैं', 'आप', 'तुम', 'वह', 'यह', 'और', 'या', 'लेकिन', 'में', 'पर',
                'से', 'को', 'का', 'की', 'के', 'हूं', 'है', 'हैं', 'था', 'थी', 'थे',
                'नमस्ते', 'हैलो', 'हाय', 'धन्यवाद', 'माफ़', 'करना', 'देखना', 'सुनना'
            },
            'gu': {
                'હું', 'તમે', 'તુ', 'તે', 'આ', 'અને', 'કે', 'પર', 'માં', 'થી', 'ને',
                'ના', 'નું', 'નો', 'છું', 'છે', 'છો', 'હતું', 'હતો', 'હતા', 'નમસ્તે',
                'હેલો', 'હાય', 'આભાર', 'માફ', 'કરવું', 'જોવું', 'સાંભળવું'
            }
        }
        
        # Common names to help with validation
        self.common_names = {
            'en': {
                'james', 'john', 'robert', 'michael', 'william', 'david', 'richard',
                'charles', 'joseph', 'thomas', 'mary', 'patricia', 'jennifer', 'linda',
                'elizabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen', 'nancy',
                'lisa', 'betty', 'helen', 'sandra', 'donna', 'carol', 'ruth', 'sharon',
                'michelle', 'laura', 'emily', 'kimberly', 'deborah', 'dorothy', 'amy',
                'angela', 'ashley', 'brenda', 'emma', 'olivia', 'cynthia', 'marie',
                'anna', 'alex', 'chris', 'sam', 'jordan', 'taylor', 'morgan', 'casey',
                'jamie', 'riley', 'avery', 'quinn', 'drew', 'sage', 'blake', 'phoenix'
            },
            'hi': {
                'राम', 'श्याम', 'गीता', 'सीता', 'अमित', 'सुमित', 'प्रिया', 'पूजा',
                'राज', 'विकास', 'अनिल', 'सुनील', 'मोहन', 'सोहन', 'रीता', 'मीता',
                'अर्जुन', 'कृष्ण', 'राधा', 'लक्ष्मी', 'शिव', 'पार्वती', 'गणेश',
                'हनुमान', 'ब्रह्मा', 'विष्णु', 'इंद्र', 'वरुण', 'अग्नि', 'वायु'
            },
            'gu': {
                'રામ', 'શ્યામ', 'ગીતા', 'સીતા', 'અમિત', 'સુમિત', 'પ્રિયા', 'પૂજા',
                'રાજ', 'વિકાસ', 'અનિલ', 'સુનીલ', 'મોહન', 'સોહન', 'રીતા', 'મીતા',
                'અર્જુન', 'કૃષ્ણ', 'રાધા', 'લક્ષ્મી', 'શિવ', 'પાર્વતી', 'ગણેશ',
                'હનુમાન', 'બ્રહ્મા', 'વિષ્ણુ', 'ઇન્દ્ર', 'વરુણ', 'અગ્નિ', 'વાયુ'
            }
        }
        
        logger.info("👤 Name extractor initialized")
    
    def extract_name(self, text: str, language: str = 'en') -> Optional[str]:
        """Extract name from text input"""
        if not text or not text.strip():
            return None
        
        text = text.strip().lower()
        
        # Try pattern-based extraction first
        name = self._extract_by_patterns(text, language)
        if name:
            return name
        
        # Try heuristic extraction
        name = self._extract_by_heuristics(text, language)
        if name:
            return name
        
        return None
    
    def _extract_by_patterns(self, text: str, language: str) -> Optional[str]:
        """Extract name using predefined patterns"""
        patterns = self.name_patterns.get(language, self.name_patterns['en'])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                potential_name = match.group(1).strip()
                if self._is_valid_name(potential_name, language):
                    return potential_name.title()
        
        return None
    
    def _extract_by_heuristics(self, text: str, language: str) -> Optional[str]:
        """Extract name using heuristic methods"""
        words = text.split()
        
        # Look for capitalized words that might be names
        for word in words:
            # Skip common words
            if word.lower() in self.stopwords.get(language, self.stopwords['en']):
                continue
            
            # Check if it looks like a name
            if self._looks_like_name(word, language):
                if self._is_valid_name(word, language):
                    return word.title()
        
        return None
    
    def _looks_like_name(self, word: str, language: str) -> bool:
        """Check if a word looks like a name"""
        # Basic checks
        if len(word) < 2 or len(word) > 20:
            return False
        
        # Should contain only letters (and possibly apostrophes, hyphens)
        if not re.match(r"^[a-zA-ZÀ-ÿ\u0900-\u097F\u0A80-\u0AFF\u0B80-\u0BFF'-]+$", word):
            return False
        
        # Should start with a letter
        if not word[0].isalpha():
            return False
        
        return True
    
    def _is_valid_name(self, name: str, language: str) -> bool:
        """Validate if extracted text is likely a real name"""
        if not name:
            return False
        
        name_lower = name.lower()
        
        # Check against stopwords
        if name_lower in self.stopwords.get(language, self.stopwords['en']):
            return False
        
        # Check if it's a common name (higher confidence)
        if name_lower in self.common_names.get(language, set()):
            return True
        
        # Basic validation rules
        if len(name) < 2 or len(name) > 20:
            return False
        
        # Should not be all the same character
        if len(set(name.lower())) < 2:
            return False
        
        # Should not contain numbers
        if any(char.isdigit() for char in name):
            return False
        
        # Should not be common command words
        command_words = {
            'en': {'help', 'stop', 'start', 'pause', 'play', 'next', 'back', 'exit', 'quit'},
            'hi': {'मदद', 'रोकें', 'शुरू', 'रुकें', 'चलाएं', 'अगला', 'पीछे', 'बाहर'},
            'gu': {'મદદ', 'રોકો', 'શરૂ', 'થોભો', 'ચલાવો', 'આગળ', 'પાછળ', 'બહાર'}
        }
        
        if name_lower in command_words.get(language, command_words['en']):
            return False
        
        return True
    
    def extract_multiple_names(self, text: str, language: str = 'en') -> List[str]:
        """Extract multiple names from text"""
        names = []
        
        # Try pattern-based extraction
        patterns = self.name_patterns.get(language, self.name_patterns['en'])
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                potential_name = match.group(1).strip()
                if self._is_valid_name(potential_name, language):
                    name = potential_name.title()
                    if name not in names:
                        names.append(name)
        
        # If no pattern matches, try heuristic extraction
        if not names:
            name = self._extract_by_heuristics(text, language)
            if name:
                names.append(name)
        
        return names
    
    def is_name_introduction(self, text: str, language: str = 'en') -> bool:
        """Check if text contains name introduction"""
        text_lower = text.lower()
        patterns = self.name_patterns.get(language, self.name_patterns['en'])
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def get_name_context(self, text: str, language: str = 'en') -> Dict[str, Any]:
        """Get comprehensive name extraction context"""
        result = {
            'text': text,
            'language': language,
            'is_introduction': self.is_name_introduction(text, language),
            'extracted_names': self.extract_multiple_names(text, language),
            'primary_name': self.extract_name(text, language),
            'confidence': 0.0
        }
        
        # Calculate confidence based on extraction method
        if result['primary_name']:
            if result['is_introduction']:
                result['confidence'] = 0.9
            elif result['primary_name'].lower() in self.common_names.get(language, set()):
                result['confidence'] = 0.8
            else:
                result['confidence'] = 0.6
        
        return result
    
    def suggest_name_prompt(self, language: str = 'en') -> str:
        """Suggest how user can introduce their name"""
        prompts = {
            'en': "You can tell me your name by saying 'My name is...' or 'I am...' or 'Call me...'",
            'hi': "आप अपना नाम बता सकते हैं 'मेरा नाम ... है' या 'मैं ... हूं' कहकर",
            'gu': "તમે તમારું નામ કહી શકો છો 'મારું નામ ... છે' અથવા 'હું ... છું' કહીને"
        }
        
        return prompts.get(language, prompts['en'])
    
    def get_status(self) -> Dict[str, Any]:
        """Get extractor status"""
        return {
            'status': 'ready',
            'supported_languages': list(self.name_patterns.keys()),
            'pattern_count': {lang: len(patterns) for lang, patterns in self.name_patterns.items()},
            'common_names_count': {lang: len(names) for lang, names in self.common_names.items()}
        }
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up name extractor...")
        # No cleanup needed for this component
        logger.info("✅ Name extractor cleanup completed")