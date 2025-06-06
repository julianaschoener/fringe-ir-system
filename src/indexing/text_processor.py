# src/indexing/text_processor.py

# src/indexing/text_preprocessor.py

import re

class TextProcessor:
    """Handles text preprocessing and tokenization"""

    @staticmethod
    def preprocess(text):
        """Basic text preprocessing"""
        if not text:
            return []

        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()
        return [word for word in words if len(word) > 1]  # Remove single characters

    @staticmethod
    def preprocess_with_bigrams(text):
        """Preprocessing with bigram support"""
        words = TextPreprocessor.preprocess(text)
        bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
        return words + bigrams
