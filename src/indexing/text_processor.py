import re

class TextProcessor:
    @staticmethod
    def preprocess(text):
        if not text:
            return []

        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        words = text.split()
        return [word for word in words if len(word) > 1]  # Remove single characters

    @staticmethod
    def preprocess_with_bigrams(text):
        """Preprocessing with bigram support"""
        words = TextProcessor.preprocess(text)
        bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
        return words + bigrams