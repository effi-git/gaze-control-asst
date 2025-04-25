# -*- coding: utf-8 -*-
"""
NLP-based predictive text module for suggesting words based on current input.
"""
import logging
import os
import json
import re
import string
import time
import threading
from collections import Counter
import nltk

# Initialize NLTK with error handling for web environments
logger = logging.getLogger(__name__)

def initialize_nltk():
    """Initialize NLTK data, with fallback for environments where download might fail."""
    try:
        nltk.data.find('corpora/words')
        return True
    except LookupError:
        try:
            logger.info("Downloading NLTK data...")
            nltk.download('words', quiet=True)
            nltk.download('punkt', quiet=True)
            return True
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {str(e)}")
            return False

# Try to initialize NLTK
nltk_available = initialize_nltk()

# Only import these if NLTK is available
if nltk_available:
    from nltk.corpus import words as nltk_words
    from nltk.util import ngrams
else:
    # Create dummy replacements for environments where NLTK isn't available
    logger.warning("NLTK corpus not available, using fallback implementation")
    class MockNltkWords:
        def words(self):
            return ["the", "be", "to", "of", "and", "a", "in", "that", "have", 
                   "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"]
    
    nltk_words = MockNltkWords()
    
    def ngrams(tokens, n):
        """Simple ngrams implementation for fallback."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

class TextPredictor:
    """
    Provides word suggestions based on the current text input.
    Uses n-gram models and word frequencies for prediction.
    """
    def __init__(self, word_frequencies_file="data/word_frequencies.json", max_suggestions=5):
        self.max_suggestions = max_suggestions
        self.word_frequencies = {}
        self.bigram_frequencies = {}
        self.trigram_frequencies = {}
        self.user_word_history = Counter()
        self.user_bigram_history = Counter()
        self.user_history_weight = 2.0  # Weight for user's own typing history
        self.lock = threading.Lock()
        
        # Load word frequencies or initialize with common English words
        self._load_word_frequencies(word_frequencies_file)
        
        # Initialize n-grams from NLTK words corpus if no frequencies were loaded
        if not self.word_frequencies:
            self._initialize_from_nltk()
            
        logger.info("Text predictor initialized")

    def _load_word_frequencies(self, file_path):
        """Load word and n-gram frequencies from a JSON file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.word_frequencies = data.get("unigrams", {})
                    self.bigram_frequencies = data.get("bigrams", {})
                    self.trigram_frequencies = data.get("trigrams", {})
                logger.info(f"Loaded word frequencies from {file_path}")
                logger.info(f"  {len(self.word_frequencies)} words")
                logger.info(f"  {len(self.bigram_frequencies)} bigrams")
                logger.info(f"  {len(self.trigram_frequencies)} trigrams")
            else:
                logger.warning(f"Word frequencies file {file_path} not found")
        except Exception as e:
            logger.error(f"Error loading word frequencies: {str(e)}")

    def _initialize_from_nltk(self):
        """Initialize the predictor with words from NLTK corpus."""
        logger.info("Initializing predictor from NLTK corpus")
        try:
            # Get common English words
            common_words = set(w.lower() for w in nltk_words.words() if len(w) > 1)
            
            # Create basic frequencies (equal weights for now)
            for word in common_words:
                self.word_frequencies[word] = 1
                
            logger.info(f"Initialized with {len(self.word_frequencies)} words from NLTK")
        except Exception as e:
            logger.error(f"Error initializing from NLTK: {str(e)}")
            # Fallback to basic set of common words
            basic_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
                          "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"]
            for word in basic_words:
                self.word_frequencies[word.lower()] = 100

    def _save_word_frequencies(self, file_path="data/word_frequencies.json"):
        """Save word and n-gram frequencies to a JSON file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            data = {
                "unigrams": self.word_frequencies,
                "bigrams": self.bigram_frequencies,
                "trigrams": self.trigram_frequencies
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f)
                
            logger.info(f"Saved word frequencies to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving word frequencies: {str(e)}")
            return False

    def predict(self, text):
        """
        Predict the next word based on the current text input.
        Returns a list of suggested words.
        """
        with self.lock:
            if not text:
                # Return most common words if no text is provided
                return self._get_most_common_words()
            
            # Tokenize the input text
            text = text.lower().strip()
            tokens = self._tokenize(text)
            
            # If the last token is a partial word, use it for prefix matching
            prefix = ""
            context_words = tokens
            
            # Check if the text ends with a partial word
            if text and not text.endswith(' '):
                # Last token is partial, use for prefix matching
                prefix = tokens[-1] if tokens else ""
                context_words = tokens[:-1] if len(tokens) > 1 else []
            
            # Get suggestions based on n-grams and prefix
            suggestions = self._get_suggestions(context_words, prefix)
            
            return suggestions

    def _tokenize(self, text):
        """Tokenize the input text into words."""
        # Remove punctuation and split by whitespace
        text = text.translate(str.maketrans('', '', string.punctuation))
        return [word.lower() for word in text.split() if word]

    def _get_suggestions(self, context_words, prefix):
        """
        Get word suggestions based on context words and prefix.
        Uses n-grams and word frequencies to rank suggestions.
        """
        suggestions = []
        
        # Get context-based suggestions (from n-grams)
        if len(context_words) >= 2:
            # Use trigram if available
            context_bigram = ' '.join(context_words[-2:])
            for trigram, freq in self.trigram_frequencies.items():
                if trigram.startswith(context_bigram + ' '):
                    word = trigram.split()[-1]
                    if prefix and not word.startswith(prefix):
                        continue
                    suggestions.append((word, freq * 3))  # Higher weight for trigram match
        
        if len(context_words) >= 1:
            # Use bigram
            last_word = context_words[-1]
            for bigram, freq in self.bigram_frequencies.items():
                if bigram.startswith(last_word + ' '):
                    word = bigram.split()[-1]
                    if prefix and not word.startswith(prefix):
                        continue
                    suggestions.append((word, freq * 2))  # Higher weight for bigram match
            
            # Also check user's history bigrams
            for bigram, freq in self.user_bigram_history.items():
                if bigram.startswith(last_word + ' '):
                    word = bigram.split()[-1]
                    if prefix and not word.startswith(prefix):
                        continue
                    suggestions.append((word, freq * self.user_history_weight))
        
        # Get prefix-based suggestions (from word frequencies)
        if prefix:
            for word, freq in self.word_frequencies.items():
                if word.startswith(prefix) and word != prefix:
                    suggestions.append((word, freq))
            
            # Also check user's word history
            for word, freq in self.user_word_history.items():
                if word.startswith(prefix) and word != prefix:
                    suggestions.append((word, freq * self.user_history_weight))
        else:
            # Without prefix, add some common words
            for word, freq in sorted(self.word_frequencies.items(), key=lambda x: x[1], reverse=True)[:20]:
                suggestions.append((word, freq))
        
        # Sort suggestions by frequency and deduplicate
        unique_suggestions = {}
        for word, freq in suggestions:
            if word in unique_suggestions:
                unique_suggestions[word] += freq
            else:
                unique_suggestions[word] = freq
        
        # Sort by frequency and take top N
        sorted_suggestions = sorted(unique_suggestions.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_suggestions[:self.max_suggestions]]

    def _get_most_common_words(self):
        """Return the most common words as suggestions when no context is available."""
        # Combine general frequencies with user history
        combined = Counter(self.word_frequencies)
        
        # Add user history with weight
        for word, count in self.user_word_history.items():
            combined[word] += count * self.user_history_weight
        
        # Return top words
        return [word for word, _ in combined.most_common(self.max_suggestions)]

    def update_with_text(self, text):
        """
        Update the predictor with new text typed by the user.
        This improves personalization by learning from user's word choices.
        """
        with self.lock:
            try:
                # Tokenize text
                tokens = self._tokenize(text)
                if not tokens:
                    return
                
                # Update unigram frequencies
                for word in tokens:
                    if len(word) > 1:  # Ignore single-letter words
                        self.user_word_history[word] += 1
                        if word in self.word_frequencies:
                            self.word_frequencies[word] += 1
                        else:
                            self.word_frequencies[word] = 1
                
                # Update bigram frequencies
                if len(tokens) >= 2:
                    bigrams_list = list(ngrams(tokens, 2))
                    for bigram in bigrams_list:
                        bigram_str = ' '.join(bigram)
                        self.user_bigram_history[bigram_str] += 1
                        if bigram_str in self.bigram_frequencies:
                            self.bigram_frequencies[bigram_str] += 1
                        else:
                            self.bigram_frequencies[bigram_str] = 1
                
                # Update trigram frequencies
                if len(tokens) >= 3:
                    trigrams_list = list(ngrams(tokens, 3))
                    for trigram in trigrams_list:
                        trigram_str = ' '.join(trigram)
                        if trigram_str in self.trigram_frequencies:
                            self.trigram_frequencies[trigram_str] += 1
                        else:
                            self.trigram_frequencies[trigram_str] = 1
                
                logger.debug(f"Updated predictor with text: {text[:20]}...")
            except Exception as e:
                logger.error(f"Error updating predictor: {str(e)}")
