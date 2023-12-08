import string
import nltk, re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')

def clean_text(text):
    
    # Lowercase while maintain CAPITALIZATION
    caps_pattern = re.compile(r'\b[A-Z]{2,}\b') # RE to identify words in all caps
    caps_words = {word: f"CAPS_{idx}" for idx, word in enumerate(caps_pattern.findall(text))}
    for word, tag in caps_words.items():
        text = text.replace(word, tag)
    
    text = text.lower()
    for word, tag in caps_words.items():
        text = text.replace(tag.lower(), word)
    
    # Remove punctuation while maintain ?!
    punctuation_to_keep = "!?"
    remove_punct_pattern = f"[{re.escape(string.punctuation).translate({ord(c): None for c in punctuation_to_keep})}]"
    text = re.sub(remove_punct_pattern, '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    cleaned_text = ' '.join(tokens)
    
    return cleaned_text


if __name__ == "__main__":
    text = "I don't like this movie (at all), but the cinematography is GREAT! Really?"
    cleaned = clean_text(text)
    print(cleaned)