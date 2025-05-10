import re
import pandas as pd
import numpy as np
from collections import Counter

import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("NLTK 'stopwords' corpus not found. Downloading...")
    nltk.download('stopwords', quiet=True) 
try:
    nltk.data.find('sentiment/vader_lexicon.zip') 
except LookupError:
    print("NLTK 'vader_lexicon' not found. Downloading...")
    nltk.download('vader_lexicon', quiet=True) 

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import syllapy 
import pronouncing 

stop_words_set = set(stopwords.words('english'))
sentiment_analyzer = SentimentIntensityAnalyzer()

FIRST_PERSON_PRONOUNS = [
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "i'm", "i've", "i'd", "i'll", "we're", "we've", "we'd", "we'll", "myself", "ourselves"
]
SECOND_PERSON_PRONOUNS = [
    "you", "your", "yours", "u", "ya", "yall", "y'all",
    "you're", "you've", "you'd", "you'll", "yourself", "yourselves"
]
THIRD_PERSON_PRONOUNS = [
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs",
    "he's", "he'd", "he'll", "she's", "she'd", "she'll", "it's", "it'll",
    "they're", "they've", "they'd", "they'll", "himself", "herself", "itself", "themselves"
]
MALE_PRONOUNS = ["he", "him", "his", "he's", "he'd", "he'll", "himself"]
FEMALE_PRONOUNS = ["she", "her", "hers", "she's", "she'd", "she'll", "herself"]


SECTION_MAP = {
    'I': 0, 'V': 1, 'PC': 2, 'C': 3, 'B': 4, 'POC': 5, 'O': 6,
    'H': 7, 'INSTR': 8, 'S': 9, 'UNK': -1 
}
MAX_SECTION_LEN = 12 


def clean_lyrics_for_model(text: str) -> str:
    """
    Cleans raw song lyrics text for TF-IDF and general linguistic features.
    Removes section headers, converts to lowercase, removes non-ASCII,
    removes apostrophes, and most punctuation.
    """
    if not isinstance(text, str) or pd.isna(text):
        return ''
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = text.lower() 
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) 
    text = text.replace("'", "") 
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\d+embed$', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    return text


def calculate_word_count(cleaned_lyrics: str) -> int:
    if not cleaned_lyrics: return 0
    return len(cleaned_lyrics.split())

def calculate_char_count(cleaned_lyrics: str) -> int:
    return len(cleaned_lyrics)

def calculate_unique_word_count(cleaned_lyrics: str) -> int: 
    if not cleaned_lyrics: return 0
    words = cleaned_lyrics.split()
    if not words: return 0
    counts = Counter(words)
    return sum(1 for count in counts.values() if count == 1)

def calculate_distinct_word_count(cleaned_lyrics: str) -> int: 
    if not cleaned_lyrics: return 0
    words = cleaned_lyrics.split()
    if not words: return 0
    return len(set(words))

def calculate_avg_word_length(cleaned_lyrics: str) -> float:
    if not cleaned_lyrics: return 0.0
    words = cleaned_lyrics.split()
    if not words: return 0.0
    return np.mean([len(word) for word in words])

def calculate_stopword_stats(cleaned_lyrics: str) -> tuple[float, int, int]:
    if not cleaned_lyrics: return 0.0, 0, 0
    words = cleaned_lyrics.split(); total_words = len(words)
    if total_words == 0: return 0.0, 0, 0
    current_stopwords = [word for word in words if word.lower() in stop_words_set]
    stopword_count_val = len(current_stopwords)
    distinct_stopword_count_val = len(set(current_stopwords)) if stopword_count_val > 0 else 0
    stopword_ratio_val = round(stopword_count_val / total_words, 4) if total_words > 0 else 0.0
    return stopword_ratio_val, stopword_count_val, distinct_stopword_count_val

def calculate_sentiment_scores(cleaned_lyrics: str) -> tuple[float, float, float, float]:
    if not cleaned_lyrics: return 0.0, 0.0, 0.0, 0.0
    scores = sentiment_analyzer.polarity_scores(cleaned_lyrics)
    return scores['pos'], scores['neg'], scores['neu'], scores['compound']

def calculate_lexical_diversity(unique_word_count_val: int, total_words: int) -> float:
    if total_words == 0: return 0.0
    return round(unique_word_count_val / total_words, 4)

def calculate_syllable_count(cleaned_lyrics: str) -> int:
    if not cleaned_lyrics: return 0
    try:
        return sum(syllapy.count(word) for word in cleaned_lyrics.split() if word)
    except Exception as e:
        print(f"Warning: Syllapy error for text starting with '{cleaned_lyrics[:30]}...': {e}")
        return 0

def calculate_rhyme_pairs(cleaned_lyrics: str) -> int:
    if not cleaned_lyrics: return 0
    words = [word for word in cleaned_lyrics.split() if word]
    if not words: return 0
    unique_words_list = sorted(list(set(words)))
    rhyme_pairs_count = 0
    for i, word1 in enumerate(unique_words_list):
        try:
            rhymes_for_word1 = pronouncing.rhymes(word1)
            if not rhymes_for_word1: continue
            for j in range(i + 1, len(unique_words_list)):
                word2 = unique_words_list[j]
                if word2 in rhymes_for_word1:
                    rhyme_pairs_count += 1
        except Exception as e:
            continue
    return rhyme_pairs_count

def calculate_rhyme_density(rhyme_pairs_val: int, distinct_words_val: int) -> float:
    if distinct_words_val == 0: return 0.0
    density = rhyme_pairs_val / distinct_words_val
    return round(density, 4) if pd.notna(density) and np.isfinite(density) else 0.0

def calculate_stopword_repetition_ratio(distinct_stopwords_val: int, total_stopwords: int) -> float:
    if total_stopwords == 0: return 0.0
    return round(1 - (distinct_stopwords_val / total_stopwords), 4)

def calculate_true_repetition_ratio(cleaned_lyrics: str) -> float:
    if not cleaned_lyrics: return 0.0
    words = cleaned_lyrics.split(); total_words = len(words)
    if total_words == 0: return 0.0
    counts = Counter(words)
    repeated_word_occurrences = sum(count for word, count in counts.items() if count > 1)
    return round(repeated_word_occurrences / total_words, 4)

def calculate_vocab_redundancy_ratio(unique_word_count_val: int, total_words: int) -> float:
    if total_words == 0: return 0.0
    return round(1 - (unique_word_count_val / total_words), 4)

def calculate_pronoun_counts(raw_lyrics: str) -> dict:
    text_to_process = str(raw_lyrics).lower() if isinstance(raw_lyrics, str) and not pd.isna(raw_lyrics) else ""
    text_to_process = re.sub(r"[^\w\s']", "", text_to_process) 
    words = text_to_process.split()

    counts = {
        "first_person": sum(word in FIRST_PERSON_PRONOUNS for word in words),
        "second_person": sum(word in SECOND_PERSON_PRONOUNS for word in words),
        "third_person": sum(word in THIRD_PERSON_PRONOUNS for word in words),
        "male_pronouns": sum(word in MALE_PRONOUNS for word in words),
        "female_pronouns": sum(word in FEMALE_PRONOUNS for word in words),
    }
    counts["total_pronouns"] = counts["first_person"] + counts["second_person"] + counts["third_person"]
    return counts

def calculate_pronoun_ratios(pronoun_counts: dict, total_words_in_cleaned_lyrics: int) -> dict:
    ratios = {}
    total_pronouns = pronoun_counts.get("total_pronouns", 0)
    ratios["pronoun_word_ratio"] = round(total_pronouns / total_words_in_cleaned_lyrics, 4) if total_words_in_cleaned_lyrics > 0 else 0.0
    ratios["first_person_ratio"] = round(pronoun_counts.get("first_person", 0) / total_pronouns, 4) if total_pronouns > 0 else 0.0
    ratios["second_person_ratio"] = round(pronoun_counts.get("second_person", 0) / total_pronouns, 4) if total_pronouns > 0 else 0.0
    ratios["third_person_ratio"] = round(pronoun_counts.get("third_person", 0) / total_pronouns, 4) if total_pronouns > 0 else 0.0
    ratios["male_pronoun_ratio"] = round(pronoun_counts.get("male_pronouns", 0) / total_pronouns, 4) if total_pronouns > 0 else 0.0
    ratios["female_pronoun_ratio"] = round(pronoun_counts.get("female_pronouns", 0) / total_pronouns, 4) if total_pronouns > 0 else 0.0
    return ratios

def count_specific_section(raw_lyrics: str, section_regex_pattern: str) -> int:
    """Counts occurrences of a specific section pattern (e.g., r'\[Verse'). Case-insensitive."""
    if not isinstance(raw_lyrics, str) or pd.isna(raw_lyrics): return 0
    return len(re.findall(section_regex_pattern, raw_lyrics, re.IGNORECASE))

def calculate_total_section_count(raw_lyrics: str) -> int:
    """Counts total number of all section markers like [Anything]."""
    if not isinstance(raw_lyrics, str) or pd.isna(raw_lyrics): return 0
    return len(re.findall(r'\[.*?\]', raw_lyrics))

def standardize_section_name_for_pattern(section_name_tag: str) -> str:
    """Standardizes section names from tags (e.g., '[Verse 1]') to short codes (V, C, etc.)."""
    name = section_name_tag.strip().lower() 
    if "pre-chorus" in name or "prechorus" in name: return "PC"
    if "post-chorus" in name or "postchorus" in name: return "POC"
    if "chorus" in name: return "C"
    if "verse" in name: return "V"
    if "bridge" in name: return "B"
    if "intro" in name: return "I"
    if "outro" in name: return "O"
    if "hook" in name: return "H"
    if "instrumental" in name or "inst" in name: return "INSTR"
    if "solo" in name: return "S"
    return "UNK" 

def extract_section_pattern_string(raw_lyrics: str) -> str:
    """Extracts a space-separated string of standardized section codes in order of appearance."""
    if pd.isna(raw_lyrics) or not isinstance(raw_lyrics, str): return ""
    section_tags = re.findall(r'\[.*?\]', raw_lyrics)
    ordered_sections = [standardize_section_name_for_pattern(tag) for tag in section_tags]
    return " ".join(ordered_sections)

def numpy_pad_sequence(sequence: list[int], maxlen: int, padding_value: int = -1, padding: str = 'post', truncating: str = 'post') -> list[int]:
    """Pads/truncates a single sequence to a specific length."""
    if len(sequence) > maxlen:
        return sequence[:maxlen] if truncating == 'post' else sequence[-maxlen:]
    elif len(sequence) < maxlen:
        pad_needed = maxlen - len(sequence)
        pad_array = [padding_value] * pad_needed
        return sequence + pad_array if padding == 'post' else pad_array + sequence
    return sequence

def generate_padded_section_sequences(section_pattern_str: str) -> list[int]:
    """Converts section pattern string to a padded sequence of integers using NumPy logic."""
    if not section_pattern_str:
        sequence_integers = []
    else:
        sequence_integers = [SECTION_MAP.get(part, SECTION_MAP['UNK']) for part in section_pattern_str.split()]
    
    return numpy_pad_sequence(
        sequence_integers,
        maxlen=MAX_SECTION_LEN,
        padding_value=SECTION_MAP['UNK'],
        padding='post',
        truncating='post'
    )

def compute_chorus_sentiment_shift(raw_lyrics: str) -> float:
    if not isinstance(raw_lyrics, str) or not raw_lyrics.strip() or pd.isna(raw_lyrics):
        return 0.0

    sections_content = {}
    current_tag_normalized = None
    
    parts = re.split(r'(\[.*?\])', raw_lyrics)
    
    for part in parts:
        if not part or not part.strip():
            continue
        if re.fullmatch(r'\[.*?\]', part.strip()): 
            current_tag_normalized = standardize_section_name_for_pattern(part.strip())
            if current_tag_normalized not in sections_content:
                sections_content[current_tag_normalized] = []
        elif current_tag_normalized: 
            cleaned_part_text = clean_lyrics_for_model(part.strip())
            if cleaned_part_text: 
                 sections_content[current_tag_normalized].append(cleaned_part_text)
    
    chorus_sentiments = []
    non_chorus_sentiments = []

    for tag, texts in sections_content.items():
        if not texts: continue
       
        full_section_text = " ".join(texts)
        if not full_section_text.strip(): continue

        section_compound_score = sentiment_analyzer.polarity_scores(full_section_text)['compound']
        
        if tag == 'C': 
            chorus_sentiments.append(section_compound_score)
        else:
            non_chorus_sentiments.append(section_compound_score)

    avg_chorus_sentiment = np.mean(chorus_sentiments) if chorus_sentiments else 0.0
    avg_non_chorus_sentiment = np.mean(non_chorus_sentiments) if non_chorus_sentiments else 0.0

    if not chorus_sentiments or not non_chorus_sentiments: 
        return 0.0 
        
    return round(avg_chorus_sentiment - avg_non_chorus_sentiment, 4)


def compute_n_gram_repeated_phrase_intensity(raw_lyrics: str, n: int) -> float:
    """Computes the intensity of repeated n-grams from cleaned lyrics."""
    if not isinstance(raw_lyrics, str) or not raw_lyrics.strip() or pd.isna(raw_lyrics):
        return 0.0

    lyrics_for_ngrams = clean_lyrics_for_model(raw_lyrics)
    words = lyrics_for_ngrams.split()

    if len(words) < n: return 0.0 

    ngrams_list = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    total_ngrams = len(ngrams_list)
    if total_ngrams == 0: return 0.0

    counts = Counter(ngrams_list)
    num_repeated_unique_ngrams = sum(1 for count_val in counts.values() if count_val > 1)
    
    
    return round(num_repeated_unique_ngrams / total_ngrams, 4) if total_ngrams > 0 else 0.0

if __name__ == '__main__':
    print("--- Testing utils.py ---")
    sample_raw = """
    [Intro] Yeah, it's an intro. I'm feeling good.
    [Verse 1] This is the first verse, talking 'bout my life. I've seen things.
    You know what I mean? He's a friend. She's cool. They're all here.
    [Pre-Chorus] Getting ready, oh yeah.
    [Chorus] This is the happy chorus! Happy, happy, joy, joy! We love it.
    [Verse 2] Second verse, a bit more reflective. It's a journey.
    [Chorus] This is the happy chorus! Happy, happy, joy, joy! We love it. Again!
    [Bridge] A change of pace, a different view.
    [Outro] Waving goodbye now. Sad to go.
    A final line happy happy happy. Another line happy happy happy.
    This has non-ASCII: éàçüö and some punctuation!! Also 123embed.
    """
    sample_cleaned = clean_lyrics_for_model(sample_raw)
    print(f"\nCleaned Lyrics:\n'{sample_cleaned}'")

    wc = calculate_word_count(sample_cleaned)
    print(f"\nWord Count: {wc}")
    # ... (call and print other base features)

    p_counts = calculate_pronoun_counts(sample_raw)
    print(f"\nPronoun Counts: {p_counts}")
    p_ratios = calculate_pronoun_ratios(p_counts, wc)
    print(f"Pronoun Ratios: {p_ratios}")

    print(f"\nVerse Count: {count_specific_section(sample_raw, r'\[Verse')}")
    pattern_str = extract_section_pattern_string(sample_raw)
    print(f"Section Pattern String: '{pattern_str}'")
    padded_seq = generate_padded_section_sequences(pattern_str)
    print(f"Padded Section Sequence: {padded_seq}")

    chorus_shift = compute_chorus_sentiment_shift(sample_raw)
    print(f"\nChorus Sentiment Shift: {chorus_shift:.4f}")

    bigram_intensity = compute_n_gram_repeated_phrase_intensity(sample_raw, 2)
    trigram_intensity = compute_n_gram_repeated_phrase_intensity(sample_raw, 3)
    print(f"Bigram Repeated Phrase Intensity: {bigram_intensity:.4f}")
    print(f"Trigram Repeated Phrase Intensity: {trigram_intensity:.4f}")
    print("\n--- End of utils.py tests ---")
