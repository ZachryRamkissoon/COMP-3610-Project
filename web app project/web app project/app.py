import flask
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import percentileofscore
import datetime 

from utils import clean_lyrics_for_model
from utils import (
    calculate_word_count, calculate_char_count,
    calculate_unique_word_count, calculate_distinct_word_count,
    calculate_avg_word_length, calculate_stopword_stats,
    calculate_sentiment_scores, calculate_lexical_diversity,
    calculate_syllable_count, calculate_rhyme_pairs,
    calculate_rhyme_density, calculate_stopword_repetition_ratio,
    calculate_true_repetition_ratio, calculate_vocab_redundancy_ratio,
    calculate_pronoun_counts, calculate_pronoun_ratios,
    count_specific_section, calculate_total_section_count,
    extract_section_pattern_string, generate_padded_section_sequences,
    MAX_SECTION_LEN, # Constant for padding length
    compute_chorus_sentiment_shift,
    compute_n_gram_repeated_phrase_intensity
)

MODEL1_PATH = 'models/oldjobs/tuned_lgbm_top20_classifier.joblib'
MODEL1_VECTORIZER_PATH = 'models/oldjobs/tfidf_vectorizer.joblib'
MODEL1_SCALER_PATH = 'models/oldjobs/standard_scaler.joblib'
MODEL1_FEATURE_NAMES_PATH = 'models/oldjobs/combined_feature_names.joblib'

MODEL2_BASE_PATH = 'models/best_regressor_all_songs/'
MODEL2_PATH = MODEL2_BASE_PATH + 'gbr_engineered_mycs_regressor.joblib'
MODEL2_VECTORIZER_PATH = MODEL2_BASE_PATH + 'tfidf_vectorizer.joblib'
MODEL2_SCALER_PATH = MODEL2_BASE_PATH + 'standard_scaler.joblib'
MODEL2_FEATURE_NAMES_PATH = MODEL2_BASE_PATH + 'combined_feature_names.joblib'

HISTORICAL_MYCS_PATH = 'models/song_mycs_scores.csv' 

try:
    print("Loading Model 1 (Top 20 Classifier) components...")
    model1_classifier = joblib.load(MODEL1_PATH)
    model1_tfidf_vectorizer = joblib.load(MODEL1_VECTORIZER_PATH)
    model1_scaler = joblib.load(MODEL1_SCALER_PATH)
    model1_combined_feature_names = joblib.load(MODEL1_FEATURE_NAMES_PATH)
    print("Model 1 components loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR loading Model 1 components: {e}. Please ensure files are in 'models/' directory.")
    exit()

try:
    print("Loading Model 2 (MYCS Regressor) components...")
    model2_regressor = joblib.load(MODEL2_PATH)
    model2_tfidf_vectorizer = joblib.load(MODEL2_VECTORIZER_PATH)
    model2_scaler = joblib.load(MODEL2_SCALER_PATH)
    model2_combined_feature_names = joblib.load(MODEL2_FEATURE_NAMES_PATH)
    print("Model 2 components loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR loading Model 2 components: {e}. Ensure files are in '{MODEL2_BASE_PATH}'.")
    exit()

historical_mycs_values = np.array([]) 
try:
    print(f"Loading historical MYCS data from {HISTORICAL_MYCS_PATH}...")
    df_historical_mycs = pd.read_csv(HISTORICAL_MYCS_PATH)
    if 'MYCS' not in df_historical_mycs.columns:
        raise ValueError("MYCS column not found in historical data CSV.")
    historical_mycs_values = df_historical_mycs['MYCS'].dropna().values
    print(f"Loaded {len(historical_mycs_values)} historical MYCS scores.")
except Exception as e:
    print(f"Warning: Error loading historical MYCS data from '{HISTORICAL_MYCS_PATH}': {e}")
    print("Percentile calculation for MYCS will not be available.")

app = flask.Flask(__name__)

def get_base_features_from_lyrics(raw_lyrics, cleaned_lyrics):
    """
    Calculates a dictionary of all base features from raw and cleaned lyrics.
    """
    features = {}
    wc = calculate_word_count(cleaned_lyrics)
    features['word_count'] = wc
    features['char_count'] = calculate_char_count(cleaned_lyrics)
    uwc = calculate_unique_word_count(cleaned_lyrics) 
    features['unique_word_count'] = uwc
    dwc = calculate_distinct_word_count(cleaned_lyrics) 
    features['distinct_word_count'] = dwc
    features['avg_word_length'] = calculate_avg_word_length(cleaned_lyrics)

    
    s_ratio, s_count, d_s_count = calculate_stopword_stats(cleaned_lyrics)
    features['stopword_ratio'] = s_ratio
    features['stopword_count'] = s_count
    features['distinct_stopword_count'] = d_s_count
    features['stopword_repetition_ratio'] = calculate_stopword_repetition_ratio(d_s_count, s_count)

    
    pos_r, neg_r, neu_r, comp_s = calculate_sentiment_scores(cleaned_lyrics)
    features['pos_ratio'] = pos_r
    features['neg_ratio'] = neg_r
    features['neu_ratio'] = neu_r
    features['compound'] = comp_s
    
    
    features['lexical_diversity'] = calculate_lexical_diversity(uwc, wc)
    features['vocab_redundancy_ratio'] = calculate_vocab_redundancy_ratio(uwc, wc) 
    
    
    features['syllable_count'] = calculate_syllable_count(cleaned_lyrics)
    rp = calculate_rhyme_pairs(cleaned_lyrics)
    features['rhyme_pairs'] = rp
    features['rhyme_density'] = calculate_rhyme_density(rp, dwc)
    
    
    features['true_repetition_ratio'] = calculate_true_repetition_ratio(cleaned_lyrics)

    
    pronoun_counts = calculate_pronoun_counts(raw_lyrics)
    pronoun_ratios_dict = calculate_pronoun_ratios(pronoun_counts, wc) 
    features.update(pronoun_counts) 
    features.update(pronoun_ratios_dict) 

    
    features['verse_count'] = count_specific_section(raw_lyrics, r'\[Verse')
    features['intro_count'] = count_specific_section(raw_lyrics, r'\[Intro')
    features['outro_count'] = count_specific_section(raw_lyrics, r'\[Outro')
    features['bridge_count'] = count_specific_section(raw_lyrics, r'\[Bridge')
    features['chorus_count'] = count_specific_section(raw_lyrics, r'\[Chorus')
    features['prechorus_count'] = count_specific_section(raw_lyrics, r'\[Pre-Chorus')
    features['postchorus_count'] = count_specific_section(raw_lyrics, r'\[Post-Chorus')
    total_sections = calculate_total_section_count(raw_lyrics)
    features['total_section_count'] = total_sections
    
    features['chorus_ratio'] = round(features['chorus_count'] / total_sections, 4) if total_sections > 0 else 0.0
    features['prechorus_ratio'] = round(features['prechorus_count'] / total_sections, 4) if total_sections > 0 else 0.0
    features['verse_ratio'] = round(features['verse_count'] / total_sections, 4) if total_sections > 0 else 0.0
    features['bridge_ratio'] = round(features['bridge_count'] / total_sections, 4) if total_sections > 0 else 0.0
    
    section_pattern_str = extract_section_pattern_string(raw_lyrics)
    features['pattern'] = section_pattern_str 
    padded_section_nums = generate_padded_section_sequences(section_pattern_str)
    for i in range(MAX_SECTION_LEN):
        features[f'section_{i}'] = padded_section_nums[i]
    
    
    features['chorus_sentiment_shift'] = compute_chorus_sentiment_shift(raw_lyrics)
    
    
    trigram_intensity = compute_n_gram_repeated_phrase_intensity(raw_lyrics, 3)
    features['repeated_phrase_intensity'] = trigram_intensity 
    features['trigram_repeated_phrase_intensity'] = trigram_intensity
    features['bigram_repeated_phrase_intensity'] = compute_n_gram_repeated_phrase_intensity(raw_lyrics, 2)
    
    return features


@app.route('/', methods=['GET'])
def home():
    """Renders the home page for Top 20 Prediction."""
    return flask.render_template(
        'index.html', 
        current_year=datetime.date.today().year,
        lyrics_text='',        
        prediction_text=None,  
        probability_text=None, 
        error_message=None     
    )

@app.route('/mycs', methods=['GET'])
def mycs_predictor_page(): 
    """Renders the page for MYCS Prediction."""
    return flask.render_template(
        'mycs_predictor.html', 
        current_year=datetime.date.today().year,
        lyrics_text='',
        prediction_mycs=None,
        percentile=None,
        rank_estimate=None,
        historical_song_count=len(historical_mycs_values),
        error_message=None
    )

@app.route('/predict', methods=['POST']) 
def predict():
    lyrics_text_to_render = flask.request.form.get('lyrics', '') 
    prediction_to_render = None
    probability_to_render = None
    error_to_render = None
    try:
        raw_lyrics_text = flask.request.form.get('lyrics') 
        if not raw_lyrics_text or not raw_lyrics_text.strip():
            error_to_render = "Lyrics input cannot be empty."
        else:
            cleaned_lyrics = clean_lyrics_for_model(raw_lyrics_text)
            
            
            model1_lyrics_tfidf_sparse = model1_tfidf_vectorizer.transform([cleaned_lyrics])
            model1_tfidf_names = model1_tfidf_vectorizer.get_feature_names_out()
            model1_lyrics_tfidf_df = pd.DataFrame(model1_lyrics_tfidf_sparse.toarray(), columns=model1_tfidf_names)
            
            other_features_values = get_base_features_from_lyrics(raw_lyrics_text, cleaned_lyrics)
            
            model1_non_tfidf_feature_names = [name for name in model1_combined_feature_names if name not in model1_tfidf_names]
            
            model1_current_other_features = {}
            for feature_name in model1_non_tfidf_feature_names:
                model1_current_other_features[feature_name] = other_features_values.get(feature_name, 0.0) # Default to 0.0 if missing
            
            model1_other_features_df = pd.DataFrame([model1_current_other_features], columns=model1_non_tfidf_feature_names) # Ensures order

            
            model1_combined_df = pd.DataFrame(0.0, index=[0], columns=model1_combined_feature_names)
            
            for col in model1_lyrics_tfidf_df.columns:
                if col in model1_combined_df.columns:
                    model1_combined_df.loc[0, col] = model1_lyrics_tfidf_df.loc[0, col]
            
            for col in model1_other_features_df.columns:
                if col in model1_combined_df.columns:
                    model1_combined_df.loc[0, col] = model1_other_features_df.loc[0, col]
            
            model1_X_scaled = model1_scaler.transform(model1_combined_df) # Scale
            
            prediction_val = model1_classifier.predict(model1_X_scaled)[0]
            probability_val = model1_classifier.predict_proba(model1_X_scaled)[0]
            prediction_to_render = "Likely Top 20 Hit!" if prediction_val == 1 else "Not Likely a Top 20 Hit."
            probability_to_render = f"Probability of being a Top 20 hit: {probability_val[1]:.2%}"

    except Exception as e:
        print(f"Error during Model 1 prediction: {e}", flush=True); import traceback; traceback.print_exc()
        error_to_render = f"An error occurred during prediction: {str(e)}"
    
    return flask.render_template('index.html',
                                 prediction_text=prediction_to_render,
                                 probability_text=probability_to_render,
                                 error_message=error_to_render,
                                 lyrics_text=lyrics_text_to_render, 
                                 current_year=datetime.date.today().year)


@app.route('/predict_mycs', methods=['POST']) 
def predict_mycs():
    lyrics_text_to_render = flask.request.form.get('lyrics', '') 
    mycs_to_render = None
    percentile_to_render = None
    rank_to_render = None
    error_to_render = None
    
    try:
        raw_lyrics_text = flask.request.form.get('lyrics') 
        if not raw_lyrics_text or not raw_lyrics_text.strip():
            error_to_render = "Lyrics input cannot be empty."
        else:
            cleaned_lyrics = clean_lyrics_for_model(raw_lyrics_text)
            base_features = get_base_features_from_lyrics(raw_lyrics_text, cleaned_lyrics)
            
            tsc = base_features.get('total_section_count', 0)
            css = base_features.get('chorus_sentiment_shift', 0.0)
            
            engineered_features_for_model2 = base_features.copy()
            engineered_features_for_model2['interact_tsc_css'] = tsc * css
            engineered_features_for_model2['tsc_sq'] = tsc**2
            engineered_features_for_model2['css_sq'] = css**2
            
            model2_lyrics_tfidf_sparse = model2_tfidf_vectorizer.transform([cleaned_lyrics])
            model2_tfidf_names = model2_tfidf_vectorizer.get_feature_names_out()
            model2_lyrics_tfidf_df = pd.DataFrame(model2_lyrics_tfidf_sparse.toarray(), columns=model2_tfidf_names)

            model2_non_tfidf_feature_names = [name for name in model2_combined_feature_names if name not in model2_tfidf_names]
            
            model2_current_other_features = {}
            for feature_name in model2_non_tfidf_feature_names:
                model2_current_other_features[feature_name] = engineered_features_for_model2.get(feature_name, 0.0)
            
            model2_other_features_df = pd.DataFrame([model2_current_other_features], columns=model2_non_tfidf_feature_names) # Ensures order

            model2_combined_df = pd.DataFrame(0.0, index=[0], columns=model2_combined_feature_names)
            for col in model2_lyrics_tfidf_df.columns:
                if col in model2_combined_df.columns:
                    model2_combined_df.loc[0, col] = model2_lyrics_tfidf_df.loc[0, col]
            for col in model2_other_features_df.columns:
                if col in model2_combined_df.columns:
                    model2_combined_df.loc[0, col] = model2_other_features_df.loc[0, col]
            
            model2_X_scaled = model2_scaler.transform(model2_combined_df) # Scale

            mycs_to_render = model2_regressor.predict(model2_X_scaled)[0]
            
            if historical_mycs_values.size > 0:
                percentile_to_render = percentileofscore(historical_mycs_values, mycs_to_render, kind='rank')
                num_historical_songs = len(historical_mycs_values)
                estimated_rank = int((1 - (percentile_to_render / 100)) * num_historical_songs) + 1
                rank_to_render = f"~{estimated_rank}"
            else:
                percentile_to_render = "N/A (no historical data)"
                rank_to_render = "N/A"


    except Exception as e:
        print(f"Error during Model 2 prediction: {e}", flush=True); import traceback; traceback.print_exc()
        error_to_render = f"An error occurred during MYCS prediction: {str(e)}"
            
    return flask.render_template('mycs_predictor.html',
                                 prediction_mycs=mycs_to_render,
                                 percentile=percentile_to_render,
                                 rank_estimate=rank_to_render,
                                 historical_song_count=len(historical_mycs_values),
                                 lyrics_text=lyrics_text_to_render,
                                 error_message=error_to_render,
                                 current_year=datetime.date.today().year)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
