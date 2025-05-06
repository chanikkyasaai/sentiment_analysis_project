from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
from pathlib import Path
import scipy.sparse as sp
import re

app = Flask(__name__)

# Load the pre-trained model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    print("Successfully loaded sentiment model and vectorizer")
except FileNotFoundError:
    print("Model or vectorizer file not found. Make sure 'model.pkl' and 'vectorizer.pkl' exist.")

# Import preprocessing function or define it inline if the file doesn't exist
try:
    from preprocess_function import preprocess_text
    print("Using imported preprocessing function")
except ImportError:
    print("Preprocessing function module not found. Using built-in preprocessing.")
    import re
    
    def preprocess_text(text):
        """Clean and normalize text for sentiment analysis"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and replace with space
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # List of stopwords to remove
        stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
            'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
            'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
            'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
            'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
            'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
            'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
            'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'course', 'class', 'professor', 'teacher', 'student', 'lecture', 'assignment',
            'would', 'could', 'should', 'also', 'one', 'two', 'three', 'many'
        }
        
        # Split into words and remove stopwords
        words = text.split()
        words = [word for word in words if word not in stopwords]
        
        # Very simple stemming
        stemmed_words = []
        for word in words:
            if word.endswith('ing'):
                word = word[:-3]
            elif word.endswith('ed'):
                word = word[:-2]
            elif word.endswith('ly'):
                word = word[:-2]
            elif word.endswith('s') and not word.endswith('ss'):
                word = word[:-1]
            stemmed_words.append(word)
        
        return ' '.join(stemmed_words)

# Function to extract important features that influenced the classification
def extract_important_features(text, prediction_label):
    """Extract keywords that were most influential in the classification decision"""
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    
    # Get the feature names (words) from the vectorizer
    try:
        feature_names = vectorizer.get_feature_names_out()
    except:
        # For older sklearn versions
        try:
            feature_names = vectorizer.get_feature_names()
        except:
            return ["Could not extract feature names from vectorizer"]
    
    # Transform the text to get the feature vector
    X = vectorizer.transform([preprocessed_text])
    
    # Get non-zero feature indices and values
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    
    # Store words and their values
    word_importance = []
    
    # For each non-zero feature, get its name and value
    for i, idx in enumerate(X.nonzero()[1]):
        word = feature_names[idx]
        value = X[0, idx]
        word_importance.append((word, value))
    
    # Sort by absolute value (importance)
    word_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Get top words (limit to 15)
    top_words = word_importance[:15]
    
    # Format the results
    result = []
    for word, value in top_words:
        # Try to find the word or a similar form in the original text for better context
        original_word = find_original_form(text, word)
        result.append(f"{original_word} ({value:.4f})")
    
    return result

# Helper function to find original form of a word in the text
def find_original_form(text, processed_word):
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # First try exact match
    if processed_word in words:
        return processed_word
    
    # Try to match the beginning of the word (for stemming cases)
    for word in words:
        if word.startswith(processed_word) or processed_word.startswith(word):
            return word
    
    # Return the processed word if no match found
    return processed_word

# Statistics for visualization
sentiment_stats = {
    'Positive': 0,
    'Negative': 0,
    'Neutral': 0
}

# Debug information to track predictions
debug_info = []

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    feedback_text = ""
    raw_prediction = None
    confidence_scores = None
    raw_text = None
    preprocessed_text = None
    important_keywords = None
    
    if request.method == 'POST':
        feedback_text = request.form['feedback']
        
        if feedback_text:
            # Store original text for display
            raw_text = feedback_text
            
            # Preprocess the text
            preprocessed_text = preprocess_text(feedback_text)
            
            # Vectorize the preprocessed text
            features = vectorizer.transform([preprocessed_text])
            
            # Make prediction
            prediction_result = model.predict(features)[0]
            raw_prediction = int(prediction_result)
            
            # Map the prediction to a sentiment label
            if prediction_result == 1:
                prediction = 'Positive'
                sentiment_stats['Positive'] += 1
            elif prediction_result == -1:
                prediction = 'Negative'
                sentiment_stats['Negative'] += 1
            else:
                prediction = 'Neutral'
                sentiment_stats['Neutral'] += 1
            
            # Extract important keywords that influenced the decision
            important_keywords = extract_important_features(feedback_text, prediction)
                
            # Get decision values for better understanding
            try:
                if hasattr(model, 'predict_proba'):
                    proba_scores = model.predict_proba(features)[0]
                    confidence_scores = {
                        "Negative": f"{proba_scores[0]:.4f}" if len(proba_scores) > 0 else "N/A",
                        "Neutral": f"{proba_scores[1]:.4f}" if len(proba_scores) > 1 else "N/A",
                        "Positive": f"{proba_scores[2]:.4f}" if len(proba_scores) > 2 else "N/A"
                    }
                else:
                    decision_values = model.decision_function(features)[0]
                    confidence_scores = {
                        "Decision Values": f"{decision_values}"
                    }
            except Exception as e:
                confidence_scores = {
                    "Note": f"Could not get confidence scores: {str(e)}"
                }
                
            # Store debug info for the latest predictions
            debug_entry = {
                "text": raw_text[:50] + "..." if len(raw_text) > 50 else raw_text,
                "preprocessed": preprocessed_text[:50] + "..." if len(preprocessed_text) > 50 else preprocessed_text,
                "raw_prediction": raw_prediction,
                "sentiment": prediction,
                "keywords": important_keywords[:3] if important_keywords else []
            }
            debug_info.append(debug_entry)
            if len(debug_info) > 5:  # Keep only the last 5 entries
                debug_info.pop(0)
    
    # Check if wordcloud exists
    wordcloud_path = Path('static/wordcloud.png')
    has_wordcloud = wordcloud_path.exists()
    
    # Check if confusion matrix exists
    confusion_matrix_path = Path('static/confusion_matrix.png')
    has_confusion_matrix = confusion_matrix_path.exists()
    
    return render_template('index.html', 
                          prediction=prediction, 
                          feedback_text=feedback_text,
                          stats=sentiment_stats,
                          has_wordcloud=has_wordcloud,
                          raw_prediction=raw_prediction,
                          confidence_scores=confidence_scores,
                          debug_info=debug_info,
                          raw_text=raw_text,
                          preprocessed_text=preprocessed_text,
                          important_keywords=important_keywords,
                          has_confusion_matrix=has_confusion_matrix)

@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify(sentiment_stats)

if __name__ == '__main__':
    app.run(debug=True)