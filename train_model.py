import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

print("Training improved sentiment analysis model...")

# Simple but effective text preprocessing function
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
    
    return text

# Create custom stopwords list
custom_stopwords = ['course', 'class', 'professor', 'teacher', 'student', 'lecture', 
                   'assignment', 'would', 'could', 'should', 'also', 'one', 'two', 'many']
                   
# Combine with English stopwords
all_stopwords = list(ENGLISH_STOP_WORDS) + custom_stopwords

# Load the dataset
try:
    df = pd.read_csv('feedback_data.csv')
    print(f"Loaded {len(df)} feedback entries")
    
    # Make sure sentiment labels are correct
    # Map strings to integers if needed
    if df['sentiment'].dtype == object:
        sentiment_map = {
            '-1': -1, 'negative': -1, 'Negative': -1,
            '0': 0, 'neutral': 0, 'Neutral': 0,
            '1': 1, 'positive': 1, 'Positive': 1
        }
        df['sentiment'] = df['sentiment'].map(sentiment_map)
    
    # Ensure sentiment is integer type
    df['sentiment'] = df['sentiment'].astype(int)
    
    # Display label distribution
    print("\nSentiment Label Distribution:")
    label_counts = df['sentiment'].value_counts().sort_index()
    print(label_counts)
    
    # Preprocess the text data
    print("\nPreprocessing text data...")
    df['cleaned_text'] = df['feedback'].apply(preprocess_text)
    
    # Display a few examples of preprocessing results
    print("\nPreprocessing Examples:")
    for i in range(min(3, len(df))):
        print(f"Original: {df['feedback'].iloc[i][:50]}...")
        print(f"Cleaned : {df['cleaned_text'].iloc[i][:50]}...")
        print(f"Label   : {df['sentiment'].iloc[i]}")
        print("---")
    
    # Prepare features and target
    X = df['cleaned_text']
    y = df['sentiment']
    
    # Create train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and fit TF-IDF vectorizer with improved parameters
    print("\nCreating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=1,          # Include terms that appear at least once
        max_df=0.9,        # Exclude terms that appear in >90% of documents
        ngram_range=(1, 3), # Include up to trigrams for better context
        stop_words=all_stopwords,  # Use our combined stopwords list
        use_idf=True,
        norm='l2'
    )
    
    # Transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Try different C values (regularization strength)
    c_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    best_model = None
    best_f1 = 0
    best_c = None
    
    print("\nTesting different regularization strengths (C values):")
    for c in c_values:
        # Train logistic regression model
        model = LogisticRegression(
            C=c,
            class_weight='balanced',
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_tfidf)
        f1 = f1_score(y_test, y_pred, average='weighted')
        acc = accuracy_score(y_test, y_pred)
        print(f"  C={c}: Accuracy={acc:.4f}, F1={f1:.4f}")
        
        # Keep track of best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_c = c
    
    print(f"\nBest C value: {best_c} with F1 score: {best_f1:.4f}")
    
    # Use the best model
    model = best_model
    
    # Final evaluation
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nFinal Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png')
    print("Confusion matrix saved to 'static/confusion_matrix.png'")
    
    # Test model on example feedback
    print("\nTesting model on some examples:")
    examples = [
        "The course was extremely well-organized, and the instructor made every topic engaging.",
        "The class was average, nothing special about it.",
        "I struggled to keep up with the course because the pace was too fast."
    ]
    
    # Preprocess examples
    examples_cleaned = [preprocess_text(ex) for ex in examples]
    
    # Transform and predict
    examples_tfidf = vectorizer.transform(examples_cleaned)
    example_preds = model.predict(examples_tfidf)
    
    # Show predictions
    sentiment_labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    for example, pred in zip(examples, example_preds):
        print(f"Text: '{example[:50]}...'")
        print(f"Prediction: {sentiment_labels[pred]}\n")
    
    # Save model and vectorizer
    print("Saving model and vectorizer...")
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
    
    # Save the preprocessing function separately
    with open('preprocess_function.py', 'w') as f:
        f.write("""import re

def preprocess_text(text):
    \"\"\"Clean and normalize text for sentiment analysis\"\"\"
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and replace with space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
""")
    
    print("Model training complete! Model and vectorizer saved.")

except Exception as e:
    import traceback
    print(f"Error during model training: {str(e)}")
    print(traceback.format_exc())