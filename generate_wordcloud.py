import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from pathlib import Path

def clean_text(text):
    """Clean and preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove common stopwords
    stop_words = set(STOPWORDS)
    # Add custom stopwords relevant to course feedback
    custom_stopwords = {'class', 'course', 'professor', 'student', 'lecture', 'would', 
                       'could', 'should', 'the', 'and', 'this', 'that', 'with'}
    stop_words.update(custom_stopwords)
    
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

def generate_wordcloud(feedback_data_path='feedback_data.csv'):
    """Generate wordcloud from feedback data"""
    
    print(f"Generating WordCloud from {feedback_data_path}...")
    
    try:
        # Read the feedback data
        df = pd.read_csv(feedback_data_path)
        print(f"Successfully loaded data with {len(df)} records")
        
        # Ensure directory exists
        Path('static').mkdir(parents=True, exist_ok=True)
        
        # Check if required columns exist
        if 'feedback' not in df.columns or 'sentiment' not in df.columns:
            print("Error: CSV must contain 'feedback' and 'sentiment' columns")
            return False
        
        # Clean the text data
        df['cleaned_feedback'] = df['feedback'].apply(clean_text)
        
        # Generate combined wordcloud for all sentiments
        plt.figure(figsize=(12, 8))
        
        # Define color function based on sentiment
        # For combined wordcloud, use blue color scheme
        all_feedback = ' '.join(df['cleaned_feedback'].dropna())
        
        if all_feedback.strip():  # Check if there's text to process
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=150,
                colormap='viridis',
                contour_width=1, 
                contour_color='steelblue'
            ).generate(all_feedback)
            
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig('static/wordcloud.png', dpi=300, bbox_inches='tight')
            print("Combined wordcloud generated successfully!")
            
            # Optional: Generate separate wordclouds by sentiment
            sentiments = df['sentiment'].unique()
            
            plt.figure(figsize=(16, 6))
            for i, sentiment in enumerate(sentiments):
                plt.subplot(1, len(sentiments), i+1)
                
                # Filter by sentiment
                sentiment_text = ' '.join(df[df['sentiment'] == sentiment]['cleaned_feedback'].dropna())
                
                if sentiment_text.strip():  # Check if there's text to process
                    # Choose color scheme based on sentiment
                    if sentiment in [1, 'positive', 'Positive']:
                        colormap = 'YlGn'  # Green for positive
                    elif sentiment in [-1, 'negative', 'Negative']:
                        colormap = 'OrRd'  # Red for negative
                    else:
                        colormap = 'Blues'  # Blue for neutral
                    
                    sentiment_cloud = WordCloud(
                        width=400, height=300,
                        background_color='white',
                        max_words=100,
                        colormap=colormap,
                    ).generate(sentiment_text)
                    
                    plt.imshow(sentiment_cloud, interpolation='bilinear')
                    plt.axis('off')
                    plt.title(f"{sentiment} Sentiment")
            
            plt.tight_layout()
            plt.savefig('static/sentiment_wordclouds.png', dpi=300, bbox_inches='tight')
            print("Sentiment-specific wordclouds generated successfully!")
            
            return True
        else:
            print("Error: No valid text data found after cleaning")
            return False
            
    except FileNotFoundError:
        print(f"Error: File '{feedback_data_path}' not found")
        return False
    except Exception as e:
        print(f"Error generating wordcloud: {str(e)}")
        return False

if __name__ == "__main__":
    generate_wordcloud()
    print("Done! Wordcloud images saved to 'static' folder.")