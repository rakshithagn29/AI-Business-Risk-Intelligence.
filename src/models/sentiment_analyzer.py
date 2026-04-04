import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import warnings
warnings.filterwarnings('ignore')

# Sample customer feedback data
SAMPLE_FEEDBACK = [
    "The service is excellent and very reliable",
    "Very disappointed with the billing issues",
    "Good network coverage but expensive",
    "Worst customer service I have ever experienced",
    "Happy with the service overall",
    "Internet speed is very slow and unreliable",
    "Great value for money and good support",
    "Constantly facing network outages",
    "Satisfied with the current plan",
    "The charges are too high for poor service",
    "Amazing service highly recommend",
    "Very bad experience with customer care",
    "Decent service nothing special",
    "Excellent support team very helpful",
    "Thinking of switching to another provider",
    "Signal drops frequently very frustrating",
    "Best telecom service in the area",
    "Bill was incorrect multiple times",
    "Happy customer for 5 years",
    "Planning to cancel subscription soon"
]

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity

    if polarity > 0.1:
        return 'Positive', polarity
    elif polarity < -0.1:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    try:
        sid = SentimentIntensityAnalyzer()
        scores = sid.polarity_scores(str(text))
        compound = scores['compound']

        if compound >= 0.05:
            return 'Positive', compound
        elif compound <= -0.05:
            return 'Negative', compound
        else:
            return 'Neutral', compound
    except:
        return analyze_sentiment_textblob(text)

def create_sentiment_dataset(df=None):
    """Create sentiment analysis dataset"""
    
    # Use sample feedback
    feedback_data = pd.DataFrame({
        'customer_id': range(len(SAMPLE_FEEDBACK)),
        'feedback': SAMPLE_FEEDBACK
    })
    
    # Analyze each feedback
    results = []
    for _, row in feedback_data.iterrows():
        sentiment, score = analyze_sentiment_vader(row['feedback'])
        
        # Convert to risk score
        if sentiment == 'Negative':
            sentiment_risk = abs(score) * 100
        elif sentiment == 'Positive':
            sentiment_risk = 0
        else:
            sentiment_risk = 25
            
        results.append({
            'customer_id': row['customer_id'],
            'feedback': row['feedback'],
            'sentiment': sentiment,
            'sentiment_score': round(score, 3),
            'sentiment_risk': round(sentiment_risk, 1)
        })
    
    return pd.DataFrame(results)

def get_sentiment_summary(sentiment_df):
    """Get summary statistics"""
    total = len(sentiment_df)
    positive = len(sentiment_df[sentiment_df['sentiment'] == 'Positive'])
    negative = len(sentiment_df[sentiment_df['sentiment'] == 'Negative'])
    neutral = len(sentiment_df[sentiment_df['sentiment'] == 'Neutral'])
    
    return {
        'total': total,
        'positive': positive,
        'negative': negative,
        'neutral': neutral,
        'positive_pct': round(positive/total*100, 1),
        'negative_pct': round(negative/total*100, 1),
        'neutral_pct': round(neutral/total*100, 1),
        'avg_risk': round(sentiment_df['sentiment_risk'].mean(), 1)
    }