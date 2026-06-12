# src/real_sentiment_collector.py
# REAL-TIME data collector from NewsAPI
# Genuine live news data about Indian Telecom

import requests
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))


def collect_live_news_data():
    """
    Collect REAL live news articles about
    Indian telecom companies using NewsAPI
    This is genuine real-time live data!
    """
    print("📡 Connecting to NewsAPI...")

    api_key = os.getenv('NEWS_API_KEY')
    if not api_key:
        print("⚠️ NEWS_API_KEY not found in .env")
        return pd.DataFrame()

    queries = ['Jio customer', 'Airtel network',
               'Vi Vodafone Idea', 'BSNL service',
               'telecom complaint India',
               'telecom churn India']

    all_articles = []

    for query in queries:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q':        query,
                'language': 'en',
                'sortBy':   'publishedAt',
                'pageSize': 20,
                'apiKey':   api_key
            }

            print(f"  📥 Searching: '{query}'...")
            response = requests.get(url, params=params,
                                    timeout=10)
            data = response.json()

            if data.get('status') == 'ok':
                for article in data.get('articles', []):
                    all_articles.append({
                        'source':       'NewsAPI',
                        'query':        query,
                        'publisher':    article['source']['name'],
                        'title':        article['title'] or '',
                        'description':  article['description'] or '',
                        'text':         (article['title'] or '') +
                                       " " +
                                       (article['description'] or ''),
                        'published_at': article['publishedAt'],
                        'url':          article['url']
                    })
            else:
                print(f"    ⚠️ {data.get('message', 'Unknown error')}")

        except Exception as e:
            print(f"  ⚠️ Error for '{query}': {e}")
            continue

    df = pd.DataFrame(all_articles)

    if len(df) > 0:
        df = df.drop_duplicates(subset='title')

    print(f"\n✅ Collected {len(df)} REAL live news articles!")
    return df


def analyze_real_sentiment(df, text_col='text'):
    """Run VADER sentiment on real collected data"""
    if len(df) == 0:
        return df

    try:
        sid = SentimentIntensityAnalyzer()
    except:
        nltk.download('vader_lexicon', quiet=True)
        sid = SentimentIntensityAnalyzer()

    results = []
    for _, row in df.iterrows():
        scores = sid.polarity_scores(str(row[text_col]))
        compound = scores['compound']

        if compound >= 0.05:
            sentiment = 'Positive'
            risk = 0
        elif compound <= -0.05:
            sentiment = 'Negative'
            risk = abs(compound) * 100
        else:
            sentiment = 'Neutral'
            risk = 25

        row_dict = row.to_dict()
        row_dict.update({
            'sentiment':       sentiment,
            'sentiment_score': round(compound, 3),
            'sentiment_risk':  round(risk, 1)
        })
        results.append(row_dict)

    return pd.DataFrame(results)


def detect_churn_signals(df):
    """Detect churn-related keywords in real news"""
    if len(df) == 0:
        return df, 0

    churn_keywords = ['switch', 'switching', 'cancel',
                      'port', 'leaving', 'complaint',
                      'outage', 'down', 'issue',
                      'problem', 'disconnect']

    df['churn_signal'] = df['text'].str.lower().apply(
        lambda x: any(kw in x for kw in churn_keywords)
    )

    signal_rate = df['churn_signal'].mean() * 100

    print(f"\n🎯 REAL-TIME CHURN SIGNAL DETECTION")
    print(f"="*45)
    print(f"  Articles analyzed   : {len(df)}")
    print(f"  Churn signals found : {df['churn_signal'].sum()}")
    print(f"  Churn signal rate   : {signal_rate:.1f}%")

    return df, signal_rate


def collect_and_save():
    """Main function - collect, analyze, save"""
    print("🚀 COLLECTING REAL-TIME TELECOM NEWS DATA")
    print("="*50)

    df = collect_live_news_data()

    if len(df) == 0:
        print("⚠️ No data collected — check API key")
        return None, None

    df = analyze_real_sentiment(df)
    df, signal_rate = detect_churn_signals(df)

    save_path = os.path.join(
        BASE_DIR, "data", "external")
    os.makedirs(save_path, exist_ok=True)

    df.to_csv(
        os.path.join(save_path, "live_news_data.csv"),
        index=False)

    summary = {
        'collection_time':   datetime.now().strftime(
            '%Y-%m-%d %H:%M:%S'),
        'total_articles':    len(df),
        'positive':          int(len(df[df['sentiment']=='Positive'])),
        'negative':          int(len(df[df['sentiment']=='Negative'])),
        'neutral':           int(len(df[df['sentiment']=='Neutral'])),
        'churn_signal_rate': round(signal_rate, 1),
        'data_source':       'NewsAPI (live)',
        'queries':           'Jio, Airtel, Vi, BSNL, telecom complaint/churn India'
    }

    with open(os.path.join(save_path,
              "live_data_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Saved to data/external/live_news_data.csv")
    print(f"✅ This is REAL-TIME LIVE DATA from NewsAPI")

    return df, summary