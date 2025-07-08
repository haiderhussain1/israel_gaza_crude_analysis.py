#!/usr/bin/env python3
"""
Israel-Gaza Conflict Impact on Brent Crude Oil Prices (Oct 7, 2023 - Mar 31, 2024)
--------------------------------------------------------------------------------
This script fetches Brent crude oil prices, analyzes news sentiment on the Israel-Gaza conflict,
and investigates relationships between sentiment and oil price movements.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from scipy.stats import pearsonr, spearmanr, ttest_ind
from wordcloud import WordCloud
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logging.info("Downloading VADER lexicon...")
    nltk.download('vader_lexicon')

def fetch_brent_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches daily Brent crude oil closing prices from Yahoo Finance."""
    logging.info(f"Fetching Brent crude data from {start_date} to {end_date}...")
    df = yf.download("BZ=F", start=start_date, end=end_date, progress=False)
    if df.empty:
        logging.error("Failed to fetch Brent crude data. Please check your internet connection or ticker.")
        sys.exit(1)
    df = df.reset_index()[['Date', 'Close', 'Volume']]
    df.columns = ['date', 'close_price', 'volume']
    logging.info(f"Fetched {len(df)} rows of Brent crude data.")
    return df

def load_news_headlines() -> pd.DataFrame:
    """Returns a DataFrame with curated news headlines and their dates."""
    logging.info("Loading curated news headlines...")
    data = [
        ("2023-10-08", "Oil prices surge on fears of Mideast conflict adding to supply tightness"),
        ("2023-10-13", "What Israel-Hamas war means for global oil market"),
        ("2023-10-23", "Oil drops over 2% as diplomatic moves in Gaza war ease supply concerns"),
        ("2023-11-01", "Oil steadies as supply fears offset demand slowdown"),
        ("2023-12-10", "Oil prices dip on improved Middle East diplomacy"),
        ("2024-01-05", "Brent rises as tanker risks grow amid Red Sea conflict spillover"),
        ("2024-02-01", "Crude slips as Israel-Hamas ceasefire hopes mount"),
        ("2024-03-25", "Oil prices rise as geopolitical risk intensifies supply concerns")
    ]
    df = pd.DataFrame(data, columns=["date", "headline"])
    df['date'] = pd.to_datetime(df['date'])
    logging.info(f"Loaded {len(df)} news headlines.")
    return df

def analyze_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Analyzes sentiment of news headlines using VADER and returns DataFrame with sentiment scores."""
    logging.info("Analyzing sentiment of news headlines with VADER...")
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['headline'].apply(lambda x: sia.polarity_scores(x)['compound'])
    return df

def merge_and_process(oil_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges oil price and news sentiment data by date.
    Interpolates missing sentiment values and calculates derived columns.
    """
    logging.info("Merging oil prices with news sentiment...")
    df = pd.merge(oil_df, news_df[['date', 'sentiment']], on='date', how='left')
    
    missing_sentiment_count = df['sentiment'].isnull().sum()
    logging.info(f"{missing_sentiment_count} days with missing sentiment; interpolating values...")
    df['sentiment'] = df['sentiment'].interpolate(method='linear', limit_direction='both')

    # Smooth sentiment with rolling mean (3 days)
    df['sentiment_smooth'] = df['sentiment'].rolling(window=3, min_periods=1).mean()

    # Daily returns (close price percentage change)
    df['price_return'] = df['close_price'].pct_change()

    # Lag sentiment by 1 day to analyze impact on next day's price
    df['sentiment_lag1'] = df['sentiment_smooth'].shift(1)

    # Drop rows with missing returns or lagged sentiment
    df = df.dropna(subset=['price_return', 'sentiment_lag1'])
    logging.info(f"Data prepared with {len(df)} valid rows after processing.")
    return df

def statistical_analysis(df: pd.DataFrame, conflict_start: datetime):
    """Performs correlation and t-test analyses, then logs and prints the results."""
    logging.info("Performing statistical tests...")

    pearson_corr, pearson_p = pearsonr(df['sentiment_lag1'], df['price_return'])
    spearman_corr, spearman_p = spearmanr(df['sentiment_lag1'], df['price_return'])

    before_returns = df[df['date'] < conflict_start]['price_return']
    after_returns = df[df['date'] >= conflict_start]['price_return']

    t_stat, t_p = ttest_ind(before_returns, after_returns, equal_var=False)

    logging.info(f"Pearson correlation (lagged sentiment vs returns): {pearson_corr:.4f} (p={pearson_p:.4f})")
    logging.info(f"Spearman correlation (lagged sentiment vs returns): {spearman_corr:.4f} (p={spearman_p:.4f})")
    logging.info(f"T-test comparing returns before/after conflict: t={t_stat:.4f}, p={t_p:.4f}")

    print("\n=== Statistical Test Results ===")
    print(f"Pearson correlation (lagged sentiment vs returns): {pearson_corr:.4f} (p={pearson_p:.4f})")
    print(f"Spearman correlation (lagged sentiment vs returns): {spearman_corr:.4f} (p={spearman_p:.4f})")
    print(f"T-test comparing returns before/after conflict: t={t_stat:.4f}, p={t_p:.4f}")

def plot_results(df: pd.DataFrame, conflict_start: datetime, news_df: pd.DataFrame):
    """Generates comprehensive plots showing price, sentiment, correlations, and word clouds."""
    logging.info("Generating plots...")

    plt.style.use('seaborn-darkgrid')
    fig, axs = plt.subplots(3, 1, figsize=(15, 18), sharex=True)

    # Plot 1: Brent crude price + sentiment over time
    axs[0].plot(df['date'], df['close_price'], label='Brent Crude Price (USD)', color='orange')
    axs[0].axvline(conflict_start, color='red', linestyle='--', label='Conflict Start')
    axs[0].set_ylabel('Price (USD)')
    axs[0].set_title('Brent Crude Oil Price & Smoothed News Sentiment')
    axs[0].legend(loc='upper left')

    ax2 = axs[0].twinx()
    ax2.plot(df['date'], df['sentiment_smooth'], label='Smoothed Sentiment', color='blue', linestyle='--')
    ax2.set_ylabel('Sentiment Score')
    ax2.legend(loc='upper right')

    # Plot 2: Scatter plot - lagged sentiment vs daily returns
    sns.scatterplot(data=df, x='sentiment_lag1', y='price_return', ax=axs[1])
    axs[1].set_title('Scatter Plot: Lagged Sentiment vs Daily Price Returns')
    axs[1].set_xlabel('Sentiment (lagged by 1 day)')
    axs[1].set_ylabel('Daily Return')

    # Plot 3: Rolling 30-day correlation between lagged sentiment and returns
    rolling_corr = df['price_return'].rolling(window=30).corr(df['sentiment_lag1'])
    sns.lineplot(data=df, x='date', y=rolling_corr, ax=axs[2], color='purple')
    axs[2].axhline(0, color='grey', linestyle='--')
    axs[2].set_title('Rolling 30-day Pearson Correlation: Price Return vs Lagged Sentiment')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('Correlation')

    plt.tight_layout()
    plt.show()

    # Word clouds for positive and negative sentiment headlines
    positive_text = ' '.join(news_df[news_df['sentiment'] > 0]['headline'])
    negative_text = ' '.join(news_df[news_df['sentiment'] < 0]['headline'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    wc_pos = WordCloud(width=600, height=400, background_color='white', colormap='Greens').generate(positive_text)
    wc_neg = WordCloud(width=600, height=400, background_color='white', colormap='Reds').generate(negative_text)

    ax1.imshow(wc_pos, interpolation='bilinear')
    ax1.set_title('Positive Sentiment Headlines Word Cloud')
    ax1.axis('off')

    ax2.imshow(wc_neg, interpolation='bilinear')
    ax2.set_title('Negative Sentiment Headlines Word Cloud')
    ax2.axis('off')

    plt.show()

def save_data(df: pd.DataFrame, filename='brent_sentiment_enhanced.csv'):
    """Saves the final merged dataset to a CSV file."""
    df.to_csv(filename, index=False)
    logging.info(f"Enhanced dataset saved as '{filename}'.")

def main():
    conflict_start = datetime.strptime("2023-10-07", "%Y-%m-%d")
    end_date = "2024-03-31"

    oil_data = fetch_brent_data(start_date=conflict_start.strftime('%Y-%m-%d'), end_date=end_date)
    news_data = load_news_headlines()
    news_data = analyze_sentiment(news_data)
    merged_df = merge_and_process(oil_data, news_data)
    statistical_analysis(merged_df, conflict_start)
    plot_results(merged_df, conflict_start, news_data)
    save_data(merged_df)

if __name__ == "__main__":
    main()
