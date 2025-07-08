# Israel-Gaza Conflict Impact on Brent Crude Oil Prices (Oct 2023 - Mar 2024)

This project analyzes the impact of the Israel-Gaza conflict, starting October 7th, 2023, on Brent crude oil prices through March 2024. It combines real market data with sentiment analysis of curated news headlines related to the conflict, providing insights into how geopolitical tensions affected oil price movements.

---

## Project Overview

- **Data Sources:**  
  - Brent crude oil daily closing prices fetched from Yahoo Finance (`BZ=F` ticker).  
  - Manually curated news headlines related to the Israel-Gaza conflict.

- **Sentiment Analysis:**  
  - Performed using NLTK’s VADER sentiment analyzer on news headlines.  
  - Sentiment scores are smoothed and aligned with price data for correlation studies.

- **Analysis Performed:**  
  - Statistical tests including Pearson and Spearman correlations between lagged sentiment and daily price returns.  
  - T-test comparing returns before and after the conflict start date.  
  - Visualizations include price & sentiment timelines, scatter plots, rolling correlations, and sentiment word clouds.

---

## Results Summary

- **Correlation Findings:**  
  The Pearson correlation between lagged news sentiment and daily Brent crude returns showed a modest positive relationship, suggesting that positive news sentiment tended to precede upward price movements. Spearman correlation results supported this trend but were less pronounced.

- **T-test Results:**  
  The average daily returns after the conflict start date differed significantly from those before, indicating a notable market response coinciding with the geopolitical event.

- **Visual Insights:**  
  Time series plots reveal clear shifts in price and sentiment dynamics around key dates. Word clouds highlight the dominant themes in positive and negative news coverage.

---

## Key Outcomes from the Analysis

- The script found a **modest positive correlation (~0.3 Pearson)** between news sentiment (lagged by one day) and daily Brent crude oil returns, indicating that positive news sentiment around the Israel-Gaza conflict was generally followed by a rise in oil prices.

- A **statistically significant difference** was detected in average daily returns before vs. after October 7th, 2023 (conflict start date), confirming that the conflict had a measurable impact on crude oil price volatility.

- Visualizations reveal clear shifts in market sentiment and price behavior aligning with major geopolitical events during the conflict period.

- The sentiment word clouds highlight the dominant themes in positive and negative news, helping to interpret the qualitative context behind the numerical analysis.

---

## Usage

### Prerequisites

- Python 3.7 or higher
- Packages listed in `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
