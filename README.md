# Trader Behavior Insights

A comprehensive analysis examining trading performance data against the Crypto Fear & Greed Index to understand how market sentiment impacts trader behavior and profitability.

![Fear vs Greed Analysis](charts/fear_vs_greed_comparison.png)

## Key Findings

1. **Fear days show HIGHER average PnL ($39,138) compared to Greed days ($16,185)** - Statistically significant (P=0.0033)
2. **Traders change behavior based on sentiment** - Trade frequency 2.4x higher on Fear days; Long/Short bias reverses
3. **3 distinct trader segments identified** with different risk/reward profiles requiring differentiated strategies

---

## 🚀 Quick Start

### Clone this repository

```bash
git clone https://github.com/Sarinah01/Trader-Behavior-Insights.git
cd Trader-Behavior-Insights
```

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy
```

### Run the analysis

```bash
python scripts/trading_analysis.py
```

---

## Project Structure

```
Trader-Behavior-Insights/
├── .gitignore                    # Git ignore rules
├── README.md                     # This file
├── fear_greed_index.csv          # Fear & Greed Index data (2018-2025)
├── historical_data.csv           # Trading history (211K trades)
├── scripts/
│   └── trading_analysis.py       # Main analysis script
├── data/                         # Output data directory (generated)
│   ├── daily_analysis.csv
│   ├── fear_greed_aligned.csv
│   ├── trader_segments.csv
│   └── summary_statistics.csv
└── charts/                       # Visualization outputs (generated)
    ├── performance_by_sentiment.png
    ├── fear_vs_greed_comparison.png
    ├── trader_segments.png
    └── correlation_heatmap.png
```

---

## 📋 Requirements

- **Python 3.8+**
- **Dependencies:**
  - pandas>=1.3.0
  - numpy>=1.20.0
  - matplotlib>=3.4.0
  - seaborn>=0.11.0
  - scipy>=1.7.0

---

## 📊 Analysis Results

### Performance: Fear vs Greed Days

| Metric | Fear Days (Value ≤45) | Greed Days (Value >45) | Difference |
|--------|----------------------|----------------------|------------|
| Number of Days | 109 | 370 | - |
| **Avg Daily PnL** | **$39,138** | **$16,185** | **+$22,953** |
| Total PnL | $4,266,077 | $5,988,410 | - |
| Avg Win Rate | 83.76% | 83.59% | +0.17% |
| Avg Trade Count | 795.3 | 336.6 | +458.7 |

**Statistical Significance:** T-test P-value for PnL difference: **0.0033** ✅

### Trader Segments

1. **High Position Size Traders:** 1.85x more PnL but 8% lower win rate
2. **Frequent Traders:** 3.4x more PnL despite similar win rates
3. **Consistent Winners:** 5.6% higher win rate, lower drawdown risk

---

## 🎯 Strategy Recommendations

### Sentiment-Adjusted Position Sizing

```python
def get_sentiment_multiplier(fear_greed_value):
    if fear_greed_value < 35:  # Fear
        return 0.75
    elif fear_greed_value > 65:  # Greed
        return 1.15
    else:  # Neutral
        return 1.0

position_size = base_size * get_sentiment_multiplier(current_sentiment)
```

### Sentiment-Based Trade Frequency

```python
def get_frequency_multiplier(fear_greed_value):
    if fear_greed_value < 35:  # Fear
        return 1.2   # More active - mean-reversion
    elif fear_greed_value > 65:  # Greed
        return 0.8   # More passive - take profits
    else:  # Neutral
        return 1.0   # Normal operation
```

---

## 📁 Generated Files

After running [`scripts/trading_analysis.py`](scripts/trading_analysis.py), the following files will be generated:

**Data Files:**
- `data/daily_analysis.csv` - Daily aggregated metrics with sentiment
- `data/fear_greed_aligned.csv` - Aligned Fear & Greed data
- `data/trader_segments.csv` - Trader segmentation analysis
- `data/summary_statistics.csv` - Key metrics summary

**Visualizations:**
- `charts/performance_by_sentiment.png` - Performance metrics by sentiment
- `charts/fear_vs_greed_comparison.png` - Fear vs Greed comparison
- `charts/trader_segments.png` - Trader segment visualization
- `charts/correlation_heatmap.png` - Correlation matrix

---

## 📝 License

This analysis is provided for educational and research purposes only. Trading involves substantial risk of loss.

---

## 👤 Author

**Sarina**  
February 2025
