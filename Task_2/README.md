# Task 2: Predict Future Stock Prices (Short-Term)

## 📌 Objective
Use historical stock market data to predict the **next day's closing price** using regression models.

---

## 🗂️ Dataset
- **Source:** Yahoo Finance via `yfinance` Python library
- **Stock Used:** Apple Inc. (`AAPL`)
- **Date Range:** January 2023 – January 2025
- **Features:** `Open`, `High`, `Low`, `Volume`
- **Target:** Next day's `Close` price (shifted by 1)

---

## 🛠️ Libraries Used
| Library | Purpose |
|--------|---------|
| `yfinance` | Fetching live stock data from Yahoo Finance |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting predictions |
| `sklearn` | Model training and evaluation |

---

## 📋 Steps Performed

### 1. Install & Import
```python
!pip install yfinance --quiet
import yfinance as yf
```

### 2. Fetch Stock Data
```python
df = yf.download('AAPL', start='2023-01-01', end='2025-01-01')
```

### 3. Feature Engineering
```python
df['Target'] = df['Close'].shift(-1)  # Next day's closing price
features = ['Open', 'High', 'Low', 'Volume']
```

### 4. Train-Test Split (No Shuffle)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # shuffle=False is critical for time series
)
```

### 5. Train Models
```python
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
```

### 6. Evaluate & Plot
```python
mean_absolute_error(y_test, predictions)
r2_score(y_test, predictions)
```

---

## 📊 Evaluation Metrics
| Metric | Description |
|--------|-------------|
| **MAE** (Mean Absolute Error) | Average dollar difference between predicted and actual price |
| **R² Score** | How well the model explains price movement (closer to 1.0 = better) |

---

## 📈 Visualizations
- **Actual vs Predicted Line Chart:** Compares real closing prices with predictions from both models over the test period

---

## 🔍 Key Observations
- Both models perform well because stock prices don't change drastically day-to-day — known as the **persistence effect**
- **Linear Regression** is simple but surprisingly effective for next-day prediction
- **Random Forest** captures non-linear patterns and tends to generalize better
- `shuffle=False` is essential in time series to avoid future data leaking into training

---

## ✅ Skills Demonstrated
- Time series data handling
- Regression modeling (Linear Regression & Random Forest)
- Data fetching using external APIs (`yfinance`)
- Plotting and comparing predictions vs real data

---

## 👨‍💻 Author
**Muhammad Hassaan**  
AI/ML Engineering Intern — DevelopersHub Corporation  
DHC-488
