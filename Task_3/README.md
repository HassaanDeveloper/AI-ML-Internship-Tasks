# Task 3: Heart Disease Prediction

## 📌 Objective
Build a binary classification model to predict whether a person is **at risk of heart disease** based on their health data.

---

## 🗂️ Dataset
- **Name:** Heart Disease UCI Dataset
- **Source:** [Public GitHub CSV](https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv)
- **Size:** 303 rows × 14 columns
- **Target Column:** `target` — `1` = Heart Disease Present, `0` = No Heart Disease

### Feature Descriptions
| Feature | Description |
|---------|-------------|
| `age` | Age of the patient |
| `sex` | Gender (1 = Male, 0 = Female) |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = True) |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = Yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy |
| `thal` | Thalassemia type |
| `target` | Heart disease presence (1 = Yes, 0 = No) |

---

## 🛠️ Libraries Used
| Library | Purpose |
|--------|---------|
| `pandas` | Data loading and cleaning |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | EDA visualizations |
| `sklearn` | Model training and evaluation |

---

## 📋 Steps Performed

### 1. Load the Dataset
```python
url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"
df = pd.read_csv(url)
```

### 2. Data Cleaning
```python
df.isnull().sum()       # Check missing values
df.duplicated().sum()   # Check duplicates
df.describe()           # Statistical summary
```
> ✅ Dataset was found to be clean with no missing values or duplicates.

### 3. Exploratory Data Analysis (EDA)
- **Bar Chart:** Class distribution of target variable
- **Box Plot:** Age vs Heart Disease presence
- **Heatmap:** Feature correlation matrix

### 4. Feature Scaling & Train-Test Split
```python
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 5. Train Models
```python
lr = LogisticRegression(max_iter=1000, random_state=42)
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
```

### 6. Evaluate Models
```python
accuracy_score(y_test, predictions)
classification_report(y_test, predictions)
roc_curve(y_test, probabilities)
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
```

### 7. Feature Importance
```python
importance = pd.Series(dt.feature_importances_, index=X.columns)
importance.sort_values(ascending=False)
```

---

## 📊 Evaluation Metrics
| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted positives, how many are actually positive |
| **Recall** | Of actual positives, how many were correctly identified |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **ROC-AUC** | Area under ROC curve — closer to 1.0 is better |

---

## 📈 Visualizations
- **Confusion Matrix** for both Logistic Regression and Decision Tree
- **ROC Curve** comparing AUC scores of both models
- **Feature Importance Bar Chart** showing top predictors

---

## 🔍 Key Observations
- **`cp` (chest pain type)**, **`thal`**, and **`ca`** were the most important features — consistent with medical research
- **Logistic Regression** achieved strong performance with high AUC, making it a reliable choice for medical classification
- In medical AI, **Recall (Sensitivity)** for the positive class matters more than raw accuracy — a false negative (missing a disease) is far more dangerous than a false positive
- The dataset is relatively balanced, reducing bias concerns in model training

---

## ✅ Skills Demonstrated
- Binary classification with real medical data
- Data cleaning and exploratory data analysis
- Model evaluation using accuracy, ROC-AUC, and confusion matrix
- Feature importance analysis and medical insight interpretation

---

## 👨‍💻 Author
**Muhammad Hassaan**  
AI/ML Engineering Intern — DevelopersHub Corporation  
DHC-488
