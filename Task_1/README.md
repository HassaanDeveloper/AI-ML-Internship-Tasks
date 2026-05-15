# Task 1: Exploring and Visualizing a Simple Dataset

## 📌 Objective
Learn how to load, inspect, and visualize a dataset to understand data trends and distributions using the classic **Iris Dataset**.

---

## 🗂️ Dataset
- **Name:** Iris Dataset
- **Source:** Built-in via `seaborn` library
- **Size:** 150 rows × 5 columns
- **Features:** `sepal_length`, `sepal_width`, `petal_length`, `petal_width`, `species`

---

## 🛠️ Libraries Used
| Library | Purpose |
|--------|---------|
| `pandas` | Data loading and inspection |
| `numpy` | Numerical operations |
| `matplotlib` | Base plotting |
| `seaborn` | Advanced visualizations |

---

## 📋 Steps Performed

### 1. Load the Dataset
```python
import seaborn as sns
df = sns.load_dataset('iris')
```

### 2. Inspect the Data
```python
print(df.shape)
print(df.columns.tolist())
df.head()
df.info()
df.describe()
```

### 3. Scatter Plot – Feature Relationships
```python
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
```

### 4. Histograms – Value Distributions
```python
df.hist(figsize=(10, 6), bins=20)
```

### 5. Box Plots – Outlier Detection
```python
sns.boxplot(data=df, x='species', y='sepal_length')
```

---

## 📊 Visualizations
- **Scatter Plot:** Shows relationship between sepal length and width, colored by species
- **Histograms:** Shows distribution of all 4 numeric features
- **Box Plots:** Identifies outliers for each feature across the 3 species

---

## 🔍 Key Observations
- The Iris dataset contains **3 species**: Setosa, Versicolor, and Virginica
- **Setosa** is clearly separable from the other two species based on petal dimensions
- **Petal length and petal width** show the most distinct distributions across species
- Minor outliers are visible in the `sepal_width` feature for the Versicolor species

---

## ✅ Skills Demonstrated
- Data loading and inspection using `pandas`
- Descriptive statistics and data exploration
- Basic plotting and visualization with `seaborn` and `matplotlib`

---

## 👨‍💻 Author
**Muhammad Hassaan**  
AI/ML Engineering Intern — DevelopersHub Corporation  
DHC-488
