# 📬 Spam Detector

This project implements a **spam classification system** using two machine learning models:

- **Logistic Regression**
- **Random Forest Classifier**

The dataset comes from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/94/spambase), and the goal is to classify emails as spam or not spam based on extracted features.

---

## 📥 Data Source

- **CSV file**: [spam-data.csv](https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv)
- **Source**: UCI ML Repository

The dataset includes **4,601 entries**, and all values are numeric with no missing data.

---

## 📊 Project Workflow

### 1. Data Import
```python
import pandas as pd
data = pd.read_csv("https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv")
```

### 2. Split Data
```python
from sklearn.model_selection import train_test_split
X = data.drop(columns=['spam'])
y = data['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
```

### 3. Scale Features
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 4. Train Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state=1).fit(X_train_scaled, y_train)
```

### 5. Train Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=1).fit(X_train_scaled, y_train)
```

### 6. Evaluate Accuracy
```python
from sklearn.metrics import accuracy_score
accuracy_lr = accuracy_score(y_test, lr_model.predict(X_test_scaled))
accuracy_rf = accuracy_score(y_test, rf_model.predict(X_test_scaled))
```

---

## 🧠 Prediction vs Reality

**Initial Prediction:**  
Random Forest would outperform Logistic Regression based on UCI’s published benchmarks.

**Result:**
- Logistic Regression Accuracy: ~92%
- Random Forest Accuracy: ~96%

While Random Forest showed better performance, there's a mild concern about potential overfitting, which is common for tree-based models.

---

## 🤖 Conclusion

- Both models are viable for spam detection.
- Random Forest offers higher accuracy but may require cross-validation or tuning to manage overfitting.

---

## 👤 Author

Josh Richardson  
[LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/jrich71)

---

## 📄 License

MIT License – Educational use encouraged.
