# Email Classification with Machine Learning

## Overview
This project applies machine learning techniques to classify emails as either spam or legitimate. The goal is to build an accurate and reliable model that can automate email filtering and reduce manual effort in handling spam.

## Dataset
The dataset used for this project consists of labeled email samples, distinguishing between spam and non-spam messages. The data was split into training and testing sets to evaluate model performance effectively. To view details about the dataset and to access the data itself, check out [UCI Machine Learning Library](https://archive.ics.uci.edu/dataset/94/spambase).

## Models Used
Two machine learning models were trained and evaluated:

1. **Logistic Regression**
2. **Random Forest Classifier**

## Dependencies

```import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
```

## Results
### Logistic Regression
- **Accuracy:** ~92%
- **Conclusion:** The model performed well, correctly categorizing approximately 92% of the emails in the test set. This indicates that logistic regression is a strong baseline model for this classification task.

### Random Forest
- **Accuracy:** ~96%
- **Conclusion:** The model achieved very high accuracy (~96%), significantly outperforming logistic regression. However, there are concerns about potential overfitting, which is a common characteristic of Random Forest models. Further tuning or alternative methods like pruning or feature selection may be required to ensure robustness.

## Author
- Josh Richardson
- [GitHub Profile](https://github.com/jrich71)
- [LinkedIn Profile](https://www.linkedin.com/in/jrichardson7/)