# Auction Verification

A machine learning project that ensures the integrity of auction results, specifically focused on verifying outcomes from Germany's 4G spectrum auctions.

## 📌 Project Overview

Designed an advanced ML model that verified auction results with 99% reliability, accurately detecting inconsistencies in high-stakes auctions. This system:

- Enhances financial decision-making, mitigating potential miscalculations amounting to billions of dollars.
- Strengthens auction transparency, validating outcomes and instilling trust in spectrum allocation.
- Serves as a robust validation framework, leveraging classification and regression to ensure data integrity and operational reliability.

By training supervised ML models, this project automates the verification of auction legitimacy and predicts the optimal verification runtime, providing both speed and accuracy.

## 📊 Dataset

Sourced from the UCI Machine Learning Repository, the dataset includes:
- 2043 instances
- 9 features
- Two prediction targets:
  - verification.result (True/False)
  - verification.time (in ms)

## 🔧 Tech Stack

- **Language**: Python 3.12
- **IDE**: Visual Studio Code
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn

## 🧠 ML Models Used

### For verification.result (classification):
- K-Nearest Neighbors (KNN)
- ✅ Decision Tree Classifier (Best Performing)

### For verification.time (regression):
- Linear Regression
- ✅ Gradient Boosting Regressor (Best Performing)

## 🧪 Data Preprocessing

- Converted categorical data to numerical
- Performed correlation analysis and dimensionality reduction
- Applied random oversampling to mitigate class imbalance
- Used a 75/25 train-test split

## 📈 Evaluation Metrics

- **Classification**: Accuracy Score (95% for Decision Tree)
- **Regression**: MSE, R² Score (best with Gradient Boosting)


## 📂 Project Structure

```
├── data/                      # Dataset and optional preprocessed data
├── models/                    # Scripts for training and evaluating models
├── plots/                     # Graphs and visualizations
├── Auction_Verification.ipynb # Core notebook
├── README.md                  # Documentation
```

## 🚀 Getting Started

Clone the repository:
```bash
git clone https://github.com/AjayVasan/Auction-Verification.git
cd Auction-Verification
```

Install required packages:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook Auction_Verification.ipynb
```

## 📎 References

- [UCI Dataset](https://doi.org/10.24432/C52K6N)
- [Gradient Boosting – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)
- [Decision Tree – GeeksForGeeks](https://www.geeksforgeeks.org/decision-tree/)
