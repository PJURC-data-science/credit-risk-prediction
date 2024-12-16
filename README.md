![banner](https://github.com/PJURC-data-science/credit-risk-prediction/blob/main/media/banner.png)

# Credit Risk Prediction: Loan Default Analysis
[View Notebook](https://github.com/PJURC-data-science/credit-risk-prediction/blob/main/Credit%20Risk%20Prediction.ipynb)

A machine learning analysis to predict loan default risk for Home Credit Group clients. This study implements multiple ML models, handles complex missing data scenarios, and deploys a production model to Google Cloud Services.

## Overview

### Business Question 
How can we predict loan default risk using client history and create a robust risk evaluation service for retail banks?

### Key Findings
- Missing data in 70+ features requires special handling
- External scores provide strong predictive power
- Age, employment, payment history are key predictors
- LightGBM achieves 0.76221 ROC AUC score
- Less than 10% of clients default (imbalanced)

### Impact/Results
- Deployed model to Google Cloud
- Created HTTP-based prediction API
- Developed risk categorization
- Established feature importance
- Implemented ensemble methods

## Data

### Source Information
- Dataset: Home Credit Default Risk
- Source: Kaggle Competition
- Coverage: Multiple related tables
- Class Balance: <10% defaults
- Missing Data: 70+ features affected

### Variables Analyzed
Across multiple tables:
- Application data
- Credit bureau records
- Previous credits
- Credit card history
- Payment history
- Installment data
- Balance information

## Methods

### Analysis Approach
1. Data Engineering
   - Missing data handling
   - Feature creation
   - Domain-based features
   - High cardinality encoding
2. Model Development
   - Multiple model training
   - Optuna optimization
   - Ensemble methods
   - Cross-validation
3. Production Deployment
   - Google Cloud integration
   - API development
   - Risk categorization
   - Probability calculation

### Tools Used
- Python (Data Science)
  - XGBoost: Baseline model
  - LightGBM: Primary model
  - CatBoost: Comparison model
  - Scikit-learn:
    - BalancedRandomForestClassifier
    - VotingClassifier
    - RepeatedStratifiedKFold
    - RobustScaler
  - Optuna: Hyperparameter tuning
  - Feature Engineering:
    - Box Cox transformation
    - Mean encoding
    - One-hot encoding
  - Performance Metrics:
    - ROC AUC: 0.76221
    - Cross-validation scores
    - Ensemble performance
- Google Cloud (Model Deployment)

## Getting Started

### Prerequisites
```python
catboost==1.2.7
category_encoders==2.6.4
ipython==8.12.3
Flask==3.1.0
joblib==1.4.2
lightgbm==4.5.0
matplotlib==3.8.4
numpy==2.2.0
optuna==4.1.0
pandas==2.2.3
phik==0.12.4
psutil==6.1.0
scikit_learn==1.6.0
scipy==1.14.1
seaborn==0.13.2
xgboost==2.1.3
```

### Installation & Usage
```bash
git clone git@github.com:PJURC-data-science/credit-risk-prediction.git
cd credit-risk-prediction
pip install -r requirements.txt
jupyter notebook "Credit Risk Prediction.ipynb"
```

The dataset can be downloaded using [this link](https://storage.googleapis.com/341-home-credit-default/home-credit-default-risk.zip)

The HTTP request can be made through the following query:
```cmd
curl -v POST https://example-endpoint.run.app/predict \
-F "client=@http_tests/client.csv" \
-F "bureau_balance=@http_tests/bureau_balance.csv" \
-F "bureau=@http_tests/bureau.csv" \
-F "previous_application=@http_tests/previous_application.csv" \
-F "credit_card_balance=@http_tests/credit_card_balance.csv" \
-F "installments_payments=@http_tests/installments_payments.csv" \
-F "pos_cash_balance=@http_tests/pos_cash_balance.csv" \
-H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Project Structure
```
credit-risk-prediction/
│   README.md
│   requirements.txt
│   Credit Risk prediction.ipynb
|   utils.py
|   app.py
|   utils_app.py
|   styles.css
|   Dockerfile
|   TunedLightGBM.pkl
└── data/
    └── application_train.csv
    └── application_test.csv
└── dicts/
    └── occupation_risk.pkl
    └── region_risk.pkl
└── exports/
    └── submission_LightGBM_post_tuning.csv
```

## Strategic Recommendations
1. **Data Processing**
   - Handle missing data strategically
   - Create domain-specific features
   - Implement robust encoding

2. **Model Selection**
   - Use LightGBM for production
   - Consider ensemble for robustness
   - Monitor class imbalance

3. **Deployment Strategy**
   - Implement cloud-based API
   - Monitor performance
   - Update risk categories

## Future Improvements
- Expand Optuna trials
- Test complete data scenarios
- Implement additional models
- Update data recency
- Enhance feature engineering
- Expand Cloud features (GUI / .csv upload)
- Address regional differences