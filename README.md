# Credit Card Default Prediction

This project applies machine learning techniques to predict whether a credit card client will default on their payment in the next month using the **UCI Credit Card Default Dataset**.

## 📊 Dataset
- Source: [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
- Records: 30,000 clients from Taiwan (2005)  
- Target variable: `default.payment.next.month` (1 = default, 0 = no default)  
- Features include:  
  - Demographics (age, sex, education, marriage)  
  - Credit limit  
  - Payment history (past 6 months)  
  - Bill amounts and payment amounts (past 6 months)  

## ⚙️ Feature Engineering
Additional features were created to capture payment behavior and credit usage trends:
- `late_payments_6m`: Late payments over 6 months  
- `recent_delinquency_severity`: Recent payment delinquency level  
- `avg_util_6m`, `max_util_6m`: Credit utilization ratios  
- `avg_bill_6m`: Average bill over 6 months  
- `sum_pay_6m`: Total payment in 6 months  
- `pay_ratio_6m`: Payment-to-bill ratio  

Categorical variables (`sex`, `education`, `marriage`) were one-hot encoded.  
Numerical features were standardized for Logistic Regression.  

## 🤖 Models Used
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  

## 📈 Evaluation Metrics
Each model was evaluated on:  
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

The script also generates:  
- **ROC Curves** for model comparison  
- **Confusion Matrices** for error analysis  

## 🏆 Results (example values)
| Model              | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.82     | 0.67      | 0.37   | 0.48     | 0.75    |
| Decision Tree      | 0.73     | 0.39      | 0.41   | 0.40     | 0.65    |
| Random Forest      | 0.82     | 0.69      | 0.35   | 0.46     | 0.77    |

👉 The **Random Forest Classifier** performed best based on ROC-AUC.  

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/credit-default-prediction.git
   cd credit-default-prediction

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Place the dataset file UCI_Credit_Card.csv in the project folder.

4. Run the script:
   ```bash
   python credit_default_prediction.py

## 🔮 Future Improvements

- Hyperparameter tuning with GridSearchCV

- Add advanced models (XGBoost, LightGBM)

- Handle class imbalance (SMOTE, class weights)

- Feature importance and SHAP analysis
