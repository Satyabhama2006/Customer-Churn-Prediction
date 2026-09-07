# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn based on demographic information, service usage, contract details, and billing information.

## 📌 Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave can help companies take proactive retention actions.

In this project, customer data is analyzed using Exploratory Data Analysis (EDA), preprocessed using Scikit-learn, and multiple machine learning models are compared to predict customer churn.

The final XGBoost model is deployed through a Streamlit web application for interactive predictions.

---

## 🎯 Objectives

* Analyze customer characteristics associated with churn.
* Perform data cleaning and exploratory data analysis.
* Preprocess numerical and categorical features.
* Compare multiple machine learning classification models.
* Evaluate models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
* Select the best-performing model.
* Build and deploy an interactive Streamlit application for churn prediction.

---

## 📊 Dataset

The project uses a Telco Customer Churn dataset containing **7,043 customer records** and **21 columns**.

### Features

* Customer demographics
* Partner and dependent information
* Tenure
* Phone and internet services
* Online security and backup services
* Device protection and technical support
* Streaming services
* Contract type
* Paperless billing
* Payment method
* Monthly charges
* Total charges

### Target Variable

`Churn`

* `0` → Customer does not churn
* `1` → Customer churns

---

## 🧹 Data Cleaning

The following preprocessing steps were performed:

* Checked for duplicate records.
* Converted `TotalCharges` from string to numeric format.
* Handled 11 missing values in `TotalCharges`.
* Missing `TotalCharges` values were estimated using:

```text
tenure × MonthlyCharges
```

* Removed `customerID` because it does not provide predictive information.
* Converted the target variable `Churn` into binary values.

---

## 🔎 Exploratory Data Analysis

Several important patterns were identified during EDA.

### Contract Type

Customers with month-to-month contracts showed significantly higher churn.

| Contract       | Churn Rate |
| -------------- | ---------: |
| Month-to-month |     42.71% |
| One year       |     11.27% |
| Two year       |      2.83% |

### Tenure

Average tenure was substantially lower among customers who churned.

| Customer Group | Average Tenure |
| -------------- | -------------: |
| Did not churn  |   37.57 months |
| Churned        |   17.98 months |

### Monthly Charges

Customers who churned had higher average monthly charges.

| Customer Group | Average Monthly Charges |
| -------------- | ----------------------: |
| Did not churn  |                   61.27 |
| Churned        |                   74.44 |

### Payment Method

Electronic check users showed the highest churn rate.

| Payment Method            | Churn Rate |
| ------------------------- | ---------: |
| Electronic check          |     45.29% |
| Mailed check              |     19.11% |
| Bank transfer (automatic) |     16.71% |
| Credit card (automatic)   |     15.24% |

### Internet Service

Fiber optic customers showed a relatively high churn rate.

| Internet Service    | Churn Rate |
| ------------------- | ---------: |
| Fiber optic         |     41.89% |
| DSL                 |     18.96% |
| No internet service |      7.40% |

---

## ⚙️ Data Preprocessing

The dataset was divided into training and testing sets using an **80:20 split** with stratification.

### Numerical Features

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

Numerical features were standardized using `StandardScaler`.

### Categorical Features

Categorical features were transformed using `OneHotEncoder`.

A `ColumnTransformer` was used to apply the appropriate preprocessing to each feature type.

The processed dataset contained **46 features** after one-hot encoding.

---

## 🤖 Models Compared

Three classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Model Performance

| Model               |   Accuracy |  Precision |     Recall | F1-score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | -------: | ---------: |
| Logistic Regression |     80.55% |     65.72% |     55.88% |   60.40% |     84.20% |
| Random Forest       |     76.37% |     54.82% | **62.30%** |   58.32% |     82.42% |
| XGBoost             | **80.62%** | **67.47%** |     52.14% |   58.82% | **84.67%** |

---

## 🏆 Final Model

**XGBoost** was selected as the final model because it achieved:

* Highest Accuracy: **80.62%**
* Highest Precision: **67.47%**
* Highest ROC-AUC: **84.67%**

Random Forest achieved the highest recall (**62.30%**), which can be useful when the primary goal is to identify as many potential churners as possible.

However, considering the overall performance and ROC-AUC, XGBoost was selected for the final application.

### XGBoost Configuration

```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
```

---

## 🎯 Classification Threshold Analysis

The default classification threshold of 0.50 was further evaluated to understand the trade-off between precision and recall.

| Threshold | Precision |    Recall |  F1-score |
| --------- | --------: | --------: | --------: |
| 0.30      |     53.3% | **76.5%** | **62.8%** |
| 0.40      |     58.7% |     65.0% |     61.7% |
| 0.50      |     67.5% |     52.1% |     58.8% |
| 0.60      |     72.4% |     38.0% |     49.8% |
| 0.70      | **80.6%** |     26.7% |     40.2% |

A threshold of **0.30** provides substantially higher recall and the highest F1-score among the evaluated thresholds. This can be useful when the business prioritizes identifying more potential churners.

The classification threshold can therefore be adjusted according to the business objective and the relative cost of false positives versus false negatives.

> **Note:** The deployed Streamlit application currently uses the default 0.50 classification threshold. The threshold analysis is included to demonstrate how model behavior can be adjusted according to business requirements.

---

## 🔍 Feature Importance

The XGBoost model was used to identify the features that contributed most strongly to its predictions.

The most influential encoded features were:

| Rank | Feature                 | Importance |
| ---- | ----------------------- | ---------: |
| 1    | Month-to-month contract |      0.371 |
| 2    | Fiber optic internet    |      0.096 |
| 3    | No online security      |      0.083 |
| 4    | No tech support         |      0.041 |
| 5    | Two-year contract       |      0.033 |
| 6    | DSL internet            |      0.026 |
| 7    | Tenure                  |      0.022 |
| 8    | Electronic check        |      0.019 |

The results indicate that **contract type, internet service, online security, technical support, and customer tenure** are among the important factors influencing the model's churn predictions.

The high importance of month-to-month contracts is also consistent with the EDA, where this customer segment had a much higher observed churn rate than customers with one-year or two-year contracts.

> Note: Feature importance indicates how much a feature contributes to the trained model's predictions. It does not by itself establish a causal relationship between the feature and customer churn.

---

## 📈 XGBoost Confusion Matrix

The model produced the following results on the test set:

```text
                 Predicted
                 No     Yes

Actual No        941     94
Actual Yes       179    195
```

Therefore:

* True Negatives = 941
* False Positives = 94
* False Negatives = 179
* True Positives = 195

---

## 🌐 Streamlit Application

An interactive Streamlit application was developed where users can enter customer details and receive:

* Churn prediction
* Churn probability
* Prediction interpretation

Example output:

```text
Customer is NOT likely to churn

Churn Probability: 21.81%
```

### 🚀 Live Demo

**[Open Customer Churn Prediction App](https://customer-churn-prediction-fcptqgjn6agpdqeuafxt9m.streamlit.app/)**

---

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── churn.csv
│
├── models/
│   ├── preprocessor.pkl
│   └── xgb_churn_model.pkl
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Visualization

* Matplotlib
* Seaborn

### Deployment / Interface

* Streamlit

### Model Serialization

* Joblib

### Development Tools

* Jupyter Notebook
* VS Code
* Git & GitHub

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Satyabhama2006/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 💡 Business Insights

The analysis suggests several customer groups that may require additional retention attention:

* Month-to-month contract customers.
* Customers with shorter tenure.
* Customers with higher monthly charges.
* Electronic check users.
* Fiber optic internet users.
* Customers without online security or technical support services.

Businesses can use these insights to design targeted retention strategies such as contract incentives, improved support services, and personalized offers.

---

## 🔮 Future Improvements

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Precision-Recall curve analysis.
* Cross-validation for more robust model evaluation.
* SHAP-based model explainability.
* Customer-level retention recommendations.
* Cost-sensitive learning based on business requirements.
* Automated monitoring of model performance after deployment.

---

## 👩‍💻 Author

**Satyabhama Kumari**

B.Tech — Computer Science & Engineering
NIT Durgapur

---

## ⭐ Project Highlights

* End-to-end machine learning workflow.
* Real-world customer churn prediction problem.
* Comparison of three classification algorithms.
* XGBoost-based final model.
* Classification threshold analysis.
* Feature importance analysis.
* Interactive Streamlit application.
* Live cloud deployment.
* Data-driven business insights.
