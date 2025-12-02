# Telco Customer Churn Analysis

A comprehensive machine learning project analyzing customer churn patterns in a telecommunications dataset and building predictive models to identify at-risk customers.

## 📊 Project Overview

This project analyzes the WA_Fn-UseC_-Telco-Customer-Churn dataset to understand factors contributing to customer churn and develops multiple machine learning models to predict which customers are likely to leave the service.

## 🎯 Objectives

- Perform exploratory data analysis on customer churn patterns
- Identify key features that influence customer churn
- Build and compare multiple classification models
- Optimize model performance for churn prediction
- Provide actionable insights for customer retention strategies

## 📁 Dataset Information

**Dataset:** WA_Fn-UseC_-Telco-Customer-Churn.csv

**Size:** 7,043 customers with 21 features

**Target Variable:** Churn (Yes/No)

**Class Distribution:**
- No Churn: ~73%
- Churn: ~27%

### Features

**Numerical Features:**
- `tenure`: Number of months the customer has stayed with the company
- `MonthlyCharges`: The amount charged to the customer monthly
- `TotalCharges`: The total amount charged to the customer

**Categorical Features:**
- Customer demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Services: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- Account information: `Contract`, `PaperlessBilling`, `PaymentMethod`

## 🔍 Analysis Pipeline

### 1. Data Preprocessing
- Converted `TotalCharges` to numeric format
- Handled missing values (filled with 0)
- Label encoded categorical variables
- Split data into training (80%) and testing (20%) sets with stratification

### 2. Exploratory Data Analysis
- Distribution analysis of numerical features
- Mosaic plots for categorical features vs. churn
- Correlation analysis using KMO and Bartlett's tests

### 3. Dimensionality Reduction
- Applied PCA (Principal Component Analysis)
- Retained 95% of variance
- Reduced features while maintaining model performance
- **KMO Score:** Appropriate for PCA
- **Bartlett's Test:** Variables are significantly correlated

## 🤖 Models Implemented

### 1. Logistic Regression
**Without PCA:**
- ROC-AUC Score: 0.8368
- Accuracy: 0.79
- Precision (Class 1): 0.61
- Recall (Class 1): 0.56

**With PCA:**
- ROC-AUC Score: 0.8263
- Accuracy: 0.72
- Precision (Class 1): 0.49
- Recall (Class 1): 0.80
- Used `class_weight='balanced'` to handle class imbalance
- Max iterations: 1000

### 2. Random Forest Classifier
**Basic Model:**
- ROC-AUC Score: 0.8239
- Accuracy: 0.79
- n_estimators: 50

**Optimized Model (without PCA):**
- n_estimators: 300
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2
- max_features: 'sqrt'
- class_weight: 'balanced'
- ROC-AUC Score: 0.8344
- Accuracy: 0.78
- Precision (Class 1): 0.58
- Recall (Class 1): 0.63

**Optimized Model (with PCA):**
- ROC-AUC Score: 0.8098
- Accuracy: 0.77

### 3. XGBoost Classifier
**GridSearchCV Tuning (without PCA):**
- Performed hyperparameter tuning with 5-fold cross-validation
- Best parameters: learning_rate=0.1, max_depth=3-9, n_estimators=50-250
- **ROC-AUC Score: 0.8451** (Best overall)
- Accuracy: 0.74
- Precision (Class 1): 0.51
- Recall (Class 1): 0.80
- scale_pos_weight: 2.77

**Manual Tuning (without PCA):**
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- gamma: 1
- min_child_weight: 5
- scale_pos_weight: 2.77 (class imbalance ratio)
- ROC-AUC Score: 0.8368
- Accuracy: 0.76

**GridSearchCV Tuning (with PCA):**
- Best parameters: learning_rate=0.1, max_depth=3, n_estimators=50
- ROC-AUC Score: 0.8274
- Accuracy: 0.73
- Best Cross-Validation Score: 0.8302

**Manual Tuning (with PCA):**
- ROC-AUC Score: 0.8197
- Accuracy: 0.76

### 4. Neural Network (Deep Learning)
**Architecture:**
- Input layer: 64 neurons (ReLU activation)
- Dropout: 0.3
- Hidden layer 1: 32 neurons (ReLU activation)
- Dropout: 0.3
- Hidden layer 2: 16 neurons (ReLU activation)
- Output layer: 1 neuron (Sigmoid activation)

**Training Configuration:**
- Optimizer: Adam
- Loss: Binary Crossentropy
- Epochs: 50
- Batch size: 32
- Validation split: 20%

**Performance:**
- **ROC-AUC Score: 0.8326**
- **Accuracy: 0.80** (Highest among all models)
- Precision (Class 1): 0.63
- Recall (Class 1): 0.57
- F1-Score (Class 1): 0.60

## 📈 Key Findings

### Model Performance Summary

| Model | Accuracy | ROC-AUC Score | Notes |
|-------|----------|---------------|-------|
| XGBoost (GridSearch, No PCA) | 0.74 | 0.8451 | **Best ROC-AUC performer** |
| XGBoost (Manual, No PCA) | 0.76 | 0.8368 | Strong balance of metrics |
| Logistic Regression (No PCA) | 0.79 | 0.8368 | Excellent accuracy |
| Random Forest (Optimized, No PCA) | 0.78 | 0.8344 | Well-tuned ensemble |
| Neural Network | 0.80 | 0.8326 | **Highest accuracy overall** |
| XGBoost (GridSearch, PCA) | 0.73 | 0.8274 | Best PCA model |
| Logistic Regression (PCA) | 0.72 | 0.8263 | Good with reduced features |
| Random Forest (Basic, No PCA) | 0.79 | 0.8239 | Baseline ensemble |
| XGBoost (Manual, PCA) | 0.76 | 0.8197 | Decent with PCA |
| Random Forest (Optimized, PCA) | 0.77 | 0.8098 | Lower with PCA |
| Random Forest (Basic, PCA) | 0.76 | 0.7943 | Baseline with PCA |

### Insights

1. **XGBoost with GridSearchCV** achieved the best ROC-AUC score (0.8451), making it ideal for identifying churners
2. **Neural Network** achieved the highest accuracy (0.80), showing deep learning effectiveness on tabular data
3. **PCA** generally reduced performance, indicating original features contain important information for prediction
4. **Class imbalance handling** (via `class_weight='balanced'` and `scale_pos_weight`) was crucial for all models
5. **Trade-off between accuracy and ROC-AUC**: Models optimized for one metric sometimes sacrificed the other
6. **XGBoost and Neural Networks** showed the most balanced performance across both metrics

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **matplotlib** - Data visualization
- **scikit-learn** - Machine learning models and preprocessing
- **xgboost** - Gradient boosting framework
- **tensorflow/keras** - Deep learning framework
- **statsmodels** - Statistical modeling and testing
- **factor_analyzer** - KMO and Bartlett's tests

## 📊 Visualizations Generated

- Churn distribution bar plot
- Numerical feature histograms (tenure, monthly charges, total charges)
- Mosaic plots for categorical features vs. churn

## 🚀 How to Run

1. Install required dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn xgboost tensorflow statsmodels factor-analyzer
```

2. Ensure the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` is in the same directory

3. Run the Jupyter notebook:
```bash
jupyter notebook telco.ipynb
```

## 💡 Business Recommendations

Based on the analysis, businesses should:

1. **Deploy XGBoost (GridSearchCV) for churn prediction** - Best ROC-AUC (0.8451) for identifying at-risk customers
2. **Use Neural Network for high-accuracy scenarios** - Highest accuracy (0.80) for general predictions
3. **Implement early warning systems** combining multiple model predictions for robust detection
4. **Monitor key features** that influence churn:
   - Contract type (month-to-month has highest churn)
   - Tenure (new customers at higher risk)
   - Internet service type (Fiber optic users show higher churn)
   - Monthly charges and total charges
   - Lack of online security and tech support services
5. **Develop targeted retention strategies** for high-risk customer segments identified by models
6. **Prioritize recall over precision** when cost of false negatives (missing churners) is high
7. **Consider customer lifetime value** when prioritizing retention efforts
8. **Focus on customers without dependents or partners** as they show higher churn rates

## 📝 Future Improvements

- Feature engineering to create more predictive variables
- Ensemble methods combining top-performing models
- Time-series analysis for temporal churn patterns
- Cost-sensitive learning to optimize business value
- A/B testing framework for retention strategies
- Real-time churn prediction system deployment


---

*For questions or suggestions, please open an issue in the repository.*
