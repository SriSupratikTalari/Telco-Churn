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
- ROC-AUC Score: ~0.8484

**With PCA:**
- ROC-AUC Score: ~0.8406
- Used `class_weight='balanced'` to handle class imbalance

### 2. Random Forest Classifier
**Basic Model:**
- ROC-AUC Score: ~0.8250

**Optimized Model (without PCA):**
- n_estimators: 300
- max_depth: 20
- min_samples_split: 5
- min_samples_leaf: 2
- max_features: 'sqrt'
- class_weight: 'balanced'
- ROC-AUC Score: ~0.8238

**With PCA:**
- ROC-AUC Score: ~0.8090

### 3. XGBoost Classifier
**Manual Tuning (without PCA):**
- n_estimators: 300
- max_depth: 6
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- gamma: 1
- min_child_weight: 5
- scale_pos_weight: ~2.87 (class imbalance ratio)
- ROC-AUC Score: ~0.8448

**GridSearchCV Tuning (with PCA):**
- Performed hyperparameter tuning with 5-fold cross-validation
- Best parameters identified through grid search
- ROC-AUC Score: ~0.8376

**Manual Tuning (with PCA):**
- ROC-AUC Score: ~0.8363

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
- ROC-AUC Score: ~0.8433

## 📈 Key Findings

### Model Performance Summary

| Model | ROC-AUC Score | Notes |
|-------|--------------|-------|
| Logistic Regression (No PCA) | 0.8484 | **Best overall performer** |
| XGBoost (Manual, No PCA) | 0.8448 | Strong performance with tuning |
| Neural Network | 0.8433 | Good deep learning baseline |
| Logistic Regression (PCA) | 0.8406 | Slight drop with PCA |
| XGBoost (GridSearch, PCA) | 0.8376 | Well-tuned with PCA |
| XGBoost (Manual, PCA) | 0.8363 | Good PCA performance |
| Random Forest (Basic) | 0.8250 | Baseline ensemble model |
| Random Forest (Optimized) | 0.8238 | Minimal improvement with tuning |
| Random Forest (PCA) | 0.8090 | Lower performance with PCA |

### Insights

1. **Logistic Regression** performed best overall, suggesting linear relationships are strong in the data
2. **PCA** slightly reduced performance for most models, indicating that original features are valuable
3. **Class imbalance handling** (via `class_weight='balanced'` and `scale_pos_weight`) was crucial for all models
4. **XGBoost** showed competitive performance with proper hyperparameter tuning
5. **Neural Networks** achieved solid results, comparable to gradient boosting methods

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

1. **Focus on high-risk customer segments** identified by the models
2. **Implement early warning systems** using the Logistic Regression model (best ROC-AUC)
3. **Monitor key features** that influence churn (tenure, contract type, monthly charges)
4. **Develop targeted retention strategies** for customers predicted to churn
5. **Consider customer lifetime value** when prioritizing retention efforts

## 📝 Future Improvements

- Feature engineering to create more predictive variables
- Ensemble methods combining top-performing models
- Time-series analysis for temporal churn patterns
- Cost-sensitive learning to optimize business value
- A/B testing framework for retention strategies
- Real-time churn prediction system deployment

## 📄 License

This project is available for educational and research purposes.

## 👤 Author

Data Science Project - Telco Customer Churn Analysis

---

*For questions or suggestions, please open an issue in the repository.*
