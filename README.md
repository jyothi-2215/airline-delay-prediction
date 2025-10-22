# Airline Delay Prediction Project

A comprehensive machine learning project that predicts airline delays using various classification, regression, and clustering techniques. This project analyzes flight data to identify patterns and build predictive models for flight delays.

## 🎯 Project Overview

This project implements a complete machine learning pipeline for airline delay prediction, including:
- **Feature Engineering & Exploratory Data Analysis**
- **Regression Analysis** for flight length prediction
- **Classification Analysis** using multiple algorithms
- **Clustering and Association Rule Mining**

## 📊 Dataset

The project uses the Airlines dataset containing flight information with features such as:
- Flight details (Flight number, Length, Time)
- Airline information
- Airport data (AirportFrom, AirportTo)
- Day of week
- Delay status (target variable)

## 🔧 Implementation Phases

### Phase I: Feature Engineering & EDA
- Data cleaning and preprocessing
- Outlier detection and removal using IQR method
- One-hot encoding for categorical variables
- Standardization of numerical features
- Correlation and covariance analysis
- Dimensionality reduction using PCA and SVD
- Feature importance analysis with Random Forest

### Phase II: Regression Analysis
- Linear regression for flight length prediction
- Backward stepwise regression for feature selection
- Statistical analysis (F-test, t-test)
- Model evaluation with R-squared, AIC, BIC metrics
- Confidence interval analysis

### Phase III: Classification Analysis
- **Decision Trees**: Pre-pruned and post-pruned with cost complexity pruning
- **Logistic Regression**: With hyperparameter tuning
- **K-Nearest Neighbors (KNN)**: With optimal k selection
- **Support Vector Machine (SVM)**: With kernel optimization
- **Naive Bayes**: Gaussian implementation
- **Multilayer Perceptron**: Neural network approach

### Phase IV: Clustering and Association
- **K-Means Clustering**: Elbow method and silhouette analysis
- **DBSCAN**: Density-based clustering
- **Apriori Algorithm**: Association rule mining for flight patterns

## 📈 Key Results

The project compares multiple classification algorithms with comprehensive evaluation metrics:
- Accuracy, Precision, Recall, Specificity
- F1-Score and AUC-ROC curves
- Confusion matrices for each model
- Cross-validation for robust evaluation

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
pip install statsmodels mlxtend prettytable
```

## 🚀 Usage

1. **Update the dataset path** in `CS5805_Jyothi_FTP_Code.py` (line 29):
   ```python
   df = pd.read_csv(r'/path/to/your/Airlines.csv')
   ```

2. **Run the code sequentially** from top to bottom:
   - Phase I: Feature Engineering & EDA
   - Phase II: Regression Analysis  
   - Phase III: Classification Analysis
   - Phase IV: Clustering and Association

## 📁 Project Structure

```
ML_Project/FTP_Jyothi/
├── CS5805_Jyothi_FTP_Code.py    # Main implementation
├── Airlines.csv                  # Dataset
├── CS5805_Jyothi_FTP_Report.pdf # Project report
└── readme.txt                    # Usage instructions
```

## 🔍 Key Features

- **Comprehensive Data Preprocessing**: Handles missing values, outliers, and categorical encoding
- **Multiple ML Algorithms**: Implements 7 different classification algorithms
- **Statistical Analysis**: Includes F-tests, t-tests, and confidence intervals
- **Visualization**: Extensive plotting for data exploration and model evaluation
- **Hyperparameter Tuning**: Grid search optimization for all models
- **Cross-Validation**: Stratified k-fold validation for robust evaluation

## 📊 Model Performance

The project evaluates models using multiple metrics and provides detailed comparison tables showing:
- Training and test accuracy
- Confusion matrices
- ROC curves and AUC scores
- Precision, recall, and F1-scores


