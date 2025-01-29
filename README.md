# Autism Prediction System

## Overview
The **Autism Prediction System** is a machine learning-based project designed to predict the likelihood of Autism Spectrum Disorder (ASD) in individuals. The system trains and evaluates different classification models, including **Support Vector Classifier (SVC), XGBoost, and Logistic Regression**, selecting the one with the best accuracy.

## Objective
The main goal of this project is to build a predictive model that classifies individuals as having ASD (1) or not (0) based on behavioral and demographic features. The model is trained on a dataset and tested to compare its accuracy with real data.

## Dataset
The project uses a dataset (`train.csv`) that includes the following features:
- **Demographic Information** (Age, Ethnicity, Country of Residence)
- **Behavioral Test Scores** (Responses to ASD screening questions)
- **Medical History** (Jaundice history, Autism in family, etc.)
- **Target Label (`Class/ASD`)** – 1 for ASD, 0 for No ASD

## Project Workflow

### 1. Data Preprocessing
- Load the dataset using `pandas`
- Handle missing values and replace uncertain responses with "Others"
- Convert categorical data into numerical form using `LabelEncoder`
- Normalize numerical values using `StandardScaler`
- Use `RandomOverSampler` to balance class distribution

### 2. Exploratory Data Analysis (EDA)
- Visualize class distribution using pie charts
- Analyze feature distributions using Seaborn plots
- Plot correlation matrices to understand feature relationships

### 3. Feature Engineering
- Create new features, such as `sum_score` (total ASD test score)
- Categorize age into groups (Toddler, Kid, Teenager, Young, Senior)
- Drop unnecessary features like `ID`, `age_desc`, and `used_app_before`

### 4. Model Training
The system trains three different machine learning models:
- **Support Vector Classifier (SVC)**: Works well for high-dimensional data
- **XGBoost Classifier**: Gradient-boosted decision tree known for high accuracy
- **Logistic Regression**: A baseline model for binary classification

### 5. Model Evaluation
- Evaluate each model using **ROC AUC Score**
- Compare training and validation accuracies
- Select the best-performing model for predictions

## Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - `numpy`, `pandas` (Data Handling)
  - `matplotlib`, `seaborn` (Data Visualization)
  - `scikit-learn` (ML Models, Preprocessing)
  - `xgboost` (XGBoost Algorithm)
  - `imblearn` (Handling Imbalanced Data)

## Installation & Execution
### 1. Install Dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

### 2. Check Backend.ipynb
Jupyter Notebook that has all backend integrations and code, showcasing comparisions of SVC, XGBOOST and LOGISTIC REGRESSION.

```

## Expected Outcome
The system predicts whether an individual is likely to have ASD based on their responses. The best model is selected based on accuracy, improving prediction performance and supporting early ASD detection.

## License
This project is open-source and available for public use.

---

Let me know if you need further improvements or customizations! 🚀

