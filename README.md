# Alzheimer’s Disease Prediction Model

## Overview
This project aims to predict whether a patient has Alzheimer's disease or not using various features like age, gender, BMI, cholesterol levels, and other health-related attributes. A Gradient Boosting Classifier is used for model training, and the project involves data preprocessing, feature selection, model training, hyperparameter tuning, and cross-validation.

## Technologies Used
- **Python** 
- **Pandas** for data manipulation
- **NumPy** for numerical operations
- **Scikit-Learn** for machine learning models and evaluation
- **Matplotlib** for visualization (if added in the future)
- **Jupyter Notebook** (for running the code interactively)

## Dataset Information
The dataset used for this project contains 2149 records with 35 attributes, including personal health-related data, such as:
- Age
- Gender
- BMI
- Smoking habits
- Alcohol consumption
- Medical history (Hypertension, Diabetes, etc.)
- Mental state evaluations (e.g., MMSE, Functional Assessment)

The target variable is `Diagnosis`, where `0` represents a healthy patient, and `1` represents a patient diagnosed with Alzheimer's disease.

## Steps Involved

### 1. **Data Preprocessing**
- Imported necessary libraries (`pandas`, `numpy`).
- Loaded the dataset and explored its structure.
- Dropped the `DoctorInCharge` column, which is confidential and irrelevant.
  
### 2. **Feature Selection**
- Used `SelectKBest` with the `f_classif` scoring function to select the top 10 important features for model training.
  
### 3. **Model Training**
- Split the dataset into training and testing sets (80% for training, 20% for testing).
- Trained the model using a **Gradient Boosting Classifier**.

### 4. **Model Evaluation**
- Predicted values for the test set.
- Calculated accuracy, confusion matrix, and classification report for model performance.

### 5. **Hyperparameter Tuning**
- Used **GridSearchCV** to find the best combination of hyperparameters (`n_estimators`, `learning_rate`, `max_depth`).
- Obtained the best model and re-trained it.

### 6. **Cross-Validation**
- Evaluated the model using **K-Fold Cross Validation** and **Stratified K-Fold Cross Validation** to ensure the model's robustness.

### 7. **Final Model Performance**
- Achieved an accuracy of **96.51%** after hyperparameter tuning.
- K-Fold and Stratified K-Fold Cross Validation results also demonstrated high accuracy.

## Results

- **Accuracy**: 96.51%
- **Confusion Matrix**:
  ```
  [[260   7]
   [  8 155]]
  ```
- **Classification Report**:
  ```
              precision    recall  f1-score   support
           0       0.97      0.97      0.97       267
           1       0.96      0.95      0.95       163
    accuracy                           0.97       430
   macro avg       0.96      0.96      0.96       430
weighted avg       0.97      0.97      0.97       430
  ```

- **Cross-Validation Scores**:
  - K-Fold Cross Validation Accuracy: **95.43%**
  - Stratified K-Fold Cross Validation Accuracy: **95.35%**

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/alzheimers-disease-prediction.git
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook to see the code and results:
   ```bash
   jupyter notebook
   ```

## Conclusion
The model successfully predicts whether a patient has Alzheimer’s disease based on various health attributes with high accuracy (96.51%). By using techniques like feature selection, model training, and hyperparameter tuning, we were able to build an effective predictive model for Alzheimer's diagnosis.
