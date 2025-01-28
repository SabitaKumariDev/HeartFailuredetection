**Heart Failure Analysis**

**Overview**

This project analyzes clinical records to predict heart failure using machine learning techniques. It leverages feature selection methods like Recursive Feature Elimination with Cross-Validation (RFECV) and hyperparameter tuning (GridSearch) to build an optimized predictive model.

**Features**

1. Dataset Analysis: Comprehensive analysis of clinical records, including feature distribution and correlation.
   
2. Feature Selection: Uses RFECV to identify the most important features for prediction.
   
3. Model Tuning: Applies GridSearchCV to optimize hyperparameters for better model performance.

4. Visualization: Generates plots for feature importance, correlation heatmaps, and performance metrics.
   
5. Machine Learning Models: Implements multiple classifiers, including Random Forest and Logistic Regression, to predict heart failure outcomes.

**Dataset**

**Dataset Details**

Name: Heart Failure Clinical Records Dataset

Source: UCI Machine Learning Repository

File: heart_failure_clinical_records_dataset.csv

Columns:
  
Age: Age of the patient.

Anaemia: Decrease in red blood cells or hemoglobin (1 = Yes, 0 = No).

Creatinine_phosphokinase (CPK): Level of the CPK enzyme in blood (mcg/L).

Diabetes: Whether the patient has diabetes (1 = Yes, 0 = No).

Ejection_fraction: Percentage of blood leaving the heart each contraction (%).

High_blood_pressure: Whether the patient has hypertension (1 = Yes, 0 = No).

Platelets: Platelet count in the blood (kiloplatelets/mL).

Serum_creatinine: Level of creatinine in the blood (mg/dL).

Serum_sodium: Level of sodium in the blood (mEq/L).

Sex: Gender of the patient (1 = Male, 0 = Female).

Smoking: Whether the patient smokes (1 = Yes, 0 = No).

Time: Follow-up period (days).

Death_event: Outcome (1 = Death, 0 = Alive).

**Installation**

Prerequisites

  Python 3.8 or higher.
  
Required Libraries

  Install the dependencies using the provided requirements.txt file:

      pip install -r requirements.txt

**Usage**

**1. Data Preprocessing**

  Load and clean the dataset.
  
  Handle missing values and outliers.
  
  Scale numerical features using standardization.

**2. Feature Selection with RFECV**

  Use Recursive Feature Elimination with Cross-Validation to identify the most impactful features.
  
  Example:

    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier()
    selector = RFECV(model, step=1, cv=5)
    selector.fit(X, y)
    print(selector.support_)

**3. Model Training**

  Train multiple models using the selected features.
  
  Apply GridSearchCV to find the best hyperparameters.
  
  Example:

      from sklearn.model_selection import GridSearchCV
      
      param_grid = {'n_estimators': [50, 100, 200]}
      grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
      grid_search.fit(X_train, y_train)
      print(grid_search.best_params_)

**4. Model Evaluation**

  Evaluate models using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.
  
  Generate classification reports and confusion matrices.
  
**5. Visualization**

  Create plots for:
  
    Feature importance.
    
    Correlation heatmap.
    
    ROC curves for models

**Results**

Performance Metrics

  ![image](https://github.com/user-attachments/assets/d9ffa89c-c3b6-47b8-9780-f5583aa8954d)

Feature Importance

Key features identified using RFECV:

  Serum_creatinine
  
  Ejection_fraction
  
  Age
  
  Time

**Visualization Examples**

Correlation Heatmap

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.show()

Feature Importance Plot

    from matplotlib import pyplot as plt

    importance = model.feature_importances_
    plt.bar(range(len(importance)), importance)
    plt.title('Feature Importance')
    plt.show()

**Challenges and Solutions**

1. Imbalanced Dataset:
   
    Applied SMOTE (Synthetic Minority Oversampling Technique) to balance the classes.
   
2. Feature Redundancy:
   
    Used RFECV to remove irrelevant or redundant features.

3. Hyperparameter Tuning:
   
    Utilized GridSearchCV for optimal parameter selection.

**Future Work**

  Include advanced models like XGBoost and LightGBM for better performance.
  
  Explore deep learning techniques for feature extraction and prediction.
  
  Integrate real-time prediction capabilities using a web interface or API.

**References**

  1. Heart Failure Dataset - UCI Repository
  
  2. Scikit-learn Documentation
     
  3. SMOTE: Synthetic Minority Oversampling Technique
