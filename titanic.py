import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import streamlit as st

# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Preprocessing
categorical_features = ['Sex', 'Embarked']
numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']

ohe = OneHotEncoder(handle_unknown='ignore')
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', imputer), ('scaler', scaler)]), numerical_features),
    ('cat', ohe, categorical_features)
])

# Splitting data
X = data[numerical_features + categorical_features]
y = data['Survived']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize MLflow
mlflow.set_experiment("Titanic_RandomForest")

with mlflow.start_run():
    # Model training
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)
    
    # Validation score
    y_valid_pred = model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid, y_valid_pred)
    
    # Logging parameters & metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("cv_accuracy", mean_cv_score)
    mlflow.log_metric("validation_accuracy", valid_accuracy)
    
    # Log model
    mlflow.sklearn.log_model(model, "random_forest_titanic")
    
print("MLflow tracking completed!")
