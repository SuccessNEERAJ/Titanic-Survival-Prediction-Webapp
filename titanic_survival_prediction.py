#!/usr/bin/env python3
"""
Titanic Survival Prediction - Complete Machine Learning Pipeline
Author: Generated from Jupyter Notebook
Date: 2025

This script performs a comprehensive analysis of the Titanic dataset including:
- Data loading and exploration
- Outlier detection and handling
- Feature engineering
- Model training and evaluation
- Hyperparameter tuning
- Performance visualization
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, classification_report, confusion_matrix, roc_curve)
import xgboost as xgb

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")


def load_and_explore_data(file_path='train.csv'):
    """
    Load and perform initial exploration of the Titanic dataset
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("1. DATA LOADING AND EXPLORATION")
    print("-" * 50)
    
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Dataset info:")
        print(df.info())
        print(f"\nFirst few rows:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please ensure the Titanic dataset is in the same directory as this script.")
        return None


def detect_outliers_iqr(df, column):
    """
    Detect outliers using IQR method
    
    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Column name to check for outliers
        
    Returns:
        pd.Series: Boolean series indicating outliers
    """
    if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
        return pd.Series(dtype=bool)

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers


def analyze_outliers(df, numerical_cols):
    """
    Analyze outliers in numerical columns
    
    Args:
        df (pd.DataFrame): Input dataframe
        numerical_cols (list): List of numerical column names
    """
    print("\nOUTLIER ANALYSIS")
    print("-" * 30)
    
    for col in numerical_cols:
        if col in df.columns:
            outliers = detect_outliers_iqr(df, col)
            outlier_count = outliers.sum()
            outlier_percentage = (outlier_count / len(df)) * 100
            print(f"{col}: {outlier_count} outliers ({outlier_percentage:.1f}%)")

            if outlier_count > 0:
                print(f"  Outlier values: {df[outliers][col].tolist()}")


def handle_outliers_iqr(data, column, method='cap'):
    """
    Handle outliers using IQR method
    
    Args:
        data (pd.DataFrame): Input dataframe
        column (str): Column name to handle outliers
        method (str): Method to handle outliers ('cap', 'remove', 'transform')
        
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    if column not in data.columns or data[column].dtype not in ['int64', 'float64']:
        return data

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    outliers_count = outliers_mask.sum()

    print(f"{column}: Found {outliers_count} outliers")

    if outliers_count > 0:
        if method == 'cap':
            # Cap outliers at the bounds
            data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
            data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
            print(f"  → Capped {outliers_count} outliers")
        elif method == 'remove':
            # Remove outliers (not recommended for small datasets)
            data = data[~outliers_mask]
            print(f"  → Removed {outliers_count} outliers")
        elif method == 'transform':
            # Log transformation for positive values
            if data[column].min() > 0:
                data[column] = np.log1p(data[column])
                print(f"  → Applied log transformation")

    return data


def visualize_outlier_handling(df_original, df_processed, numerical_cols):
    """
    Visualize before and after outlier handling
    
    Args:
        df_original (pd.DataFrame): Original dataframe
        df_processed (pd.DataFrame): Processed dataframe
        numerical_cols (list): List of numerical columns
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Outlier Handling: Before vs After', fontsize=16, fontweight='bold')

    # Age
    if 'Age' in numerical_cols:
        axes[0, 0].boxplot([df_original['Age'].dropna(), df_processed['Age'].dropna()],
                           labels=['Before', 'After'])
        axes[0, 0].set_title('Age Outliers')
        axes[0, 0].set_ylabel('Age')

    # Fare
    if 'Fare' in numerical_cols:
        axes[0, 1].boxplot([df_original['Fare'].dropna(), df_processed['Fare'].dropna()],
                           labels=['Before', 'After'])
        axes[0, 1].set_title('Fare Outliers')
        axes[0, 1].set_ylabel('Fare')

    # SibSp
    if 'SibSp' in numerical_cols:
        axes[1, 0].boxplot([df_original['SibSp'], df_processed['SibSp']],
                           labels=['Before', 'After'])
        axes[1, 0].set_title('SibSp Outliers')
        axes[1, 0].set_ylabel('SibSp')

    # Parch
    if 'Parch' in numerical_cols:
        axes[1, 1].boxplot([df_original['Parch'], df_processed['Parch']],
                           labels=['Before', 'After'])
        axes[1, 1].set_title('Parch Outliers')
        axes[1, 1].set_ylabel('Parch')

    plt.tight_layout()
    plt.show()


def engineer_features(df):
    """
    Create new features from existing ones
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    print("\n2. FEATURE ENGINEERING")
    print("-" * 50)
    
    df_processed = df.copy()
    
    # Create new features
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)

    # Extract title from Name
    df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
    df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                           'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                           'Jonkheer', 'Dona'], 'Rare')
    df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
    df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
    df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')

    # Age groups
    df_processed['AgeGroup'] = pd.cut(df_processed['Age'],
                                      bins=[0, 12, 18, 35, 60, 100],
                                      labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

    # Fare groups
    df_processed['FareGroup'] = pd.qcut(df_processed['Fare'].fillna(df_processed['Fare'].median()),
                                        q=4, labels=['Low', 'Medium', 'High', 'Very High'])

    # Family size categories
    df_processed['FamilySizeGroup'] = 'Medium'
    df_processed.loc[df_processed['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'
    df_processed.loc[df_processed['FamilySize'] <= 4, 'FamilySizeGroup'] = 'Small'
    df_processed.loc[df_processed['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Large'

    print("New features created:")
    print("- FamilySize: SibSp + Parch + 1")
    print("- IsAlone: Binary indicator for solo travelers")
    print("- Title: Extracted from Name")
    print("- AgeGroup: Age categories")
    print("- FareGroup: Fare quartiles")
    print("- FamilySizeGroup: Family size categories")
    
    return df_processed


def prepare_data(df):
    """
    Prepare data for modeling
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (X, y, preprocessor, feature_names)
    """
    print("\n3. DATA PREPARATION")
    print("-" * 50)
    
    # Select features for modeling
    features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                       'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup', 'FamilySizeGroup']

    # Prepare the dataset
    X = df[features_to_use].copy()
    y = df['Survived'].copy()

    print(f"Features selected: {len(features_to_use)}")
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")

    # Identify numerical and categorical columns
    numerical_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'FamilySizeGroup']

    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor, features_to_use


def split_data(X, y):
    """
    Split data into training and testing sets
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n4. TRAIN-TEST SPLIT")
    print("-" * 50)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    print(f"Training target distribution:\n{y_train.value_counts()}")
    print(f"Test target distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def initialize_models():
    """
    Initialize all models for comparison
    
    Returns:
        dict: Dictionary of models
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    }

    print(f"\n5. MODEL INITIALIZATION")
    print("-" * 50)
    print(f"Initialized {len(models)} models:")
    for model_name in models.keys():
        print(f"- {model_name}")
    
    return models


def evaluate_baseline_models(models, preprocessor, X_train, y_train, X_test, y_test):
    """
    Evaluate baseline models with cross-validation
    
    Args:
        models (dict): Dictionary of models
        preprocessor: Sklearn preprocessor
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple: (baseline_results, baseline_cv_scores)
    """
    print("\n6. BASELINE MODEL EVALUATION")
    print("-" * 50)
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store results
    baseline_results = {}
    baseline_cv_scores = {}

    print("Evaluating baseline models with 5-fold cross-validation...")

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Cross-validation scores
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        baseline_cv_scores[name] = cv_scores

        # Fit on training data
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        baseline_results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std(),
            'Training_Time': time.time() - start_time
        }

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Training Time: {time.time() - start_time:.2f}s")

    # Create baseline results DataFrame
    baseline_df = pd.DataFrame(baseline_results).T
    baseline_df = baseline_df.sort_values('F1-Score', ascending=False)

    print("\nBASELINE MODEL COMPARISON:")
    print("=" * 80)
    print(baseline_df.round(4))
    
    return baseline_results, baseline_cv_scores, baseline_df


def hyperparameter_tuning(baseline_df, models, preprocessor, X_train, y_train, X_test, y_test):
    """
    Perform hyperparameter tuning on top models
    
    Args:
        baseline_df (pd.DataFrame): Baseline results DataFrame
        models (dict): Dictionary of models
        preprocessor: Sklearn preprocessor
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        tuple: (tuned_results, best_models, tuned_df)
    """
    print("\n7. HYPERPARAMETER TUNING")
    print("-" * 50)
    
    # Select top 3 models for hyperparameter tuning
    top_models = baseline_df.head(3).index.tolist()
    print(f"Selected models for hyperparameter tuning: {top_models}")

    # Define parameter grids
    param_grids = {
        'Random Forest': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1, 0.15],
            'classifier__max_depth': [3, 5, 7],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        },
        'Logistic Regression': {
            'classifier__C': [0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        },
        'SVM': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf', 'linear'],
            'classifier__gamma': ['scale', 'auto']
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1, 0.15],
            'classifier__max_depth': [3, 5, 7],
            'classifier__subsample': [0.8, 0.9, 1.0]
        }
    }

    # Hyperparameter tuning results
    tuned_results = {}
    best_models = {}

    for model_name in top_models:
        if model_name in param_grids:
            print(f"\nTuning {model_name}...")
            start_time = time.time()

            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', models[model_name])
            ])

            # GridSearchCV
            grid_search = GridSearchCV(
                pipeline,
                param_grids[model_name],
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)

            # Best model predictions
            y_pred = grid_search.predict(X_test)
            y_pred_proba = grid_search.predict_proba(X_test)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            tuned_results[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'Best_CV_Score': grid_search.best_score_,
                'Training_Time': time.time() - start_time
            }

            best_models[model_name] = grid_search.best_estimator_

            print(f"  Best CV Score: {grid_search.best_score_:.4f}")
            print(f"  Test F1-Score: {f1:.4f}")
            print(f"  Best Parameters: {grid_search.best_params_}")
            print(f"  Tuning Time: {time.time() - start_time:.2f}s")

    # Create comparison DataFrame
    tuned_df = None
    if tuned_results:
        tuned_df = pd.DataFrame(tuned_results).T
        tuned_df = tuned_df.sort_values('F1-Score', ascending=False)

        print("\nTUNED MODEL COMPARISON:")
        print("=" * 60)
        print(tuned_df.round(4))
    
    return tuned_results, best_models, tuned_df


def visualize_performance(baseline_df, baseline_cv_scores):
    """
    Visualize model performance comparison
    
    Args:
        baseline_df (pd.DataFrame): Baseline results DataFrame
        baseline_cv_scores (dict): Cross-validation scores
    """
    print("\n8. PERFORMANCE VISUALIZATION")
    print("-" * 50)
    
    # Visualization of model performance
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    # Baseline model comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    baseline_metrics = baseline_df[metrics].head(5)

    x_pos = np.arange(len(baseline_metrics.index))
    width = 0.2

    for i, metric in enumerate(metrics):
        axes[0, 0].bar(x_pos + i*width, baseline_metrics[metric], width, label=metric)

    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Baseline Model Performance')
    axes[0, 0].set_xticks(x_pos + width * 1.5)
    axes[0, 0].set_xticklabels(baseline_metrics.index, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # F1-Score comparison
    top_5_models = baseline_df.head(5)
    axes[0, 1].barh(range(len(top_5_models)), top_5_models['F1-Score'], color='skyblue')
    axes[0, 1].set_yticks(range(len(top_5_models)))
    axes[0, 1].set_yticklabels(top_5_models.index)
    axes[0, 1].set_xlabel('F1-Score')
    axes[0, 1].set_title('Top 5 Models by F1-Score')
    axes[0, 1].grid(True, alpha=0.3)

    # Cross-validation scores
    cv_means = [baseline_cv_scores[model].mean() for model in baseline_cv_scores.keys()]
    cv_stds = [baseline_cv_scores[model].std() for model in baseline_cv_scores.keys()]
    model_names = list(baseline_cv_scores.keys())

    axes[1, 0].errorbar(range(len(model_names)), cv_means, yerr=cv_stds,
                        fmt='o', capsize=5, capthick=2, elinewidth=2)
    axes[1, 0].set_xticks(range(len(model_names)))
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].set_ylabel('CV Accuracy')
    axes[1, 0].set_title('Cross-Validation Scores with Std Dev')
    axes[1, 0].grid(True, alpha=0.3)

    # Training time comparison
    training_times = baseline_df['Training_Time'].head(5)
    axes[1, 1].bar(range(len(training_times)), training_times.values, color='lightcoral')
    axes[1, 1].set_xticks(range(len(training_times)))
    axes[1, 1].set_xticklabels(training_times.index, rotation=45)
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_title('Model Training Time')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def final_evaluation(tuned_results, tuned_df, best_models, baseline_df, models, preprocessor, X_train, y_train, X_test, y_test):
    """
    Final evaluation of the best model
    
    Args:
        tuned_results (dict): Tuned model results
        tuned_df (pd.DataFrame): Tuned results DataFrame
        best_models (dict): Best models from tuning
        baseline_df (pd.DataFrame): Baseline results DataFrame
        models (dict): Dictionary of models
        preprocessor: Sklearn preprocessor
        X_train, y_train: Training data
        X_test, y_test: Test data
    """
    print("\n9. FINAL MODEL EVALUATION")
    print("-" * 50)
    
    # Select the best model
    if tuned_results and tuned_df is not None:
        best_model_name = tuned_df.index[0]
        best_model = best_models[best_model_name]
        print(f"Best Model: {best_model_name}")
        print(f"Best F1-Score: {tuned_df.loc[best_model_name, 'F1-Score']:.4f}")
    else:
        best_model_name = baseline_df.index[0]
        best_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', models[best_model_name])
        ])
        best_model.fit(X_train, y_train)
        print(f"Best Model: {best_model_name}")
        print(f"Best F1-Score: {baseline_df.loc[best_model_name, 'F1-Score']:.4f}")

    # Detailed evaluation of best model
    y_pred_best = best_model.predict(X_test)
    y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]

    print(f"\nDETAILED EVALUATION - {best_model_name}:")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_best):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_best):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_best):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred_best):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_best):.4f}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_best))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Died', 'Survived'],
                yticklabels=['Died', 'Survived'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_proba_best):.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {best_model_name}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

    return best_model, best_model_name


def main():
    """
    Main function to run the complete Titanic survival prediction pipeline
    """
    print("TITANIC SURVIVAL PREDICTION - COMPLETE PIPELINE")
    print("=" * 60)
    
    # 1. Load and explore data
    df = load_and_explore_data('train.csv')
    if df is None:
        return
    
    # Define numerical columns
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    
    # 2. Analyze outliers
    analyze_outliers(df, numerical_cols)
    
    # 3. Handle outliers
    print("\nOUTLIER HANDLING")
    print("-" * 30)
    df_processed = df.copy()
    
    for col in numerical_cols:
        if col in df_processed.columns:
            df_processed = handle_outliers_iqr(df_processed, col, method='cap')
    
    # 4. Visualize outlier handling
    visualize_outlier_handling(df, df_processed, numerical_cols)
    
    # 5. Feature engineering
    df_processed = engineer_features(df_processed)
    
    # 6. Prepare data for modeling
    X, y, preprocessor, features_to_use = prepare_data(df_processed)
    
    # 7. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 8. Initialize models
    models = initialize_models()
    
    # 9. Evaluate baseline models
    baseline_results, baseline_cv_scores, baseline_df = evaluate_baseline_models(
        models, preprocessor, X_train, y_train, X_test, y_test
    )
    
    # 10. Hyperparameter tuning
    tuned_results, best_models, tuned_df = hyperparameter_tuning(
        baseline_df, models, preprocessor, X_train, y_train, X_test, y_test
    )
    
    # 11. Visualize performance
    visualize_performance(baseline_df, baseline_cv_scores)
    
    # 12. Final evaluation
    best_model, best_model_name = final_evaluation(
        tuned_results, tuned_df, best_models, baseline_df, models, 
        preprocessor, X_train, y_train, X_test, y_test
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Best Model: {best_model_name}")
    print("=" * 60)
    
    return best_model, best_model_name, df_processed


if __name__ == "__main__":
    # Run the complete pipeline
    try:
        best_model, best_model_name, processed_data = main()
        print("\nScript executed successfully!")
        print(f"Best performing model: {best_model_name}")
        print("You can now use the 'best_model' for making predictions on new data.")
    except Exception as e:
        print(f"Error occurred during execution: {str(e)}")
        print("Please check that the 'train.csv' file is in the same directory as this script.")
        print("Also ensure all required libraries are installed:")
        print("pip install pandas numpy matplotlib seaborn scikit-learn xgboost")