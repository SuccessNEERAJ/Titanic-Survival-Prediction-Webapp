#!/usr/bin/env python3
"""
Titanic Survival Prediction - Streamlit Web Application
Author: Generated for Interactive ML Analysis
Date: 2025

This Streamlit app provides an interactive interface for:
- Uploading custom test datasets
- Data preprocessing and outlier handling
- Feature engineering
- Model training and evaluation
- Bulk survival predictions
- Performance visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
import time
import io

# Set page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Set random seed for reproducibility
np.random.seed(42)

@st.cache_data
def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    if column not in df.columns or df[column].dtype not in ['int64', 'float64']:
        return pd.Series(dtype=bool)

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers

def handle_outliers_iqr(data, column, method='cap'):
    """Handle outliers using IQR method"""
    if column not in data.columns or data[column].dtype not in ['int64', 'float64']:
        return data, 0

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
    outliers_count = outliers_mask.sum()

    if outliers_count > 0:
        if method == 'cap':
            data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
            data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
        elif method == 'remove':
            data = data[~outliers_mask]
        elif method == 'transform':
            if data[column].min() > 0:
                data[column] = np.log1p(data[column])

    return data, outliers_count

def engineer_features(df):
    """Create new features from existing ones"""
    df_processed = df.copy()

    # Create new features
    df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1
    df_processed['IsAlone'] = (df_processed['FamilySize'] == 1).astype(int)

    # Extract title from Name
    if 'Name' in df_processed.columns:
        df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)
        df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess','Capt', 'Col',
                                                               'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                               'Jonkheer', 'Dona'], 'Rare')
        df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
        df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
        df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')
    else:
        df_processed['Title'] = 'Mr'  # Default title if Name column is missing

    # Age groups
    if 'Age' in df_processed.columns:
        df_processed['AgeGroup'] = pd.cut(df_processed['Age'],
                                          bins=[0, 12, 18, 35, 60, 100],
                                          labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
    else:
        df_processed['AgeGroup'] = 'Adult'  # Default age group

    # Fare groups
    if 'Fare' in df_processed.columns:
        df_processed['FareGroup'] = pd.qcut(df_processed['Fare'].fillna(df_processed['Fare'].median()),
                                            q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    else:
        df_processed['FareGroup'] = 'Medium'  # Default fare group

    # Family size categories
    df_processed['FamilySizeGroup'] = 'Medium'
    df_processed.loc[df_processed['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'
    df_processed.loc[df_processed['FamilySize'] <= 4, 'FamilySizeGroup'] = 'Small'
    df_processed.loc[df_processed['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Large'

    return df_processed

@st.cache_data
def prepare_data_for_modeling(df):
    """Prepare data for modeling"""
    # Select features for modeling
    features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                       'FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup', 'FamilySizeGroup']

    # Check if target variable exists (for training data)
    has_target = 'Survived' in df.columns

    # Prepare the dataset
    available_features = [f for f in features_to_use if f in df.columns]
    X = df[available_features].copy()
    y = df['Survived'].copy() if has_target else None

    # Identify numerical and categorical columns
    numerical_features = [f for f in ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone'] if f in available_features]
    categorical_features = [f for f in ['Pclass', 'Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'FamilySizeGroup'] if f in available_features]

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

    return X, y, preprocessor, available_features, has_target

def initialize_models():
    """Initialize all models for comparison"""
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
    return models

def evaluate_models(models, preprocessor, X_train, y_train, X_test, y_test):
    """Evaluate all models with cross-validation"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    cv_scores = {}

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, model) in enumerate(models.items()):
        status_text.text(f'Training {name}...')
        start_time = time.time()

        # Create pipeline with preprocessing
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Cross-validation scores
        cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        cv_scores[name] = cv_score

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

        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'CV_Mean': cv_score.mean(),
            'CV_Std': cv_score.std(),
            'Training_Time': time.time() - start_time,
            'Pipeline': pipeline
        }

        progress_bar.progress((i + 1) / len(models))

    status_text.text('Model evaluation completed!')

    # Create results DataFrame
    results_df = pd.DataFrame({k: {metric: v[metric] for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'CV_Mean', 'CV_Std', 'Training_Time']}
                              for k, v in results.items()}).T
    results_df = results_df.sort_values('F1-Score', ascending=False)

    return results, results_df, cv_scores

def create_visualizations(df, results_df, cv_scores):
    """Create comprehensive visualizations"""

    # Model Performance Comparison
    st.subheader("📊 Model Performance Comparison")

    # Metrics comparison
    fig_metrics = go.Figure()

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    top_5_models = results_df.head(5)

    for metric in metrics:
        fig_metrics.add_trace(go.Bar(
            name=metric,
            x=top_5_models.index,
            y=top_5_models[metric],
            text=top_5_models[metric].round(3),
            textposition='auto',
        ))

    fig_metrics.update_layout(
        title="Top 5 Models Performance Metrics",
        xaxis_title="Models",
        yaxis_title="Score",
        barmode='group',
        height=500
    )

    st.plotly_chart(fig_metrics, use_container_width=True)

    # ROC-AUC vs F1-Score scatter plot
    fig_scatter = px.scatter(
        results_df.reset_index(),
        x='F1-Score',
        y='ROC-AUC',
        text='index',
        title="Model Performance: F1-Score vs ROC-AUC",
        labels={'index': 'Model'}
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(height=500)

    st.plotly_chart(fig_scatter, use_container_width=True)

def main():
    """Main Streamlit application"""

    # Title and description
    st.markdown('<h1 class="main-header">🚢 Titanic Survival Prediction</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Upload your dataset and get comprehensive survival predictions with advanced ML analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["📊 Data Upload & Overview", "🔍 Data Analysis", "🤖 Model Training", "📈 Predictions & Results"]
    )

    # Initialize session state
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False

    if page == "📊 Data Upload & Overview":
        show_data_upload_page()
    elif page == "🔍 Data Analysis":
        show_data_analysis_page()
    elif page == "🤖 Model Training":
        show_model_training_page()
    elif page == "📈 Predictions & Results":
        show_predictions_page()

def show_data_upload_page():
    """Data upload and overview page"""
    st.markdown('<h2 class="sub-header">📊 Data Upload & Overview</h2>', unsafe_allow_html=True)

    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your Titanic dataset or any compatible dataset with passenger information"
    )

    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            st.session_state.original_data = df
            st.session_state.data_uploaded = True

            # Display success message
            st.markdown("""
            <div class="success-box">
                ✅ <strong>Data uploaded successfully!</strong>
            </div>
            """, unsafe_allow_html=True)

            # Basic information
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("📋 Total Rows", df.shape[0])
            with col2:
                st.metric("📊 Total Columns", df.shape[1])
            with col3:
                st.metric("🎯 Has Target", "Yes" if 'Survived' in df.columns else "No")
            with col4:
                st.metric("❓ Missing Values", df.isnull().sum().sum())

            # Display first few rows
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # Column information
            st.subheader("📊 Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)

            # Basic statistics
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                st.subheader("📈 Basic Statistics")
                st.dataframe(df.describe(), use_container_width=True)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to get started")

        # Show sample data format
        st.subheader("📋 Expected Data Format")
        sample_data = pd.DataFrame({
            'PassengerId': [1, 2, 3],
            'Pclass': [3, 1, 3],
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John Bradley', 'Heikkinen, Miss. Laina'],
            'Sex': ['male', 'female', 'female'],
            'Age': [22, 38, 26],
            'SibSp': [1, 1, 0],
            'Parch': [0, 0, 0],
            'Ticket': ['A/5 21171', 'PC 17599', 'STON/O2. 3101282'],
            'Fare': [7.25, 71.28, 7.92],
            'Cabin': ['', 'C85', ''],
            'Embarked': ['S', 'C', 'S'],
            'Survived': [0, 1, 1]  # Optional for prediction datasets
        })
        st.dataframe(sample_data, use_container_width=True)
        st.caption("Note: 'Survived' column is optional for prediction datasets")

def show_data_analysis_page():
    """Data analysis and preprocessing page"""
    st.markdown('<h2 class="sub-header">🔍 Data Analysis & Preprocessing</h2>', unsafe_allow_html=True)

    if not st.session_state.get('data_uploaded', False):
        st.warning("⚠️ Please upload data first in the 'Data Upload & Overview' section.")
        return

    df = st.session_state.original_data

    # Outlier Analysis Section
    st.subheader("🎯 Outlier Detection & Handling")

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'PassengerId' in numerical_cols:
        numerical_cols.remove('PassengerId')
    if 'Survived' in numerical_cols:
        numerical_cols.remove('Survived')

    if numerical_cols:
        # Show outlier analysis
        outlier_summary = []
        for col in numerical_cols:
            outliers = detect_outliers_iqr(df, col)
            outlier_count = outliers.sum()
            outlier_percentage = (outlier_count / len(df)) * 100
            outlier_summary.append({
                'Column': col,
                'Outlier Count': outlier_count,
                'Outlier Percentage': f"{outlier_percentage:.1f}%"
            })

        outlier_df = pd.DataFrame(outlier_summary)
        st.dataframe(outlier_df, use_container_width=True)

        # Outlier handling options
        st.subheader("🛠️ Outlier Handling")
        outlier_method = st.selectbox(
            "Choose outlier handling method:",
            ["cap", "remove", "transform"],
            help="Cap: Replace outliers with boundary values, Remove: Delete outlier rows, Transform: Apply log transformation"
        )

        if st.button("🔧 Process Data"):
            df_processed = df.copy()

            # Handle outliers
            outlier_results = {}
            for col in numerical_cols:
                df_processed, outliers_handled = handle_outliers_iqr(df_processed, col, method=outlier_method)
                outlier_results[col] = outliers_handled

            # Feature engineering
            df_processed = engineer_features(df_processed)

            # Store processed data
            st.session_state.processed_data = df_processed
            st.session_state.data_processed = True
            st.session_state.outlier_results = outlier_results

            st.success("✅ Data preprocessing completed!")

            # Show processing results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Outliers Handled")
                for col, count in outlier_results.items():
                    st.metric(col, count)

            with col2:
                st.subheader("🆕 New Features Created")
                new_features = ['FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup', 'FamilySizeGroup']
                for feature in new_features:
                    if feature in df_processed.columns:
                        st.write(f"✓ {feature}")

            # Show before/after comparison
            st.subheader("📈 Before vs After Processing")

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Data Shape:**", df.shape)
                st.write("**Missing Values:**", df.isnull().sum().sum())

            with col2:
                st.write("**Processed Data Shape:**", df_processed.shape)
                st.write("**Missing Values:**", df_processed.isnull().sum().sum())

    else:
        st.info("No numerical columns found for outlier analysis.")

    # Data visualization
    if st.session_state.get('data_processed', False):
        st.subheader("📊 Data Visualizations")

        df_viz = st.session_state.processed_data

        # Survival distribution (if target exists)
        if 'Survived' in df_viz.columns:
            col1, col2 = st.columns(2)

            with col1:
                fig_survival = px.pie(
                    df_viz,
                    names='Survived',
                    title="Survival Distribution",
                    labels={'Survived': 'Survival Status'},
                    color_discrete_map={0: '#ff7f7f', 1: '#7fbf7f'}
                )
                st.plotly_chart(fig_survival, use_container_width=True)

            with col2:
                # Survival by class
                survival_by_class = df_viz.groupby(['Pclass', 'Survived']).size().unstack(fill_value=0)
                fig_class = px.bar(
                    survival_by_class.reset_index(),
                    x='Pclass',
                    y=[0, 1],
                    title="Survival by Passenger Class",
                    labels={'value': 'Count', 'variable': 'Survived'}
                )
                st.plotly_chart(fig_class, use_container_width=True)

        # Age and Fare distributions
        col1, col2 = st.columns(2)

        with col1:
            if 'Age' in df_viz.columns:
                fig_age = px.histogram(df_viz, x='Age', title="Age Distribution", nbins=30)
                st.plotly_chart(fig_age, use_container_width=True)

        with col2:
            if 'Fare' in df_viz.columns:
                fig_fare = px.histogram(df_viz, x='Fare', title="Fare Distribution", nbins=30)
                st.plotly_chart(fig_fare, use_container_width=True)

def show_model_training_page():
    """Model training and evaluation page"""
    st.markdown('<h2 class="sub-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)

    if not st.session_state.get('data_processed', False):
        st.warning("⚠️ Please process your data first in the 'Data Analysis' section.")
        return

    df = st.session_state.processed_data

    # Check if we have target variable for training
    has_target = 'Survived' in df.columns

    if not has_target:
        st.warning("⚠️ No target variable 'Survived' found. This dataset can only be used for predictions.")
        return

    st.subheader("🎯 Model Training Configuration")

    # Training options
    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)

    with col2:
        cv_folds = st.selectbox("Cross-Validation Folds", [3, 5, 10], index=1)

    if st.button("🚀 Train Models"):
        with st.spinner("Training models... This may take a few minutes."):

            # Prepare data
            X, y, preprocessor, features, _ = prepare_data_for_modeling(df)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            st.info(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

            # Initialize models
            models = initialize_models()

            # Train and evaluate models
            results, results_df, cv_scores = evaluate_models(
                models, preprocessor, X_train, y_train, X_test, y_test
            )

            # Store results
            st.session_state.model_results = results
            st.session_state.results_df = results_df
            st.session_state.cv_scores = cv_scores
            st.session_state.models_trained = True
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.preprocessor = preprocessor

            st.success("✅ Model training completed!")

    # Show results if models are trained
    if st.session_state.get('models_trained', False):
        st.subheader("📊 Model Performance Results")

        results_df = st.session_state.results_df

        # Display results table
        st.dataframe(results_df.round(4), use_container_width=True)

        # Best model information
        best_model_name = results_df.index[0]
        best_f1_score = results_df.loc[best_model_name, 'F1-Score']

        st.markdown(f"""
        <div class="success-box">
            🏆 <strong>Best Model:</strong> {best_model_name}<br>
            📈 <strong>F1-Score:</strong> {best_f1_score:.4f}
        </div>
        """, unsafe_allow_html=True)

        # Model performance visualizations
        create_visualizations(None, results_df, None)

        # Detailed evaluation of best model
        st.subheader(f"🔍 Detailed Analysis - {best_model_name}")

        best_pipeline = st.session_state.model_results[best_model_name]['Pipeline']
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # Predictions
        y_pred = best_pipeline.predict(X_test)
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]

        # Confusion Matrix
        col1, col2 = st.columns(2)

        with col1:
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                x=['Died', 'Survived'],
                y=['Died', 'Survived']
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col2:
            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC Curve (AUC = {roc_auc:.3f})',
                line=dict(color='darkorange', width=2)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='navy', width=2, dash='dash')
            ))
            fig_roc.update_layout(
                title='ROC Curve',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=400
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # Classification Report
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

def show_predictions_page():
    """Predictions and results page"""
    st.markdown('<h2 class="sub-header">📈 Predictions & Results</h2>', unsafe_allow_html=True)

    if not st.session_state.get('data_processed', False):
        st.warning("⚠️ Please process your data first in the 'Data Analysis' section.")
        return

    df = st.session_state.processed_data

    # Check if we have trained models
    if st.session_state.get('models_trained', False):
        st.subheader("🎯 Bulk Predictions")

        # Get best model
        results_df = st.session_state.results_df
        best_model_name = results_df.index[0]
        best_pipeline = st.session_state.model_results[best_model_name]['Pipeline']

        st.info(f"Using best model: **{best_model_name}** (F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.4f})")

        # Prepare data for prediction
        X, _, preprocessor, features, has_target = prepare_data_for_modeling(df)

        if st.button("🔮 Generate Predictions"):
            with st.spinner("Generating predictions..."):

                # Make predictions
                predictions = best_pipeline.predict(X)
                prediction_probabilities = best_pipeline.predict_proba(X)[:, 1]

                # Create results dataframe
                results_df_pred = df.copy()
                results_df_pred['Predicted_Survival'] = predictions
                results_df_pred['Survival_Probability'] = prediction_probabilities
                results_df_pred['Prediction_Confidence'] = np.where(
                    prediction_probabilities > 0.5,
                    prediction_probabilities,
                    1 - prediction_probabilities
                )

                # Store predictions
                st.session_state.predictions = results_df_pred

                st.success("✅ Predictions generated successfully!")

                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Passengers", len(predictions))
                with col2:
                    st.metric("Predicted Survivors", int(predictions.sum()))
                with col3:
                    st.metric("Predicted Deaths", int((1-predictions).sum()))
                with col4:
                    st.metric("Survival Rate", f"{predictions.mean():.1%}")

                # Prediction results
                st.subheader("📊 Prediction Results")

                # Show predictions table
                display_cols = ['PassengerId', 'Name', 'Sex', 'Age', 'Pclass', 'Predicted_Survival', 'Survival_Probability']
                available_display_cols = [col for col in display_cols if col in results_df_pred.columns]

                st.dataframe(
                    results_df_pred[available_display_cols].head(20),
                    use_container_width=True
                )

                # Download predictions
                csv = results_df_pred.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name="titanic_predictions.csv",
                    mime="text/csv"
                )

                # Prediction visualizations
                st.subheader("📈 Prediction Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Survival prediction distribution
                    fig_pred_dist = px.pie(
                        values=[int(predictions.sum()), int((1-predictions).sum())],
                        names=['Survived', 'Died'],
                        title="Predicted Survival Distribution",
                        color_discrete_map={'Survived': '#7fbf7f', 'Died': '#ff7f7f'}
                    )
                    st.plotly_chart(fig_pred_dist, use_container_width=True)

                with col2:
                    # Prediction confidence distribution
                    fig_conf = px.histogram(
                        results_df_pred,
                        x='Prediction_Confidence',
                        title="Prediction Confidence Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)

                # Survival by different factors
                if 'Pclass' in results_df_pred.columns:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Survival by class
                        class_survival = results_df_pred.groupby('Pclass')['Predicted_Survival'].mean()
                        fig_class = px.bar(
                            x=class_survival.index,
                            y=class_survival.values,
                            title="Predicted Survival Rate by Class",
                            labels={'x': 'Passenger Class', 'y': 'Survival Rate'}
                        )
                        st.plotly_chart(fig_class, use_container_width=True)

                    with col2:
                        # Survival by sex
                        if 'Sex' in results_df_pred.columns:
                            sex_survival = results_df_pred.groupby('Sex')['Predicted_Survival'].mean()
                            fig_sex = px.bar(
                                x=sex_survival.index,
                                y=sex_survival.values,
                                title="Predicted Survival Rate by Gender",
                                labels={'x': 'Gender', 'y': 'Survival Rate'}
                            )
                            st.plotly_chart(fig_sex, use_container_width=True)

    else:
        st.info("🤖 Please train models first in the 'Model Training' section to generate predictions.")

        # If no target variable, show option to use pre-trained model
        if 'Survived' not in df.columns:
            st.subheader("🔮 Alternative: Use Pre-trained Model")
            st.info("Since your dataset doesn't have a target variable, you can use a pre-trained model for predictions.")

            if st.button("🚀 Use Default Pre-trained Model"):
                st.warning("This feature would require a pre-trained model file. For now, please use a dataset with 'Survived' column for training.")

if __name__ == "__main__":
    main()