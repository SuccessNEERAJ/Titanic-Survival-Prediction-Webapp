# 🚢 Titanic Survival Prediction - Streamlit App

A comprehensive machine learning web application for predicting Titanic passenger survival using advanced data analysis and multiple ML algorithms.

## ✨ Features

- **📊 Interactive Data Upload**: Upload your own CSV datasets
- **🔍 Advanced Data Analysis**: Automatic outlier detection and handling
- **🛠️ Data Preprocessing**: Feature engineering and data cleaning
- **🤖 Multiple ML Models**: Train and compare 10 different algorithms
- **📈 Rich Visualizations**: Interactive charts and performance metrics
- **🔮 Bulk Predictions**: Generate survival predictions for entire datasets
- **📥 Export Results**: Download predictions as CSV files

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📋 Complete Step-by-Step Guide

### 🎯 **Phase 1: Model Training (Using Training Data)**

#### Step 1: Upload Training Data
1. Navigate to **"📊 Data Upload & Overview"** tab
2. Click **"Browse files"** button
3. Go to `Input/For training/` folder
4. Select and upload **`train.csv`** file
5. Review the data overview, statistics, and column information

![Data Upload Page](Dashboard%20Screenshots/1.%20Data%20Upload%20Page.png)

#### Step 2: Data Analysis & Preprocessing
1. Navigate to **"🔍 Data Analysis"** tab
2. Review the **Outlier Detection & Handling** section
3. Choose your preferred outlier handling method:
   - **Cap**: Replace outliers with boundary values (recommended)
   - **Remove**: Delete outlier rows
   - **Transform**: Apply log transformation
4. Click **"🔧 Process Data"** button
5. Review the processing results and new features created

![Data Analysis Page](Dashboard%20Screenshots/2.%20Data%20Analysis%20and%20Outlier%20Handling%20Page.png)

#### Step 3: Explore Data Visualizations
- View survival distribution charts
- Analyze survival rates by passenger class
- Examine age and fare distributions

![Data Visualizations](Dashboard%20Screenshots/3.%20Data%20Visualization.png)

#### Step 4: Train Machine Learning Models
1. Navigate to **"🤖 Model Training"** tab
2. Configure training parameters:
   - **Test Set Size**: 0.2 (20% for testing)
   - **Cross-Validation Folds**: 5 (recommended)
3. Click **"🚀 Train Models"** button
4. Wait for all 10 models to complete training

![Model Training Page](Dashboard%20Screenshots/4.%20Model%20Training%20Page.png)

#### Step 5: Review Model Performance
- Examine the **Model Performance Results** table
- Identify the best performing model (highest F1-Score)
- Review performance comparison charts

![Model Performance Results](Dashboard%20Screenshots/5.%20Model%20Performace%20Results.png)
![Model Performance Comparison](Dashboard%20Screenshots/6.%20Model%20Performance%20Comparison.png)

#### Step 6: Analyze Best Model Details
- View confusion matrix and ROC curve
- Review classification report
- Understand model strengths and weaknesses

![Detailed Analysis](Dashboard%20Screenshots/7.%20Detailed%20analysis%20of%20Best%20Model.png)

### 🎯 **Phase 2: Making Predictions (Using Test Data)**

#### Step 7: Upload Test Data
1. Return to **"📊 Data Upload & Overview"** tab
2. Click **"Browse files"** button again
3. Go to `Input/For testing/` folder
4. Select and upload **`test.csv`** file
5. Review the test data overview

#### Step 8: Process Test Data
1. Navigate to **"🔍 Data Analysis"** tab
2. Apply the same outlier handling method used for training
3. Click **"🔧 Process Data"** button
4. Ensure consistent preprocessing with training data

#### Step 9: Generate Predictions
1. Navigate to **"📈 Predictions & Results"** tab
2. The app will automatically use the best trained model
3. Click **"🔮 Generate Predictions"** button
4. Review prediction summary statistics

![Predictions Page](Dashboard%20Screenshots/8.%20Predictions%20Page.png)

#### Step 10: Download and Analyze Results
1. Review the **Prediction Results** table
2. Click **"📥 Download Predictions as CSV"** to save results
3. Analyze prediction patterns and confidence levels
4. Explore survival rate visualizations by different factors

![Prediction Results](Dashboard%20Screenshots/9.%20Predictions%20Result%20and%20Download%20as%20CSV.png)
![Prediction Analysis](Dashboard%20Screenshots/10.%20Prediction%20Analysis.png)

## 💡 **Key Workflow Tips**

### ✅ **Best Practices**
- **Always use the same outlier handling method** for both training and test data
- **Process training data completely** before uploading test data
- **Keep the app running** throughout the entire workflow to maintain trained models
- **Download predictions immediately** after generation to avoid losing results

### ⚠️ **Important Notes**
- The trained models are stored in session state and will be lost if you refresh the page
- Test data should have the same structure as training data (except for the 'Survived' column)
- For best results, ensure your test data comes from the same distribution as training data

## 🤖 Supported Models

The application trains and compares 10 different machine learning algorithms:

- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Ensemble of decision trees with voting
- **Gradient Boosting**: Sequential boosting algorithm
- **Support Vector Machine (SVM)**: Kernel-based classification
- **K-Nearest Neighbors**: Instance-based learning
- **Naive Bayes**: Probabilistic classifier
- **Decision Tree**: Single tree-based classifier
- **AdaBoost**: Adaptive boosting ensemble
- **Extra Trees**: Extremely randomized trees
- **XGBoost**: Extreme gradient boosting (often best performer)

## 📊 Data Format Requirements

### 📁 **Folder Structure**
```
Input/
├── For training/
│   └── train.csv          # Training data with 'Survived' column
└── For testing/
    └── test.csv           # Test data without 'Survived' column
```

### 📋 **Required Columns**
Your CSV files should contain passenger information with these columns:

#### **Training Data (train.csv)**
- `PassengerId`: Unique identifier (integer)
- `Pclass`: Passenger class (1=First, 2=Second, 3=Third)
- `Name`: Passenger name (string)
- `Sex`: Gender (male/female)
- `Age`: Age in years (numeric, can have missing values)
- `SibSp`: Number of siblings/spouses aboard (integer)
- `Parch`: Number of parents/children aboard (integer)
- `Ticket`: Ticket number (string)
- `Fare`: Passenger fare (numeric, can have missing values)
- `Cabin`: Cabin number (string, can be empty)
- `Embarked`: Port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)
- **`Survived`**: Survival status (0=Died, 1=Survived) - **Required for training**

#### **Test Data (test.csv)**
- Same columns as training data **except** `Survived` column
- Used for generating predictions after model training

## 🎯 Key Features

### Automatic Feature Engineering
- **FamilySize**: Total family members aboard
- **IsAlone**: Binary indicator for solo travelers
- **Title**: Extracted from passenger names
- **AgeGroup**: Age categories (Child, Teen, Young Adult, Adult, Senior)
- **FareGroup**: Fare quartiles (Low, Medium, High, Very High)
- **FamilySizeGroup**: Family size categories

### Advanced Data Processing
- **Outlier Detection**: IQR-based outlier identification
- **Missing Value Handling**: Intelligent imputation strategies
- **Data Scaling**: Standardization for numerical features
- **Encoding**: One-hot encoding for categorical variables

### Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC scores
- Cross-validation results
- Confusion matrices
- ROC curves

## 🛠️ Technical Details

- **Framework**: Streamlit for web interface
- **ML Library**: Scikit-learn with XGBoost
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas and NumPy
- **Styling**: Custom CSS for enhanced UI

## 📁 Project Structure

```
Titanic Survival Prediction/
├── streamlit_app.py                    # Main Streamlit application (878 lines)
├── titanic_survival_prediction.py     # Original ML pipeline script
├── requirements.txt                    # Python dependencies
├── README.md                          # This documentation
├── Dashboard Screenshots/             # UI screenshots for reference
│   ├── 1. Data Upload Page.png
│   ├── 2. Data Analysis and Outlier Handling Page.png
│   ├── 3. Data Visualization.png
│   ├── 4. Model Training Page.png
│   ├── 5. Model Performace Results.png
│   ├── 6. Model Performance Comparison.png
│   ├── 7. Detailed analysis of Best Model.png
│   ├── 8. Predictions Page.png
│   ├── 9. Predictions Result and Download as CSV.png
│   └── 10. Prediction Analysis.png
├── Input/                             # Data folders
│   ├── For training/
│   │   └── train.csv                  # Training dataset
│   └── For testing/
│       └── test.csv                   # Test dataset
└── Output/                            # Generated predictions
    └── titanic_predictions.csv       # Sample output file
```

## 🔧 Troubleshooting

### **Common Issues & Solutions**

#### **1. "Please upload data first" Warning**
- **Solution**: Make sure you've uploaded a CSV file in the "Data Upload & Overview" tab first

#### **2. "Please process your data first" Warning**
- **Solution**: Complete the data preprocessing in the "Data Analysis" tab before training models

#### **3. Models not training**
- **Check**: Ensure your training data has a 'Survived' column
- **Check**: Verify data has been processed successfully
- **Solution**: Try refreshing the page and starting over

#### **4. Predictions not generating**
- **Check**: Ensure models have been trained first
- **Check**: Verify test data has been processed
- **Solution**: Make sure you're using the same outlier handling method for both training and test data

#### **5. Download not working**
- **Solution**: Ensure predictions have been generated successfully
- **Try**: Right-click the download button and "Save link as..."

### **Performance Tips**
- **Large datasets**: Consider using smaller test sizes (10-15%) for faster training
- **Memory issues**: Close other applications while training models
- **Slow performance**: Try using fewer cross-validation folds (3 instead of 5)

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements! Areas for enhancement:
- Additional machine learning algorithms
- More advanced feature engineering
- Enhanced visualizations
- Model persistence and loading
- Batch processing capabilities

## 📄 License

This project is open source and available under the MIT License.

---

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the step-by-step guide
3. Ensure all dependencies are properly installed
4. Verify your data format matches the requirements

**Happy Predicting! 🚢✨**