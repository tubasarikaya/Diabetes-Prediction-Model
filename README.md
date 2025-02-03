# Diabetes Prediction Model using Logistic Regression

This project implements a machine learning model to predict diabetes using various health metrics. The model uses logistic regression and includes comprehensive data preprocessing, analysis, and evaluation steps.

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature scaling
- Model training and evaluation
- Visualization of results
- Feature importance analysis

## Dataset

The project uses the diabetes dataset with the following features:
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Age
- Outcome (Target variable)

## Requirements

```
numpy
pandas
seaborn
matplotlib
scikit-learn
```

## Project Structure

The project follows these main steps:

1. **Data Loading and Initial Exploration**
   - Reading the dataset
   - Basic data overview
   - Memory usage analysis
   - Missing value detection

2. **Data Preprocessing**
   - Handling zero values
   - Missing value imputation
   - Outlier detection and treatment
   - Feature scaling using StandardScaler

3. **Exploratory Data Analysis**
   - Statistical summaries
   - Correlation analysis
   - Distribution analysis
   - Target variable analysis

4. **Model Building**
   - Train-test split (80-20)
   - Logistic Regression implementation
   - Model training and prediction

5. **Model Evaluation**
   - ROC AUC Score
   - F1 Score
   - Recall
   - Precision
   - Accuracy
   - Confusion Matrix
   - Classification Report

## Key Functions

- `check_df()`: Provides comprehensive dataframe information
- `grab_col_names()`: Categorizes variables based on their types
- `num_summary()`: Generates numerical summaries with optional plotting
- `correlation_matrix()`: Creates correlation heatmap
- `evaluate_model()`: Calculates and displays various model metrics
- `plot_feature_importance()`: Visualizes feature importance
- `plot_confusion_matrix()`: Displays confusion matrix with accuracy score

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Install required packages:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## Usage

1. Ensure your data file is in the correct location
2. Update the file path in the code:
```python
df = pd.read_csv("path_to_your_data/diabetes.csv")
```
3. Run the script:
```python
python diabetes_prediction.py
```

## Model Performance

The logistic regression model provides:
- ROC AUC Score evaluation
- Classification metrics (Precision, Recall, F1-Score)
- Visual representation of confusion matrix
- Feature importance analysis

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
