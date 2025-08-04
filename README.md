# Titanic Data Cleaning & Preprocessing – ML Internship Task
# Data_Cleaning_and_Preprocessing_ML
Data clean and preprocess of Titanic CSV data for Machine Learning.

This project demonstrates a complete data preprocessing pipeline for the Titanic dataset. It's shows how to clean, prepare, and transform raw tabular data to make it suitable for machine learning (ML) models.  
The main goal is to ensure the dataset is clean, well-structured, and free from issues that might mislead an ML algorithm.


### 1. Data Import & Exploration

- Loaded the Titanic dataset using Pandas.
- Used `read_csv` to import the CSV file.
- To examine the structure, understand the columns, data types, and check for missing values or anomalies.
  

### 2. Handling Missing Values

- Checked columns for missing data and decided how to handle them.
- Dropped the `Cabin` column due to too many missing values.
- Filled missing `Age` and `Fare` with their median values (to avoid bias from extreme outliers).
- Filled missing `Embarked` values with the most common embarkation point (mode).
- ML models can't handle missing values; imputation provides sensible fills without distorting data.

  
### 3. Encoding Categorical Variables

- Converted categorical columns into numerical values that can be used by ML algorithms.
- Encoded `Sex` as 0 (male) and 1 (female).
- One-hot encoded `Embarked` column to create separate columns for each embarkation point.
- ML models require all input data to be numeric.


### 4. Feature Scaling (Standardization)

- Normalized / standardized the numeric columns (`Age`, `Fare`, `SibSp`, `Parch`).
- Used StandardScaler from scikit-learn to give each feature a mean of 0 and standard deviation of 1.
- Helps ML models converge faster and prevents features with large values from dominating the model.


### 5. Outlier Detection and Removal

- Identified and removed extreme outliers from the `Age` and `Fare` columns.
- Used boxplots for visualization.
- Removed data points outside the 1st and 99th percentiles for these columns.
- Outliers can skew the learning process and bias the results of some ML models.

### 6. Save the Cleaned Data

- Saved the final cleaned dataset to a new CSV file.
- Used Pandas `to_csv` method.
- To provide a ready-to-use dataset for ML modeling.


## How to Use

1. Download this repository in zip file from Kaggle.
2. Place your Titanic dataset CSV into the repo folder.
3. Run the Colab notebook to preprocess your file (Titanic_Dataset.ipynb).
4. The processed dataset (`Titanic_Cleaned.csv`) will be saved in the same directory, ready for further ML experiments.
5. Review code and explanations to understand each cleaning step.

## Why Data Preprocessing?

Data cleaning and transformation is one of the most crucial steps in the entire ML pipeline. Without it, even the most sophisticated algorithms will perform poorly.  
Proper preprocessing leads to better, more robust, and reliable models.
