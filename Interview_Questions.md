# Interview Questions:
### 1. What are the different types of missing data?
Missing data types:
- MCAR (Missing Completely at Random):** There’s no pattern; data is missing by pure chance.
- MAR (Missing at Random):** The missingness is related to seen data, not the missing variable itself.
- NMAR (Not Missing at Random):** Missingness is related to the missing values themselves.

### 2. How do you handle categorical variables?
You turn categorical variables into numbers using encoding methods:
- Label encoding** for binary or ordinal categories.
- One-hot encoding** for variables with more than two categories and no true order.

### 3. What is the difference between normalization and standardization?

- Normalization** scales all the values between 0 and 1, useful when data doesn’t follow a normal distribution.
- Standardization** transforms data so it has a mean of 0 and a standard deviation of 1, useful for many ML algorithms.

### 4. How do you detect outliers?
- Use boxplots, histograms, or scatter plots.
- Statistically: Identify values that fall outside the 1st and 99th percentiles, or those with a z-score greater than 3.

### 5. Why is preprocessing important in ML?
Good models need good data. Preprocessing removes noise, handles missing/unfair/bias data, allows algorithms to learn accurate patterns, and improves accuracy and reliability.

### 6. What is one-hot encoding vs label encoding?

- Label encoding:** Assigns each category a number (0,1,2).
- One-hot encoding:** Makes a new binary column for each category.
- Use one-hot if there’s no ordering to the categories.

### 7. How do you handle data imbalance?
You can:
- Oversample the minority class.
- Undersample the majority class.
- Use techniques like SMOTE.
- Adjust model class weights.

### 8. Can preprocessing affect model accuracy?
Yes, greatly! Good preprocessing gives cleaner, more consistent data and usually improves model accuracy, stability, and reproducibility.
