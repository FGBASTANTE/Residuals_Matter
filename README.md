![logo_](https://github.com/user-attachments/assets/cd06965c-bc3d-4dd7-bf52-e5cdb3e003e2)

# Residuals_Matter

- This notebook provides a practical walkthrough of the essential components of Exploratory Data Analysis and Predictive Modelling. The techniques showcased are fundamental for transforming raw data into actionable insights
---
## 🎯 Objective

- The goal of this notebook is to demonstrate how to apply Python tools to analyze, transform, and model data from scratch. It covers everything from initial cleaning to benchmarking predictive models including linear regression, penalized regressions, PCA, PLS, and random forest.
---
## 🧪 Requirements

1. Make sure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels openpyxl pingouin graphviz python-graphviz dot
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter-lab eda_dm.ipynb
   ```
---
## 🧰 Techniques

- Linear/Lasso/Ridge regressions
- RFECV, PCA, PLS
- Random Forest
- Cross-validation
---
## 📑 Notebook Structure

1. **Data Loading and Cleaning:**
   - Load data from Excel
   - Column renaming and cleaning
   - Removing duplicates and constant columns

2. **Variable Transformation:**
   - Type conversion
   - New feature creation
   - Selection of numerical columns

3. **Exploratory Data Analysis:**
   - Descriptive statistics
   - Outlier visualization (z-score, IQR)
   - Correlation matrix, histograms, boxplots, scatter plots

4. **Linear Regression with `statsmodels`:**
   - VIF analysis, feature selection
   - Diagnostic plots

5. **Stepwise Regression via AIC/BIC**

6. **Regression with `scikit-learn` and Cross-Validation:**
   - `LinearRegression`, `RFECV`

7. **Model Comparison:**
   - Linear, Lasso, Ridge
   - Evaluation with RMSE and R²

8. **Principal Component Regression (PCR)**

9. **Partial Least Squares Regression (PLS)**

10. **Random Forest Regression:**
    - Hyperparameter tuning with `GridSearchCV`
    - Feature importance
    - OOB score

11. **Tree Visualization with `graphviz`**

## 📚 Recommended Bibliography and Resources (in English)

1. [**Pandas Documentation – Data Manipulation**](https://pandas.pydata.org/docs/)  
   Official reference for `pandas`, the core library used for data loading, cleaning, and transformation.

2. [**Seaborn: Statistical Data Visualization**](https://seaborn.pydata.org/tutorial.html)  
   Guide to Seaborn's functionality, including correlation plots, pairplots, and heatmaps for EDA.

3. [**Scikit-Learn User Guide**](https://scikit-learn.org/stable/user_guide.html)  
   Covers regression models, feature selection (`SelectKBest`, `RFE`), pipelines, cross-validation, PCA, and more.

4. [**Statsmodels Documentation**](https://www.statsmodels.org/stable/index.html)  
   Useful for linear models, OLS diagnostics, VIF analysis, and statistical hypothesis testing.

5. [**A Visual Guide to Feature Selection**](https://towardsdatascience.com/a-feature-selection-toolbox-in-python-b64dd23710f0)  
   Explains and compares multiple feature selection methods with examples in Python.

6. [**Hands-On Exploratory Data Analysis with Python**](https://realpython.com/python-data-cleaning-numpy-pandas/)  
   Practical guide to cleaning, analyzing, and visualizing datasets using NumPy and Pandas.

