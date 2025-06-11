![logo_](https://github.com/user-attachments/assets/cd06965c-bc3d-4dd7-bf52-e5cdb3e003e2)

# Residuals_Matter

This notebook provides a practical walkthrough of the essential components of Exploratory Data Analysis and Predictive Modelling. The techniques showcased are fundamental for transforming raw data into actionable insights.

<i>Fernando García Bastante<br>
Universidade de Vigo<br>
For Educational Purposes</i>

---
## 🎯 Objective

- The goal of this notebook is to demonstrate how to apply Python tools to analyze, transform, and model data from scratch. It covers everything from initial cleaning to benchmarking predictive models including linear regression, penalized regressions, PCA, PLS, and random forest.
---
## 🧪 Requirements

1. Make sure Python 3.12 is installed. V.g. with conda:
   ```bash
   conda create -n myeda_env python=3.12
   conda activate myeda_env
   ```
3. Install dependencies:
   ```bash
   conda install conda-forge::ipython scikit-learn pingouin pandas matplotlib cython openpyxl seaborn jupyterlab tabulate statsmodels graphviz python-graphviz pydot shap ipywidgets
   ```
4. Launch Jupyter lab
   ```bash
   jupyter lab
   ```
   ...and load the file: eda_dm.ipynb
---
## 🧰 Techniques

- Linear/Lasso/Ridge regressions
- RFECV, PCA, PLS
- Random Forest
- Cross-Validation
- SHAP
---
## ![image](https://github.com/user-attachments/assets/d5dcb612-da3e-4c32-8a0e-a8eaf9ebfa15) DataBase

The database employed to illustrate the tools and techniques presented in this Jupyter Notebook is derived from the distinguished article: <i>Chemical Descriptors for a Large-Scale Study on Drop-Weight Impact Sensitivity of High Explosives</i>. This study investigates the relationship between the results of the drop-weight impact test—used to evaluate the handling sensitivity of high explosives—and a compendium of molecular and chemical descriptors associated with the explosives under examination.

<b>Frank W. Marrs, Jack V. Davis, Alexandra C. Burch, Geoffrey W. Brown, Nicholas Lease, Patricia L. Huestis, Marc J. Cawkwell, and Virginia W. Manner (2023).</b>
<i>Chemical Descriptors for a Large-Scale Study on Drop-Weight Impact Sensitivity of High Explosives</i>. <i>Journal of Chemical Information and Modeling.</i><br>
https://pubs.acs.org/doi/10.1021/acs.jcim.2c01154<br>

<i>**DISCLAIMER:** This code is provided for educational and demonstrative purposes only. Its sole objective is to illustrate Python techniques for data visualisation and analysis. The datasets used in the examples serve purely as illustrative material; no comprehensive or contextual analysis of these specific datasets has been undertaken or is implied. The primary focus remains on the implementation of technical methodologies, rather than the in-depth interpretation of the data itself. For the purposes of this notebook, minor modifications have been introduced into the database in order to facilitate the illustration of certain techniques presented herein.</i>

---
## 📑 Notebook Structure

1. **Data Loading and Cleaning**
3. **Variable Transformation**
4. **Exploratory Data Analysis**
5. **Linear Regression with `statsmodels`**
6. **Stepwise Regression via AIC/BIC**
7. **Regression with `scikit-learn` and Cross-Validation:**
8. **Model Comparison:**
9. **Principal Component Regression (PCR)**
10. **Partial Least Squares Regression (PLS)**
11. **Random Forest Regression**
12. **Shap (SHapley Additive exPlanations)**
13. **Tree Visualization with `graphviz`**
---

## 📚 Recommended Bibliography and Resources (in English)

1. [**Pandas Documentation – Data Manipulation**](https://pandas.pydata.org/docs/)  
   Official reference for `pandas`, the core library used for data loading, cleaning, and transformation.

2. [**Seaborn: Statistical Data Visualization**](https://seaborn.pydata.org/tutorial.html)  
   Guide to Seaborn's functionality, including correlation plots, pairplots, and heatmaps for EDA.

3. [**Scikit-Learn User Guide**](https://scikit-learn.org/stable/user_guide.html)  
   Covers regression models, feature selection (`SelectKBest`, `RFE`), pipelines, cross-validation, PCA, and more.

4. [**Statsmodels Documentation**](https://www.statsmodels.org/stable/index.html)  
   Useful for linear models, OLS diagnostics, VIF analysis, and statistical hypothesis testing.

5. [**Hands-On Exploratory Data Analysis with Python**](https://realpython.com/python-data-cleaning-numpy-pandas/)  
   Practical guide to cleaning, analyzing, and visualizing datasets using NumPy and Pandas.
   
7. [**SHAP documentation**](https://shap.readthedocs.io/en/latest/)
   Approach to explain the output of any machine learning model.

---
## 📘 Summary of Workflow

This notebook exemplifies a comprehensive data science pipeline implemented using Python. It encompasses the following key stages:

---

### **1. Data Loading and Initial Appraisal**

- **Seamless Data Ingestion:** Demonstrates proficient use of the `pandas` library to import datasets.
- **Structural Inspection:** Utilizes methods such as `.shape`, `.columns`, and `.dtypes` to examine dataset dimensions, column names, and data types—crucial for identifying structural inconsistencies.
- **Preliminary Data Integrity Checks:** Functions like `.head()`, `.tail()`, and `.sample()` offer a visual snapshot to verify data quality and detect loading anomalies.
- **Concise Dataset Overview:** The `.info()` method summarizes non-null counts, types, and memory usage.
- **Descriptive Statistics:** The use of `.describe(include='all')` provides statistical summaries for both numeric and categorical features.

---

### **2. Comprehensive Data Cleansing and Preprocessing**

- **Missing Data Handling:** Identifies and quantifies missing values using `.isnull().sum()`, and explores various imputation strategies (mean, median, mode, or advanced methods).
- **Duplicate Detection and Removal:** Employs `.duplicated()` and `.drop_duplicates()` to ensure data uniqueness.
- **Data Type Correction:** Uses `astype()` to convert columns to appropriate types (e.g., numeric, datetime).
- **Inconsistency Resolution:** Standardizes categorical entries and corrects formatting issues to maintain coherence.

---

### **3. Univariate Analysis**

- **Individual Feature Examination:** Analyses each variable in isolation to understand distribution and quality.
- **Numerical Features:** Visualized using histograms, KDE plots, and box plots to reveal distribution patterns and outliers.
- **Categorical Features:** Evaluated to explore class distributions.
- **Descriptive Measures:** Summarizes central tendency, dispersion, and shape via statistical metrics (mean, median, IQR, etc.).

---

### **4. Bivariate and Multivariate Analysis**

- **Numerical-Numerical Relationships:** Uses scatter plots and correlation matrices (heatmaps) to assess linear associations.
- **Categorical-Numerical Insights:** Explores grouped box plots, violin plots, and aggregated bar plots.
- **Multidimensional Exploration:** Employs `seaborn.pairplot()` and variable encodings (hue, size) to visualize interactions across multiple dimensions.

---

### **5. Outlier Detection and Management**

- **Visual and Statistical Techniques:** Identifies outliers using box plots, Z-scores, and the IQR method, enabling thoughtful exclusion or treatment.

---

### **6. Feature Engineering and Transformation**

- **Derived Features:** Highlights opportunities to construct new features (e.g., from date fields or binning) to enrich modeling potential.
- **Scaling and Encoding:** Implements standardization/normalization and categorical encoding (e.g., One-Hot, Label Encoding) as needed for downstream modeling.

---

### **7. Modeling Pipeline Preparation**

- **Feature Selection:** Uses correlation analysis, domain knowledge, and model-based importance to select relevant predictors.
- **Data Partitioning:** Applies train-test splits and optionally cross-validation to ensure robust model evaluation.

---

### **8. Predictive Modeling**

- **Regression Techniques:**
  - *Linear Regression:* Models linear dependencies for interpretability.
  - *Random Forest Regressor:* Captures nonlinear relationships using ensemble learning.

- **Model Training and Prediction:** Fits the model on training data and generates predictions for unseen instances.

---

### **9. Model Evaluation and Optimization**

- **Performance Metrics:** Assesses model accuracy via MSE, RMSE, and R².
- **Hyperparameter Tuning:** Where applicable, employs GridSearch or RandomizedSearch for parameter optimization.
- **Interpretation and Insight:** Relates performance to domain-specific expectations and explores model behavior.

---
