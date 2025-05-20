![logo_](https://github.com/user-attachments/assets/cd06965c-bc3d-4dd7-bf52-e5cdb3e003e2)

  # Residuals_Matter
This notebook provides a practical walkthrough of the essential components of Exploratory Data Analysis and Predictive Modelling. The techniques showcased are fundamental for transforming raw data into actionable insights
---
## 🎯 Objective

The goal of this notebook is to demonstrate how to apply Python tools to analyze, transform, and model data from scratch. It covers everything from initial cleaning to benchmarking predictive models including linear regression, penalized regressions, PCA, PLS, and random forest.
---
## 🧰 Techniques and Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`, `statsmodels`
- `graphviz`, `openpyxl`
- Cross-validation, Lasso/Ridge regressions, RFECV, PCA, PLS, Random Forest
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
### 🧪 Requirements

1. Make sure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn statsmodels openpyxl pingouin graphviz python-graphviz dot
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter-lab eda_dm.ipynb
   ```



* **Data Loading and Initial Appraisal:**
    * **Seamless Data Ingestion:** Demonstrates the proficient use of the `pandas` library to import datasets from various file formats (e.g., CSV, Excel), ensuring a smooth start to the analytical journey.
    * **Preliminary Structural Assessment:** Employs functions such as `.shape` to quickly grasp the dimensions of the dataset (number of rows and columns).
    * **Column and Data Type Scrutiny:** Utilises `.columns` to review feature names and `.dtypes` to ascertain the initial data types assigned to each column, which is crucial for identifying immediate inconsistencies.
    * **Snapshotting Data Integrity:** Leverages `.head()`, `.tail()`, and `.sample()` to inspect random subsets of data, offering a first glance at the content and format, and aiding in the early detection of potential loading errors or unexpected values.
    * **Quantitative Overview:** Introduces `.info()` for a concise summary of the dataset, including data types, non-null counts for each column, and memory usage, providing a holistic view of the data's completeness.
    * **Statistical First Impressions:** Makes use of `.describe(include='all')` to generate descriptive statistics (count, mean, standard deviation, min/max, quartiles for numerical data; count, unique, top, frequency for categorical data), offering vital clues about the distribution and central tendencies of each attribute.

* **Comprehensive Data Cleansing and Preprocessing:**
    * **Systematic Missing Value Management:** Details robust methods for identifying missing data points (e.g., using `.isnull().sum()`). It then explores various imputation strategies, from simple mean, median, or mode imputation for numerical and categorical features respectively, to more sophisticated approaches if applicable, explaining the rationale and potential impact of each choice.
    * **Duplicate Record Neutralisation:** Showcases techniques to pinpoint and eliminate duplicate entries using `.duplicated()` and `.drop_duplicates()`, ensuring that the analysis is based on unique and accurate observations.
    * **Data Type Rectification and Standardisation:** Illustrates the importance of correcting data types (e.g., converting strings to numerical types or datetime objects) using functions like `astype()`. This step is vital for accurate computation and appropriate visualisation.
    * **Handling Inconsistent Data Entries:** May cover strategies for addressing inconsistencies in categorical data, such as standardising labels or correcting typographical errors, to ensure data uniformity.

* **In-depth Univariate Analysis:**
    * **Individual Variable Deep Dive:** Emphasises the importance of analysing each variable in isolation to understand its intrinsic characteristics before examining relationships.
    * **Visualising Numerical Distributions:** Demonstrates the creation of histograms and kernel density estimates (KDEs) using `matplotlib` and `seaborn` to visualise the shape, skewness, and modality of numerical data distributions. Box plots are employed to highlight central tendency, spread, and potential outliers.
    * **Exploring Categorical Data Frequencies:** Utilises count plots and bar charts to display the frequency distribution of categorical variables, revealing the prevalence of different categories.
    * **Interpreting Descriptive Statistics:** Reinforces the calculation and interpretation of key statistical measures such as mean, median, mode, variance, standard deviation, and interquartile range to quantitatively summarise individual variables.

* **Insightful Bivariate and Multivariate Analysis:**
    * **Uncovering Inter-Variable Relationships:** Focuses on exploring the connections and interactions between pairs of variables (bivariate) and among multiple variables (multivariate).
    * **Numerical vs. Numerical Relationships:** Employs scatter plots to visually inspect the relationship (linear, non-linear, strength, direction) between two numerical variables. Correlation matrices, often visualised as heatmaps, are used to quantify the linear association between all pairs of numerical variables.
    * **Categorical vs. Numerical Relationships:** Utilises grouped box plots, violin plots, or bar charts (showing mean/median) to compare the distribution of a numerical variable across different categories of a categorical variable.
    * **Multidimensional Visualisations:** May introduce techniques like pair plots (`seaborn.pairplot`) for a matrix of scatterplots and histograms, offering a comprehensive overview of relationships in datasets with multiple numerical variables, or using hue/size/style encodings in plots to incorporate additional dimensions.

* **Outlier Detection and Considered Treatment:**
    * **Identifying Anomalous Data Points:** Showcases statistical methods (such as the Z-score or the Interquartile Range (IQR) criterion) and visual techniques (primarily box plots and scatter plots) to effectively identify observations that deviate significantly from the norm.

* **Implicit Feature Augmentation and Engineering:**
    * **Deriving Value from Existing Data:** While not the primary focus, the EDA process naturally uncovers opportunities for feature engineering. The notebook may implicitly guide towards or demonstrate the creation of new, more informative features from existing ones (e.g., binning numerical variables, extracting components from datetime objects, creating interaction terms) which can significantly enhance model performance.

* **Data Visualisation for Exploration&Communication:**
    * **Crafting Compelling Narratives:** Reinforces that a key outcome of EDA is the ability to tell a story with data. This is achieved through the creation of clear, concise, and impactful visualisations.
    * **Leveraging `matplotlib` and `seaborn`:** Demonstrates the effective use of these powerful Python libraries to generate a wide array of static and potentially interactive plots, tailored to the specific type of data and the insight being conveyed.
    * **Customisation for Clarity and Impact:** Highlights the importance of customising plots (e.g., titles, labels, colours, annotations) to improve readability, interpretability, and aesthetic appeal, ensuring that visualisations effectively communicate findings to both technical and non-technical audiences.
    * **Iterative Visual Exploration:** Emphasises that visualisation in EDA is often an iterative process, where initial plots lead to further questions and refined visualisations.
  * **Feature Preparation for Modelling:**
    * **Feature Selection:** Discusses methods used to select the most relevant features identified during EDA to improve model performance and reduce dimensionality (e.g., based on correlation analysis, domain knowledge, or feature importance from preliminary models).
    * **Feature Scaling/Encoding:** Details the necessity and application of techniques like Standardisation or Normalisation for numerical features, and One-Hot Encoding or Label Encoding for categorical features to make them suitable for machine learning algorithms.
* **Data Splitting and Model Preparation:**
    * **Train-Test Split:** Explains the critical step of dividing the dataset into training and testing subsets to evaluate the model's performance on unseen data, ensuring generalisability.
    * **Cross-Validation Strategies:** If used, describes techniques like k-fold cross-validation for more robust model evaluation and hyperparameter tuning.

* **Implementation of Predictive Models**
    * **Regression Models (predicting continuous values):**
        * *Linear Regression:* Demonstrates fitting a linear model to understand relationships and predict a continuous target variable.
        * *Random Forest Regressor:* Illustrates tree-based methods for regression tasks.
   
* **Model Training and Prediction:**
    * **Fitting the Model:** Shows the process of training the chosen model(s) on the training dataset.
    * **Making Predictions:** Illustrates how to use the trained model to make predictions on the test dataset (or new data).

* **Model Evaluation and Interpretation:**
    * **Performance Metrics (Regression):** Details the use of metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared to assess the accuracy of regression models.
    * **Interpretation of Results:** Discusses how to interpret the model's predictions and evaluation metrics in the context of the problem.
    * **Hyperparameter Tuning:** If covered, mentions techniques like GridSearch or RandomizedSearch for optimising model parameters.
 

