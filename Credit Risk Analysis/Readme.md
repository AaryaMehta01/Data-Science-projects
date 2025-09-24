# Credit Risk Analysis
A comprehensive, end-to-end project evaluating credit risk through statistical and machine learning approaches. This project demonstrates the full data science workflow from exploratory analysis to predictive modeling, using real-world credit datasets.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Data Description](#data-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Modeling & Evaluation](#modeling--evaluation)
- [Key Findings & Insights](#key-findings--insights)
- [Usage Instructions](#usage-instructions)
- [Author & Contact](#author--contact)

## Project Overview

**Objective:**  
Financial organizations need robust methods to predict the likelihood of borrowers defaulting on loans. This project aims to:

- Identify critical patterns and determinants of credit risk
- Build, compare, and evaluate classification models to predict loan default likelihood
- Communicate actionable insights based on model outputs and analysis

## Project Structure

```text
Credit Risk Analysis/
├── Data/ or data/                        # Raw and processed datasets
├── J074_AaryaMehta_CapstoneProject.ipynb # Main Jupyter notebook (data cleaning, EDA, modeling)
├── J074_AaryaMehta.pdf                   # Project report: executive summary, methodology, results
├── Readme.md                             # Project documentation
```

## Data Description

The analysis uses tabular datasets containing anonymized information about credit applicants:

- **Sample features:** Age, Income, Employment Status, Credit Score, Loan Amount, Past Defaults, etc.
- **Target variable:** Loan Default Indicator (binary)

Refer to the Data/ folder for full file listings and descriptions. Datasets are loaded and analyzed in the notebook.

## Exploratory Data Analysis (EDA)

Key steps in the EDA (see notebook for full code and visualizations):

- Outlier and missing value assessment
- Distribution analysis of numerical and categorical variables
- Correlation heatmaps
- Detection of high-risk borrower profiles via grouping and aggregation

Sample code, plots, and summary tables are available in `J074_AaryaMehta_CapstoneProject.ipynb`.

## Feature Engineering

- Transformation and normalization of numerical features (age, income, etc.)
- Encoding of categorical features (employment, housing status, etc.)
- Creation of new indicators based on financial health and history
- Dataset splitting (train/test) for unbiased model evaluation

## Modeling & Evaluation

**Multiple models implemented and rigorously compared:**
- Logistic Regression
- Random Forest Classifier
- Other Algorithms as described in the notebook

**Evaluation Metrics:**
- Confusion matrix
- Accuracy, precision, recall, F1-score
- ROC-AUC

You will find all comparative results, model parameter tuning, and interpretations in both the notebook and the PDF report.

## Key Findings & Insights

- **Top determinants:** Credit history, prior defaults, income level, and certain categorical demographics significantly impact default risk.
- **Best model:** In testing, the Random Forest Classifier achieved the highest balance of precision and recall.
- **Business recommendations:** Early identification of high-risk applicants is possible, and risk-adjusted lending strategies are suggested.

See "Results" and "Discussion" sections in `J074_AaryaMehta.pdf` for specific tables and plots.

## Usage Instructions

1. **Clone or download the repository:**
   ```bash
   git clone https://github.com/amehta7850/Data-Science-projects.git
   ```

2. **Install Python dependencies:**  
   Open the notebook in your Python environment with recommended libraries, e.g.:
   - pandas, numpy, scikit-learn, matplotlib, seaborn

3. **Run the notebook:**
   - Launch Jupyter Notebook or JupyterLab
   - Open `J074_AaryaMehta_CapstoneProject.ipynb`
   - Execute cells sequentially to reproduce the full workflow and results

## Author & Contact

**Author:** Aarya Mehta  
**GitHub:** [amehta7850](https://github.com/amehta7850)
