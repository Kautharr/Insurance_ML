# Machine Learning for predicting Insurance Premiums

*NTU SCSE*  
Course: SC1015  
Year: 22/23 Semester 2  
Lab: A139  
Team: 4 

Members: 

1) Kauthar Ahmed Basharahil
2) Bhupathiraju Mihir Varma 
3) Parth Batra

## Description:
This repository contains all the Jupyter Notebooks, dataset, images, video presentations, and the source materials/references we have used and created as part of the Mini Project for SC1015: Introduction to Data Science and AI.
This README briefly highlights what we have accomplished in this project. For a more detailed explanation, you can refer to the the Jupyter Notebooks in this repository. They contain more in-depth descriptions and finer details which are not mentioned here in the README.

---

### Table of Contents:
1. [Problem Formulation](#problem-formulation)
2. [Data Preparation and Cleaning](#data)
3. [Exploratory Data Analysis](#eda)
4. [Machine Learning Models](#ml-models)
5. [Experiments and Insights](#exp-and-insights)
6. [Conclusion]
7. [References]
---
<a name="problem-formulation"></a>
### 1. [Problem formulation:](#problem)
<a name="problem"></a>
**Our Dataset:** [https://www.kaggle.com/datasets/simranjain17/insurance](https://www.kaggle.com/datasets/simranjain17/insurance)

**Problem Definition:** To develop a machine learning model that can, as far as possible, accurately predict insurance premiums based on a given set of demographic and health-related data.

**Motivation:** We want to minimise the information imbalance in the insurance industry. It is crucial to accurately price policies based on an individual's unique characteristics as it can be useful in detecting fraudulent claims and assessing a potential customers' risk. By developing predictive models based on this dataset, insurance companies can reduce the likelihood of overcharging or undercharging, save significant amounts of revenue by detecting fraudulent claims and make informed decisions about offering coverage to potential customers and at the appropriate price.

---
<a name="data"></a>
### 2. [Data preparation and cleaning:](https://github.com/Kautharr/Insurance_ML/blob/main/Part_2_DataCleaning.ipynb)

The dataset used consists of individuals' insurance-related information. The variables in the dataset are `age`, `sex`, `smoker`, `bmi`, `charges`, `region` and `children`. Their datatypes are as follows:

| Variable | Data Type |
|---|---|
| Age | Integer |
| Sex | String (Boolean) |
| Smoker | String(Boolean) |
| BMI | Float |
| Charges | Float |
| Region | String (Categorical) |
| Number of Children | Float |

Compiled on Kaggle, our data was a relatively well-curated one with `7` columns. Some missing cells were filled with a standard value of `0` based on reasonable assumptions.  

To obtain favourable data types, we made the following changes to the dataset:
1. Changed the data type of the `children` column to `int`.
2. Converted and appended the `smoker` and `sex` columns to binary `int` variables: `smoker_bool` and `sex_bool` respectively.
4. Converted and appended the `region` column to a categorical `int` variable `region_cat`.  

The updated dataframe consists of `10` columns, with the `smoker_bool`, `sex_bool` and `region_cat` columns appended to allow for further analysis below (EDA, correlation, etc.). 

---
<a name="eda"></a>
### 3. [Exploratory Data Analysis:](https://github.com/Kautharr/Insurance_ML/blob/main/Part_3_EDA.ipynb)
The first step in the exploratory data analysis was to identify the variables that have the most significant impact on the medical insurance charges. To achieve this, we conducted a correlation analysis between the variables. The correlation matrix revealed that `smoker_bool`, `age`, and `bmi` have the highest correlation with charges.
1. Exploring BMI v Charges: A pie chart was generated, with the population breakdown of BMI categories; which was compared to another pie chart that tabulated the breakdown of total charges into BMI categories. To further investigate this relationship, a scatter plot was created to plot charges against BMI. No clear linear trend observed but the comparative pie charts suggest that higher BMI leads to higher charges.
2. Exploring AGE v Charges: After removing outliers, we found a linear trend between the mean age and charges incurred. The p-value from the plot suggested strong evidence against the null hypothesis (of no correlation), supporting the importance of age in predicting charges.
3. Exploring SMOKER v Charges: The box plot tells us that the median charges for non-smokers were significantly lower than those of smokers, by around 300%. There was also less variability in charges for non-smokers. Surprisingly, the median charge for smokers is almost the same as the highest outlier for non-smokers.
4. Further analysis conducted for SMOKER category  on Tableau. We found that splitting the category of clients into smokers and non-smokers can provide further insights when analysed again with AGE and BMI variables.  

Interactive Dashboard available on [Tableau](https://public.tableau.com/app/profile/kauthar.ahmed/viz/InsuranceDataDashboard/Dashboard1)

---
<a name="ml-models"></a>
### 4. [Machine Learning Models:](https://github.com/Kautharr/Insurance_ML/blob/main/Part_4_ML.ipynb)
We explored: `Linear Regression`, `Decision Tree` and `XGBoost` models based on their ability to handle numeric data, complex interactions and feasible computational requirements. Two commonly used metrics, R-squared and RMSE, were used to evaluate the performance of the models. The goal was to ensure the models fit the data well and make accurate predictions.

1. Multivariate linear regression was our first model applied to the insurance dataset for predicting charges. We utilised the `get_dummies()` method to convert categorical variables into numerical variables, enabling our model to generate a numerically-derived equation. We then trained the model further by defining the cost function, which is the Residual Sum of Squares (`RSS`) that needs to be minimised. We implemented a gradient descent algorithm to iteratively adjust the correlation coefficients of our model to minimise the cost function. We observed that the data points in our model lie quite randomly away from the line, which indicates that there is still room for improvement in the performance metric scores of our model.
2. We applied the CART algorithm to develop a decision tree model for predicting insurance charges. The CART algorithm uses a threshold value of an attribute to split the nodes of the decision tree into sub-nodes and searches for the best homogeneity for the sub-nodes using the Gini Index criterion. Hyperparameters such as `max_depth=4` were adjusted to optimise the model's performance, and `OneHotEncoder` and `ColumnTransformer` were used for data preprocessing. The model produced consistent results comparable to a basic linear regression model.
3. The XGBoost algorithm was used to train a regression model that predicts insurance charges based on input features. The categorical features were one-hot encoded and 80% of the dataset was used for training. The model was then trained to minimise the mean squared error loss function. Overall, the XGBoost model was the most accurate machine learning prediction model for the insurance prediction problem. As our strongest model, we proceeded to conduct experiments to improve the "baseline" XGBoost. This was done by adding weights to the training samples based on whether a person's smoker status and the charges they incur based on their BMI and age. The XGBoost algorithm's capability to compute additional columns with multipliers from the response variable was also utilised to increase the weight of certain samples in the training process. 

---
<a name="exp-and-insights"></a>
### 5. [Experiments and Insights:](https://github.com/Kautharr/Insurance_ML/blob/main/codebase.ipynb)
We incorporated findings from our EDA to stregthen our model of choice by:
1. Setting the weights for the smoker, bmi, and age variables to be higher for smokers
2. Computing charges using linear regression for `smoker` status against `bmi` and `age` (equations generated from Tableau)
3. Fine-tuning hyper parameters such as: `n_estimators`, `max_depth` and `learning_rate`

Obtained a Model Score of 90.8% with an RMSE of 3554.67
