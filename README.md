# Insurance_ML
Machine Learning for predicting Insurance Premiums

The problem covered in this notebook is quite a common one in Data Science. Suppose we work at an insurance company as analysts; our task is to allocate appropriate premium charges for our clients; given the indicators (and response variable) in our historical dataset -  `insurance.csv`. 

Dataset: https://www.kaggle.com/datasets/simranjain17/insurance


School of Computer Science and Engineering Nanyang Technological University 
1. Lab: 
2. SC Group: 
3. Members:

### Description:
This repository contains all the Jupyter Notebooks, datasets, images, video presentations, and the source materials/references we have used and created as part of the Mini Project for SC1015: Introduction to Data Science and AI.
This README briefly highlights what we have accomplished in this project. If you want a more detailed explanation of things, please refer to the the Jupyter Notebooks in this repository. They contain more in-depth descriptions and smaller details which are not mentioned here in the README.

### Table of Contents:
1. Problem formulation
2. Data preparation and cleaning
3. Exploratory Data Analysis
4. Machine Learning Models

### 1. Problem formulation:
**Our Dataset:** [https://www.kaggle.com/datasets/simranjain17/insurance](https://www.kaggle.com/datasets/simranjain17/insurance)

**Problem Definition:** Develop a machine learning model that can, as far as possible, accurately predict insurance premiums based on a given set of demographic and health-related features, such as age, sex, BMI, number of children, smoker status, and region.

### 2. Data preparation and cleaning:
The initial step was to read the insurance.csv file using pandas and then check for any missing values. Two columns had missing values, it was filled with a standard value of zero(0): df = df.fillna(0).
Data Manipulation: Before we proceed with the exploration of the dataset, it is essential to manipulate the dataset to give us favourable data types. We make the following changes to the dataset.
1. Change the data type of the 'children' column to integer.
2. Convert the 'smoker' column to a binary variable ('smoker_bool').
3. Convert the 'sex' column to a binary variable ('sex_bool').
4. Convert the 'region' column to a categorical variable ('region_cat').
The dataframe now consists of 10 columns, with the 'smoker_bool', 'sex_bool' and 'region_cat' columns appended to allow for further analysis below (EDA, correlation, etc.). The new data types could also be used to predict our response variable (charges, dtype->float).

### 3. ****Exploratory Data Analysis:****
The first step in the exploratory data analysis was to identify the variables that have the most significant impact on the medical insurance charges. To achieve this, we conducted a correlation analysis between the following variables: age, bmi, children, charges, smoker_bool, sex_bool, and region_cat. The correlation matrix revealed that smoker_bool, age, and bmi have the highest correlation with charges.
1. BMI: The next step in the analysis was to investigate the relationship between BMI and charges. A pie chart was created to show the population breakdown of each category. To further investigate this relationship, a scatter plot was created with charges plotted against BMI.
2. AGE: After removing outliers, we found a linear trend between age and charges and strong evidence against the null hypothesis of no correlation, supporting the importance of age in predicting charges.
3. SMOKER: From the box plot, we found that the median charges for non-smokers were significantly lower than those of smokers. Additionally, there was less variability in charges for non-smokers. For smokers, there were several outliers with very high charges, indicating that some smokers incur much higher medical expenses than others.

### 4. Machine Learning Models:
The aim of this project is to build a machine learning model that can accurately predict the insurance charges based on the given features. The dataset contains numerical and categorical data, and the expected output, which is the predicted charges, is also numeric. Therefore, we have selected three regression models, Linear Regression, Random Forest Regression, and Gradient Boosting Regression, to perform this task. These models were chosen based on their ability to handle numeric data and complex interactions between features.
To evaluate the performance of our models, we used two commonly used metrics, R-squared and RMSE. R-squared measures the proportion of variance explained by the model, while RMSE gives an idea about the magnitude of the error.

1. Multivariate linear regression is a mathematical technique that involves solving a system of equations to obtain the values of regression coefficients that minimize the sum of squared residuals. In this project, we used the ‘get_dummies()’ to convert categorical variables into numerical variables. After that, we defined the cost function, which is the RSS that needs to be minimized. We implemented a gradient descent algorithm that iteratively adjusts the correlation coefficients of our model to minimize the cost function. The performance of the linear regression model was not satisfactory, with an R-squared score of 0.8 and an RMSE of 5641.63. The scatter plot of predicted charges against actual charges shows that the data points lie randomly away from the line. Therefore, we must try to reduce this randomness by using more complex algorithms in ML models.

2. The CART algorithm was used to generate the decision tree model. The CART algorithm splits the nodes of the decision tree into sub-nodes based on a threshold value of an attribute. The Gini Index criterion is used to search for the best homogeneity for the sub-nodes. The feature with the lowest Gini coefficient is chosen as the split, and this step is repeated for each child node until the tree is fully grown. Hyperparameters such as ‘maximum depth’ and ‘minimum samples for split’ were experimentally tweaked to avoid over or underfitting. OneHotEncoder and ColumnTransformer were used to convert the data into readable types by the decision tree. We found that the score was rather consistent with that of a basic linear regression model. 

3. In our implementation, we used the XGBoost algorithm to train a regression model that predicts the insurance charges based on the input features. We one-hot encoded the categorical features and used 80% of the dataset for training. The model was trained to minimize the mean squared error loss function, which measures the deviation between the predicted and actual values. Our implementation of the XGBoost model achieved a Model Score of 0.849 and an RMSE of $4829.28, making it the most accurate machine learning prediction model for the insurance prediction problem. The scatter plot of predicted charges against actual charges shows that the data points are more closely clustered around the line than in the previous models. To improve the predictive power of the model, we used the findings gained earlier from exploratory data analysis and Tableau visualization. In particular, we added weights to the training samples based on whether the person is a smoker or not and the charges they incur based on their BMI and age.


