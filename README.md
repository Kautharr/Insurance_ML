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
The problem at hand is to build a machine learning model that can predict insurance charges based on the given features. Our expected output, which is the predicted charges, is numeric in nature, which streamlines our model selection process. The model should be able to handle complex interactions between the features to make accurate predictions. We have shortlisted three models for this task: Linear Regression, Random Forest Regression, and Gradient Boosting Regression. These models were chosen based on their ability to handle numeric data and complex interactions between features. We also made sure that the models were not overfitting and were computationally feasible. To evaluate the performance of our models, we used two commonly used metrics: R-squared and RMSE value. R-squared measures the proportion of variance explained by the model, while RMSE gives an idea about the magnitude of the error. In this case, we used these metrics because we wanted to ensure that the models could fit the data well and make accurate predictions.

1. Multivariate linear regression is a mathematical technique that involves solving a system of equations to obtain the values of regression coefficients that minimize the sum of squared residuals. In this project, we used the ‘get_dummies()’ to convert categorical variables into numerical variables, allowing our model to generate a numerically-derived equation by assigning appropriate values to all Beta constants and epsilon. Once the unknown coefficients have been found, the model is ready to make predictions. To train the model further, we defined the cost function, which is the RSS that needs to be minimized. We implemented a gradient descent algorithm that iteratively adjusts the correlation coefficients of our model to minimize the cost function. After that, we generated charge predictions from the dataset and compared them with corresponding actual values. Ideally, all the points would lie on the red line; however, this is almost never the case in real life, especially with something as unpredictable as insurance charges. As such, we observed that the data points lie quite randomly away from the line. In order to improve the performance metric scores of our model, we must try to reduce this randomness by using more complex algorithms in ML models.

2. The CART algorithm was used to generate the decision tree model. The CART algorithm splits the nodes of the decision tree into sub-nodes based on a threshold value of an attribute. The Gini Index criterion is used to search for the best homogeneity for the sub-nodes. It calculates the weighted sum of the Gini impurities using the Gini(feature) formula, where the training input variables and response variable are passed through. The feature with the lowest Gini coefficient is chosen as the split, and this step is repeated for each child node until the tree is fully grown. The pre-processing and modelling steps were combined into a pipeline for ease of training the model. Hyperparameters such as ‘maximum depth’ and ‘minimum samples for split’ were experimentally tweaked to avoid over or underfitting. OneHotEncoder and ColumnTransformer were used to convert the data into readable types by the decision tree. The model outputs a charge prediction based on the path followed to a specific child node, based on the feature values. We found that the score was rather consistent with that of a basic linear regression model. 

3. In our implementation, we used the XGBoost algorithm to train a regression model that predicts the insurance charges based on the input features. To prepare the data for the model, we one-hot encoded the categorical features and used 80% of the dataset for training. The model was trained to minimize the mean squared error loss function, which measures the deviation between the predicted and actual values. One of the advantages of XGBoost is that it can build an ensemble of decision trees based on random feature subsets, which reduces overfitting and improves the model's generalization ability. The training process is streamlined by the iterative tree processing algorithm, which assigns appropriate weights to each tree. 

    Our implementation of the XGBoost model achieved a Model Score of 84.9%, making it the most accurate machine learning prediction model for the insurance prediction problem. The Root Mean Square (RMS) value for the model was $ 4829.28, indicating that the average deviation of the predictions from the actual values was relatively small. To improve the predictive power of the model, we used the findings gained earlier from exploratory data analysis and Tableau visualization. In particular, we added weights to the training samples based on whether the person is a smoker or not and the charges they incur based on their BMI and age. Smokers were given higher weights to account for the fact that they are more likely to incur higher charges than non-smokers. We also adjusted the weights based on charges for BMI and age, assuming that people with higher charges in these categories are more likely to incur higher overall charges. In addition, we used XGBoost's capability to add additional columns to the input data with multipliers from the response variable to increase the weight of certain samples in the training process. We added two additional columns with multipliers from the charges variable for the smoker, BMI, and age variables. This was done to increase the weight of samples from smokers with higher BMI and age, since they tend to have higher charges. By doing so, the model paid more attention to these samples during training and learned to better predict their charges.
