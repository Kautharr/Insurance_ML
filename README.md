# Machine Learning for predicting Insurance Premiums

*SCSE, NTU*  
Course: SC1015  
Year: 22/23 Semester 2
Lab: A139  
Team: 4 

Members: 

1) Kauthar Ahmed Basharahil
2) Bhupathiraju Mihir Varma 
3) Parth Batra

### Description:
This repository contains all the Jupyter Notebooks, datasets, images, video presentations, and the source materials/references we have used and created as part of the Mini Project for SC1015: Introduction to Data Science and AI.
This README briefly highlights what we have accomplished in this project. For a more detailed explanation, you can refer to the the Jupyter Notebooks in this repository. They contain more in-depth descriptions and finer details which are not mentioned here in the README.

### Table of Contents:
1. Problem Formulation
2. Data Preparation and Cleaning
3. Exploratory Data Analysis
4. Machine Learning Models
5. Metrics and Findings
6. Conclusion
7. References

### 1. Problem formulation:
**Our Dataset:** [https://www.kaggle.com/datasets/simranjain17/insurance](https://www.kaggle.com/datasets/simranjain17/insurance)

**Problem Definition:** To develop a machine learning model that can, as far as possible, accurately predict insurance premiums based on a given set of demographic and health-related dataset.



### 2. Data preparation and cleaning:
Obtained from Kaggle, our data was a relatively well-curated one with 7 columns. Some missing cells filled with a standard value of zero(0): df = df.fillna(0) based on reasonable assumption.  
To obtain favourable data types, we made the following changes to the dataset:
1. Changed the data type of the 'children' column to integer.
2. Converted the 'smoker' and 'sex' columns to binary variables: 'smoker_bool' and 'sex_bool' respectively.
4. Converted the 'region' column to a categorical variable 'region_cat'.  

The updated dataframe consists of 10 columns, with the 'smoker_bool', 'sex_bool' and 'region_cat' columns appended to allow for further analysis below (EDA, correlation, etc.). 



### 3. ****Exploratory Data Analysis:****
The first step in the exploratory data analysis was to identify the variables that have the most significant impact on the medical insurance charges. To achieve this, we conducted a correlation analysis between the variables. The correlation matrix revealed that smoker_bool, age, and bmi have the highest correlation with charges.
1. Exploring BMI v Charges: A pie chart was generated, with the population breakdown of BMI categories; which was compared to another pie chart that tabulated the breakdown of total charges into BMI categories. To further investigate this relationship, a scatter plot was created to plot charges against BMI. No clear linear trend observed but the comparative pie charts suggest that higher BMI leads to higher charges.
2. Exploring AGE v Charges: After removing outliers, we found a linear trend between the mean age and charges incurred. The p-value from the plot suggested strong evidence against the null hypothesis (of no correlation), supporting the importance of age in predicting charges.
3. Exploring SMOKER v Charges: The box plot tells us that the median charges for non-smokers were significantly lower than those of smokers, by around 300%. There was also less variability in charges for non-smokers. Surprisingly, the median charge for smokers is almost the same as the highest outlier for non-smokers.
4. Further analysis conducted for SMOKER category  on Tableau. We found that splitting the category of clients into smokers and non-smokers can provide further insights when analysed again with AGE and BMI variables.  

Link to Tableau Dashboard: https://public.tableau.com/app/profile/kauthar.ahmed/viz/InsuranceDataDashboard/Dashboard1



### 4. Machine Learning Models:
The problem at hand is to build a machine learning model that can predict insurance charges based on the given features. Our expected output, which is the predicted charges, is numeric in nature, which streamlines our model selection process. The model should be able to handle complex interactions between the features to make accurate predictions. We have shortlisted three models for this task: Linear Regression, Random Forest Regression, and Gradient Boosting Regression. These models were chosen based on their ability to handle numeric data and complex interactions between features. We also made sure that the models were not overfitting and were computationally feasible. To evaluate the performance of our models, we used two commonly used metrics: R-squared and RMSE value. R-squared measures the proportion of variance explained by the model, while RMSE gives an idea about the magnitude of the error. In this case, we used these metrics because we wanted to ensure that the models could fit the data well and make accurate predictions.

1. Multivariate linear regression is a mathematical technique that involves solving a system of equations to obtain the values of regression coefficients that minimize the sum of squared residuals. In this project, we used the ‘get_dummies()’ to convert categorical variables into numerical variables, allowing our model to generate a numerically-derived equation by assigning appropriate values to all Beta constants and epsilon. Once the unknown coefficients have been found, the model is ready to make predictions. To train the model further, we defined the cost function, which is the RSS that needs to be minimized. We implemented a gradient descent algorithm that iteratively adjusts the correlation coefficients of our model to minimize the cost function. After that, we generated charge predictions from the dataset and compared them with corresponding actual values. Ideally, all the points would lie on the red line; however, this is almost never the case in real life, especially with something as unpredictable as insurance charges. As such, we observed that the data points lie quite randomly away from the line. In order to improve the performance metric scores of our model, we must try to reduce this randomness by using more complex algorithms in ML models.

2. The CART algorithm was used to generate the decision tree model. The CART algorithm splits the nodes of the decision tree into sub-nodes based on a threshold value of an attribute. The Gini Index criterion is used to search for the best homogeneity for the sub-nodes. It calculates the weighted sum of the Gini impurities using the Gini(feature) formula, where the training input variables and response variable are passed through. The feature with the lowest Gini coefficient is chosen as the split, and this step is repeated for each child node until the tree is fully grown. The pre-processing and modelling steps were combined into a pipeline for ease of training the model. Hyperparameters such as ‘maximum depth’ and ‘minimum samples for split’ were experimentally tweaked to avoid over or underfitting. OneHotEncoder and ColumnTransformer were used to convert the data into readable types by the decision tree. The model outputs a charge prediction based on the path followed to a specific child node, based on the feature values. We found that the score was rather consistent with that of a basic linear regression model. 

3. In our implementation, we used the XGBoost algorithm to train a regression model that predicts the insurance charges based on the input features. To prepare the data for the model, we one-hot encoded the categorical features and used 80% of the dataset for training. The model was trained to minimize the mean squared error loss function, which measures the deviation between the predicted and actual values. One of the advantages of XGBoost is that it can build an ensemble of decision trees based on random feature subsets, which reduces overfitting and improves the model's generalization ability. The training process is streamlined by the iterative tree processing algorithm, which assigns appropriate weights to each tree. 

    Our implementation of the XGBoost model achieved a Model Score of 84.9%, making it the most accurate machine learning prediction model for the insurance prediction problem. The Root Mean Square (RMS) value for the model was $ 4829.28, indicating that the average deviation of the predictions from the actual values was relatively small. To improve the predictive power of the model, we used the findings gained earlier from exploratory data analysis and Tableau visualization. In particular, we added weights to the training samples based on whether the person is a smoker or not and the charges they incur based on their BMI and age. Smokers were given higher weights to account for the fact that they are more likely to incur higher charges than non-smokers. We also adjusted the weights based on charges for BMI and age, assuming that people with higher charges in these categories are more likely to incur higher overall charges. In addition, we used XGBoost's capability to add additional columns to the input data with multipliers from the response variable to increase the weight of certain samples in the training process. We added two additional columns with multipliers from the charges variable for the smoker, BMI, and age variables. This was done to increase the weight of samples from smokers with higher BMI and age, since they tend to have higher charges. By doing so, the model paid more attention to these samples during training and learned to better predict their charges.
