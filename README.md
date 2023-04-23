# Machine Learning for Predicting Insurance Premiums

*SCSE, NTU Singapore*  
Year: 22/23, Semester 2   
Course Title: SC1015  
Lab Group: A139  
Team: 4 

Members: 

1) [Kauthar](https://github.com/Kautharr) Ahmed Basharahil
2) Bhupathiraju [Mihir](https://github.com/mvbr23) Varma 
3) [Parth](https://github.com/bparth25) Batra

## Description:
This repository contains all the Jupyter Notebooks, images, video demonstrations, dataset and the sources we have referenced as part of the Mini Project for SC1015: Introduction to Data Science and AI.  
This README briefly highlights what we have accomplished in this project. If you would like a more detailed explanation, please refer to the the Jupyter Notebooks in this repository as they contain in-depth descriptions and finer details which may not be mentioned here.

---

### Table of Contents:
1. [Problem Formulation](#problem-formulation)
2. [Data Preparation and Cleaning](#data)
3. [Exploratory Data Analysis](#eda)
4. [Machine Learning Models](#ml-models)
    1. [Multivariate Linear Regression](#linreg)
    2. [Decision/Classification Tree](#dectree)
    3. [XGBoost (Extreme Gradient Boosting)](#xgboost)
5. [Experiments and Insights](#exp-and-insights)
6. [Conclusion](#conclusion)
7. [References](#references)

---
<a name="problem-formulation"></a>
### 1. [Problem formulation:](#problem)
<a name="problem"></a>

**Our Motivation:** We want to minimise the information imbalance in the insurance industry. It is crucial to accurately price policies based on an individual's unique characteristics as it can be useful in detecting fraudulent claims and assessing a potential customers' risk. By developing predictive models based on this dataset, insurance companies can reduce the likelihood of overcharging or undercharging, save significant amounts of revenue by detecting fraudulent claims and make informed decisions about offering coverage to potential customers and at the appropriate price.

**Problem Definition:** To develop a machine learning model that can, as far as possible, accurately predict appropriate insurance premiums based on a given set of demographic and health-related data.

**Assumption:** "Appropriate" insurance premiums are highly subjective, as it is with medical bills. We assume that the current level of charges given to respective users in the `insurance.csv` dataset has been regulated "appropriately".

---
<a name="data"></a>
### 2. [Data preparation and cleaning:](https://github.com/Kautharr/Insurance_ML/blob/main/Part_2_DataCleaning.ipynb)  

**Our Dataset:** [https://www.kaggle.com/datasets/simranjain17/insurance](https://www.kaggle.com/datasets/simranjain17/insurance)  
The dataset used consists of individuals' insurance-related information. The variables in the dataset are `age`, `sex`, `smoker`, `bmi`, `region` and `children`. `charges` is our response variable, which is the insurance premiums.  

Their datatypes are as follows:

| Variable | Data Type |
|---|---|
| Age | Integer |
| Sex | String (Boolean) |
| Smoker | String (Boolean) |
| BMI | Float |
| Charges | Float |
| Region | String (Categorical) |
| Number of Children | Float |

Compiled on Kaggle, our data was a relatively well-curated one with `7` columns. Some missing cells were filled with a standard value of `0` based on reasonable assumptions.  

To obtain favourable data types, we made the following changes to the dataset:
1. Changed the data type of the `children` column to `int`.
2. Converted and appended the `smoker` and `sex` columns to binary `int` variables: `smoker_bool` and `sex_bool` respectively.
3. Converted and appended the `region` column to a categorical `int` variable `region_cat`.  

The above changes were made so that we can generate a correlation matrix as it requires purely numerical data types. Other EDA methods such as linear regression also handle `int` much better than `string` types. The updated dataframe consists of `10` columns, with the `smoker_bool`, `sex_bool` and `region_cat` columns appended to allow for further analysis below (EDA, correlation, etc.). Non-numeric columns were dropped for the following section.

---
<a name="eda"></a>
### 3. [Exploratory Data Analysis:](https://github.com/Kautharr/Insurance_ML/blob/main/Part_3_EDA.ipynb)
The first step in the exploratory data analysis was to identify the variables that have the most significant impact on the medical insurance charges. To achieve this, we conducted a correlation analysis between the variables. The correlation matrix revealed that `smoker_bool`, `age`, and `bmi` have the highest correlation with charges.
1. Exploring BMI v Charges: A pie chart was generated, with the population breakdown of BMI categories; which was compared to another pie chart that tabulated the breakdown of total charges into BMI categories. To further investigate this relationship, a scatter plot was created to plot charges against BMI. No clear linear trend observed but the comparative pie charts suggest that higher BMI leads to higher charges.
2. Exploring AGE v Charges: After removing outliers, we found a linear trend between the mean age and charges incurred. The p-value from the plot suggested strong evidence against the null hypothesis (of no correlation), supporting the importance of age in predicting charges.
3. Exploring SMOKER v Charges: The box plot tells us that the median charges for non-smokers were significantly lower than those of smokers, by around 300%. There was also less variability in charges for non-smokers. Surprisingly, the median charge for smokers is almost the same as the highest outlier for non-smokers.
4. Further analysis conducted for SMOKER category  on Tableau. We found that splitting the category of clients into smokers and non-smokers can provide further insights when analysed again with AGE and BMI variables.  

*Interactive Dashboard available on [Tableau](https://public.tableau.com/app/profile/kauthar.ahmed/viz/InsuranceDataDashboard/Dashboard1)

---
<a name="ml-models"></a>
### 4. [Machine Learning Models:](https://github.com/Kautharr/Insurance_ML/blob/main/Part_4_ML.ipynb)
We explored: `Linear Regression`, `Decision Tree` and `XGBoost` models based on their ability to handle numeric data, complex interactions and feasible computational requirements. Two commonly used metrics, R-squared and RMSE, were used to evaluate the performance of the models. The goal was to ensure the models fit the data well and make accurate predictions.

Training models: The data-training process involved fitting each of our 3 models on the training set and evaluating its performance on the test set. For this, we used an 80-20 split on the train-test dataset.
During the training process, the hyper-parameters were adjusted to find the optimal values that gave the best performance on the test set. The performance of the model was evaluated on the test set to ensure that it generalises well to unseen data.
Overall, the training process involved finding the best combination of hyper-parameters that minimised the error between the predicted and actual values, as measured by the chosen metrics. 

<a name="linreg"></a>
1. Multivariate linear regression was our first model applied to the insurance dataset for predicting charges. We utilised the `get_dummies()` method to convert categorical variables into numerical variables, enabling our model to generate a numerically-derived equation. We then trained the model further by defining the cost function, which is the Residual Sum of Squares (`RSS`) that needs to be minimised. We implemented a gradient descent algorithm to iteratively adjust the correlation coefficients of our model to minimise the cost function. We observed that the data points in our model lie quite randomly away from the line, which indicates that there is still room for improvement in the performance metric scores of our model.
<a name="dectree"></a>
2. We applied the CART algorithm to develop a decision tree model for predicting insurance charges. The CART algorithm uses a threshold value of an attribute to split the nodes of the decision tree into sub-nodes and searches for the best homogeneity for the sub-nodes using the Gini Index criterion. Hyperparameters such as `max_depth` were adjusted to optimise the model's performance, and `OneHotEncoder` and `ColumnTransformer` were used for data preprocessing. The model produced consistent results comparable to a basic linear regression model.
<a name="xgboost"></a>
3. The XGBoost algorithm was used to train a regression model that predicts insurance charges based on input features. The categorical features were one-hot encoded and used for training. The model was then trained to minimise the mean squared error loss function. Overall, the XGBoost model was the most accurate machine learning prediction model for the insurance prediction problem. As our strongest model, we proceeded to conduct experiments to improve the "baseline" XGBoost. This was done by adding weights to the training samples based on whether a person's smoker status and the charges they incur based on their BMI and age. The XGBoost algorithm's capability to compute additional columns with multipliers from the response variable was also utilised to increase the weight of certain samples in the training process.  
*[Insights](https://github.com/Kautharr/Insurance_ML/blob/main/Video%20Demos/TableauInsights.MOV) demonstration on Tableau available for download*

*The performance of the models were evaluated using `R2` score and `RMSE` metrics. The `R2` score measures how well the model fits the data, while the `RMSE` measures the difference between the predicted and actual values.

---
<a name="exp-and-insights"></a>
### 5. [Experiments and Insights:](https://github.com/Kautharr/Insurance_ML/blob/main/Part_5_FinalModel.ipynb)
We incorporated findings from our EDA to stregthen our model of choice by:
1. Setting the weights for the smoker, bmi, and age variables to be higher for smokers
2. Computing charges using linear regression for `smoker` status against `bmi` and `age` (equations generated from Tableau)
3. Fine-tuning hyper parameters such as: `n_estimators`, `max_depth` and `learning_rate`

After iterating between values for hyper parameters, we obtained a Model Score (`R2`) of 90.8% with an `RMSE` of 3554.67 for this improved version; highest amongst other models generated, by at least 5%. This indicates that our experiments produced a more holistic model which: manages to explain the variance in charges with an exceptional degree, does not overfit the trained dataset and produces predictions on test datasets which has the least deviation (most accurate) from actual `charge` values.

---
<a name="conclusion"></a>
### 6. [Conclusion:](https://github.com/Kautharr/Insurance_ML/blob/main/Part_5_FinalModel.ipynb)
**Based on our experimentations with all the models, some learning points we tookaway were:**
1. Finding out that adjusting hyper parameters can have adverse impacts on overall model quality. Though, solely increasing certain parameters may not be advisable as overfitting usually occurs which jeopardises the model explainability, as indicated by the `R2` score. 
2. Complex algorithms are useful when it comes to dealing with real life scenarios which are highly unpredictable. In cases like insurance `charge` prediction, it is highly dependent on inter-variable conditions, where complex algorithms can be much more insightful and effective than a simple linear regression implementation.

**Review of Problem Definition:**  
In conclusion, we believe that the problem has been addressed successfully. We were able to implement Machine Learning algorithms learnt, as well as incorporate real life exploratory data analysis, to generate an improved model with impressive predictive abilities (by our standards). Our beliefs in the project's success were further affirmed when we set out to test it on **individual data**. 

**Testing on individual data**
1. Trained our final model with 100% of the dataset
2. Converted this model into a callable function, *IMPROVED_predict_charges(age, sex, bmi, children, smoker, region)*
3. Created a widget to obtain the `6` arguments from "User"
4. Entered random data points from the `insurance.csv`
5. Compared the model-predicted `charges` to the actual one in csv
  
This [video](https://github.com/Kautharr/Insurance_ML/blob/main/Video%20Demos/ToolDemo.MOV) demonstrates a simulation of how analysts, or even clients, of insurance companies can interact with a machine learning tool. The model would then generate a prediction of appropriate premium charges by entering User's demographics and health data.  
*"View raw" or "Download" to access content of video*

**Limitations:**
1. There is an assumption that our two metrics are sufficient performance measures of our model. For an even more holistic approach, other indicators such as the Cohen's Kappa can be included to take into account the possibility of random agreements between predicted and actual charges.
2. The dataset is highly sitiuational, and therefore predicted `charges` may not be representative of everyone around the world. New and more applicable datasets might have to be generated for other purposes.
3. Historical datasets may be unrepresentative of future occurence. Specifically in the insurance industry, worldwide pandemic such as Covid-19 may cause the appropriate `charges` to be higher due to increased overall hospital bills. 

---
<a name="references"></a>
### 7. [References:](#ref-list)
<a name="ref-list"></a>
1. https://www.iii.org/publications/a-firm-foundation-how-insurance-supports-the-economy/introduction/insurance-industry-at-a-glance 
2. https://www.kaggle.com/datasets/simranjain17/insurance 
3. https://medium.com/analytics-vidhya/ordinary-least-square-ols-method-for-linear-regression-ef8ca10aadfc 
4. https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/ 
5. https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html 
6. https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d 



