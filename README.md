# Median House Value Prediction using Various Regression Models

This Jupyter notebook contains code to predict the median house value in California using various regression models. The models evaluated in this notebook include Ridge Regression, Decision Tree Regression, Random Forest Regression, and Gradient Boosting Regression.

## Description

The notebook begins by importing necessary libraries such as Pandas, NumPy, Seaborn, and scikit-learn. The California housing dataset is loaded using scikit-learn's fetch_california_housing function. The dataset is then examined for any missing values and descriptive statistics are calculated.

Exploratory data analysis is performed to visualize the distribution of the median house values and the correlation between different features using Seaborn's plotting functions.

The dataset is split into training and testing sets using scikit-learn's train_test_split function. Four regression models are then trained on the training data and evaluated using the testing data. 

The models include:

Ridge Regression
Decision Tree Regression
Random Forest Regression
Gradient Boosting Regression
Evaluation metrics such as R-squared score and Mean Absolute Error (MAE) are calculated for each model to assess their performance in predicting the median house values.

## Usage

To run the code:

Make sure you have the required libraries installed, including Pandas, NumPy, Seaborn, and scikit-learn.

Run the code cells in sequential order.

Examine the output which includes R-squared scores and MAE for each regression model.

## Dataset

The dataset used in this notebook is the California housing dataset, which contains information about housing attributes, such as median income, house age, average rooms, average bedrooms, population, and average occupancy, and the corresponding median house values.
