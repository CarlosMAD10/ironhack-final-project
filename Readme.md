# Ironhack final project

# Project description

This project explores the possibility of extracting trend information from historical index data, and using it to predict the price of stocks.

It has several parts:
- Jupyter notebook explaining the modeling flow.
- Jupyter notebook used to run the program and predict stock prices.
- Data: Kaggle dataset with information from 14 global indexes for over 30 years.
- Python scripts with helper functions and the program flow.

The requirements to run the program are:
- API key to connect to Alphavantage, a website that offers stock price information for free.
- Pycaret, a Python package that automatizes many machine learning tasks. In order to install it, an environment running in Python 3.7 should be created.

# Machine Learning Model

The program uses the index data to train a model that is then used to predict the expected percentage change between a stock current value and its value a certain number of days in the future (projection step). The index data is tranformed into a training dataset by selecting random dates (pivot dates) and extracting snippets, that is, price and value information for several days before (snippet size). The target value of the model is the percentage change. The price and volume information are normalised with a minmax function, which allows to extrapolate the trend information to all kinds of stocks and indexes.

The model is divided in three parts:
1. Clustering with KMeans, which are used to cluster the trends around 10 main trends.
2. Classifying model: with Random Forest Classifier or Extra Trees Classifier. It assigns a category (good, bad or neutral outlook) to each row.
3. Regression model: using all the information (price and volume, cluster, and classification category) in a Ligh Gradient Boosting Machine model, or an Extra Trees Regressor, to predict the percentage increase or decrease between the current stock or index price, and the future one (the projection step).



