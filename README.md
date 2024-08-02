# Predicting-YouTube-Trending-Video-Factors

Required Libraries (excluding base Python):
* statsmodels
* matplotlib.pyplot
* numpy
* pandas
* scipy.stats
* seaborn
* sklearn

Our two datasets can be found inside the **data** directory. CAvideos.csv contains the YouTube trending data from 2017/2018, and CAvideos_2020-2024 the rest.

The source code for our three python files can be find in **src**.
* main.py: Our initial data discovery... TODO
* inferential_tests.py: Statistical tests... TODO
* predict_models.py: Run this to train/fit a RandomForestClassifier that labels times_trending, year and views. It also predicts how many times a mock video (generated data) will trend based off of its characteristics (Note: this file can take up to five minutes to run). 
