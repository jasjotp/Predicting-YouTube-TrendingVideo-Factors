# Predicting-YouTube-Trending-Video-Factors

## Project Overview

This report looks at YouTube trending videos in Canada from 2017-2018 and 2020-2024. The goal is to find out what makes a video trend and how this has changed over the years. We used different statistical tests and visualizations to explore the relationship between a video's features, like tags, categories, and engagement, and its likelihood to trend. We also developed a machine learning model to predict key factors like when a video will trend, how many times it will trend, and how many views it might get. The main question we aim to answer is: What makes a video trend on YouTube in Canada, and how have these factors evolved over time?

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

* main.py: This script completes exploratory data analysis for the YouTube trending video data. It loads the data, cleans/filters it, and performs initial analysis to uncover patterns and trends. It also generates various visualizations, like bar charts and heatmaps, to help us see the relationships between different video attributes, such as tags, categories, and the number of views or likes.
* inferential_tests.py: This script runs statistical tests like ANOVA, Chi-Squared tests, and Mann-Whitney U tests to find significant differences or relationships between video characteristics and their likelihood of trending. It also includes regression analysis to predict how different factors might influence a video’s success on the trending list. Last, it also uses the Kruskal-Wallis H Test to determine if factors such as the category type affected the number of days a video would trend. The results help us understand which attributes are most important for making a video trend
* predict_models.py: Run this to train/fit a RandomForestClassifier that labels times_trending, year and views. It also predicts how many times a mock video (generated data) will trend based off of its characteristics (Note: this file can take up to five minutes to run). 

## Data Directory

```bash
Predicting-YouTube-Trending-Video-Factors/
│
├── data/                       # Dataset folder, containing YouTube video data from 2017-2024
│   ├── CAvideos.csv            # Data from 2017-2018
│   ├── CAvideos_2020-2024.csv  # Data from 2020-2024
│   ├── .gitattributes          # Tells Git that the above data files should be handled by Git LFS
│
├── src/                        # Source code directory
│   ├── main.py                 # Main script for preprocessing and visualization
│   └── inferential_stats.py    # Script for performing inferential statistical analysis
│   ├── predict_models.py       # Script for performing model predictions on data
│
├── graphs/                # Output directory for graphs
│   ├── characteristics/   # Graphs of characteristics over time
│   ├── compare_years/     # Line plots of characteristics by year
│   ├── tags_by_year/      # Top tags by year
│   ├── tags_by_cat/       # Top tags by category
│   ├── days_by_cat/       # Number of days videos trend in each category
│   ├── heatmaps/          # Correlation heatmaps
│   ├── boxplots/          # Boxplots comparing variables across years
│   ├── histograms/        # Histograms for distribution checks
│   ├── residuals_histogram/  # Histogram for residual normality check for views 
│   └──  number_of_trending_videos_by_category.png # Counts of trending videos by category
│   └──  number_of_trending_videos_by_day_of_the_week.png # Counts of trending videos by weekday
│   └──  number_of_trending_videos_by_month.png # Counts of trending videos by month
│   └──  number_of_trending_videos_by_time_interval.png # Counts of trending videos by time posted
│   └──  number_of_trending_videos_by_weekdays_vs_weekends.png # Weekday vs. weekend video count 
│   └──  tag_by_days_trending.png # Comparison between tags/days trending
│   └──  tags_2017_2018.png # Comparison of top tags in 2017 vs. 2018
│   └──  top_15_tags_by_count_of_trending_videos.png # Top 15 tags by count of trending videos
│   └──  Further Exploratory Data Analysis Graphs
```

## Running the Code/Analysis

1. **Ensure that Git LFS is installed on your machine Or Download Datasets from Drive**

This repository uses Git LFS to manage large files. Please ensure you have [Git LFS installed](https://git-lfs.github.com/) before cloning the repository:

```bash
git lfs install
```
To verify that Git LFS is installed on your machine, check the version: 

```bash
git lfs vesion
```

Or:

Download the datasets from Google Drive: [here](https://drive.google.com/drive/folders/1Lwf9Y3GvF0Xiq-SA-dw0bzKiv0kTotgs) and move them to the **data/** directory (refer to above data directory) after cloning the repo in step 2.

2. **Clone the repository**
```bash
git clone https://github.sfu.ca/yourUsername/Predicting-YouTube-Trending-Video-Factors.git
cd Predicting-YouTube-Trending-Video-Factors
```

Ensure that **CAvideos.csv** and **CAvideos_2020-2024.csv** are in the **data/** directory. 
    
3. **Run the preprocessing and exploratory data analysis/visualization file (main.py)** 
    
    main.py processes/cleans the data, creates visualizations and outputs various informative graphs about     significant attributes into the **graphs/** directory.
```bash
python src/main.py
```
4. **Run the inferntial statistics analysis file (inferential_stats.py)**
    
    inferential_stats.py performs statistical tests to further analyze the relationship between certain         attributes and trending videos, and generates related visualizations. 
```bash
python src/inferential_stats.py
```

5. **Run the machine learning file (predict_models.py)**

    predict_models.py filters and combines data form both csv files, then trains a Random Forest Classifier, outputting accuracy score and predictions of generated data (see the console output).
```bash
python src/predict_models.py
```
