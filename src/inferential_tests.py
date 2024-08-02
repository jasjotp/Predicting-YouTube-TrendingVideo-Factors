from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from scipy.stats import f_oneway, chi2_contingency, normaltest, kruskal
from scipy.stats import mannwhitneyu

categories = [1, 2, 15, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30]
category_names = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals", 17: "Sports", 18: "Short Movies", 19: "Travel & Events", 20: "Gaming", 21: "Videoblogging",
    22: "People & Blogs", 23: "Comedy", 24: "Entertainment", 25: "News & Politics", 26: "Howto & Style", 27: "Education", 28: "Science & Technology", 29: "Nonprofits & Activism",
    30: "Movies", 31: "Anime & Animation", 32: "Action & Adventure", 33: "Classics", 34: 'Comedy', 35: "Documentary", 36: "Drama", 37: "Family", 38: "Foreign", 39: "Horror",
    40: "Sci-Fi & Fantasy", 41: "Thriller", 42: "Shorts", 43: "Shows", 44: "Trailers"
}

# Function to convert date to a standard format to further modify for month/week analysis
def convert_dates(date_str):
    date = pd.to_datetime(date_str, format='%y.%d.%m') 
    return date
    
# Get the list of tags
def split_tags(tags):
    tags_list = tags.split('|')
    return [tag.strip().lower() for tag in tags_list]

# Get the number of words in a string
def count_words(text):
    return len((str(text)).split())

# Function to convert numerical variables to categorical, used for help: https://stackoverflow.com/questions/32633977/how-to-create-categorical-variable-based-on-a-numerical-variable
def convert_to_categorical(data, column, bins, labels):
    data[f'{column}_category'] = pd.cut(data[column], bins=bins, labels=labels)
    return data

# Compare a category over the years
def compare_years(title):

    # Create a boxplot by year to check for equal/similar variance before doing ANOVA
    plt.figure(figsize=(12, 6))
    combined_data.boxplot(column=title, by='year', grid=False)
    plt.title(f'Boxplot of {title} Across Years')
    plt.suptitle('')
    plt.xlabel('Year')
    plt.ylabel(title)
    plt.tight_layout()

    # Save the boxplot
    save_path = os.path.join('..', 'graphs', 'boxplots', f'{title}_boxplot.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    # Sort the category by year
    combined_years = combined_data.groupby('year')[title].mean().reset_index()

    # Perform ANOVA to check for statistically significant results before Post-Hoc Analysis
    categories = combined_data['year'].unique()
    values_by_category = [combined_data[combined_data['year'] == category][title] for category in categories]
    f_stat, p_value = f_oneway(*values_by_category) # got help from: https://scipy.github.io/devdocs/reference/generated/scipy.stats.mstats.f_oneway.html
    print(f'ANOVA results for {title} by year: F-statistic = {f_stat}, p-value = {p_value}')

    # Post-Hoc analysis
    if p_value < 0.05:
        tukey = pairwise_tukeyhsd(endog=combined_data[title], groups=combined_data['year'], alpha=0.05)
        print('\n\n==========', title, '==========')
        print(tukey.summary())

    # Plot results
    plt.plot(combined_years['year'], combined_years[title], marker='o', color='b')
    plt.title(f'{title} By Year (check output for post-hoc summary)')
    plt.xlabel('Year')
    plt.ylabel(title)
    plt.tight_layout()

    # Ensure the directory exists
    save_path = os.path.join('..', 'graphs', 'compare_years', f'{title}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# Perform OLS regression, got help from: https://www.geeksforgeeks.org/ordinary-least-squares-ols-using-statsmodels/
def perform_ols_regression(data, y_var, x_vars):
    # Filter to include only those present in the DataFrame
    x_vars = [var for var in x_vars if var in data.columns]
    X = data[x_vars]
    y = data[y_var]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    print(model.summary())

    # Histogram of residuals to check for normality in residuals (assumption of OLS), got help from: https://stackoverflow.com/questions/35417111/python-how-to-evaluate-the-residuals-in-statsmodels
    plt.figure(figsize=(8, 6))
    plt.hist(model.resid, bins=20, histtype='bar', edgecolor='k', alpha=0.7)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.tight_layout()

    # Save the histogram
    save_path = os.path.join('..', 'graphs', 'residuals_histogram', f'{y_var}_residuals_histogram.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# Perform Chi-Squared test
def perform_chi_squared(data, group_var, condition_var):
    contingency_table = pd.crosstab(data[group_var], data[condition_var])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    print(f'Chi-Squared results for {group_var} and {condition_var}: chi2_statistic = {chi2_stat}, p-value = {p_value}')

# Display histogram to check for Normality
def perform_histogram_tests(data, features):
    for feature in features:
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(data[feature].dropna(), bins=30, edgecolor='k', alpha=0.7)
        plt.title(f'Histogram for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')

        save_path = os.path.join('..', 'graphs', 'histograms', f'{feature}.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

# Function to map hour to time interval
def map_hour_to_interval(hour):
    for start, end, label in time_bins:
        if start <= hour < end:
            return label

def plot_bar_chart(data, title, xlabel, ylabel, xticks_labels=None, rotation=45, save_dir='graphs', filename=None):
    plt.figure(figsize=(12, 8))
    barplot = data.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha='right')
    plt.tight_layout()

    # Annotate bars with the counts
    for index, value in enumerate(data):
        barplot.annotate(str(value), xy=(index, value), ha='center', va='bottom')

    # Set custom x-ticks if provided
    if xticks_labels:
        barplot.set_xticklabels(xticks_labels)

    # Set the save path
    save_path = os.path.join(current_dir, '..', save_dir, f"{title.replace(' ', '_').lower()}.png")
    
    # Ensure the directory exists and save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_top_tags_bar_chart(tag_counts):
    plot_bar_chart(
        data=tag_counts,
        title='Top 15 Tags by Count of Trending Videos',
        xlabel='Tags',
        ylabel='Count of Trending Videos'
    )

def plot_category_bar_chart(category_counts):
    plot_bar_chart(
        data=category_counts,
        title='Number of Trending Videos by Category',
        xlabel='Category Name',
        ylabel='Count of Trending Videos'
    )

def plot_trending_month_bar_chart(trending_monthly_counts, month_names):
    plot_bar_chart(
        data=trending_monthly_counts,
        title='Number of Trending Videos by Month',
        xlabel='Month',
        ylabel='Number of Trending Videos',
        xticks_labels=month_names
    )

def plot_trending_time_interval_bar_chart(trending_time_interval_counts):
    plot_bar_chart(
        data=trending_time_interval_counts,
        title='Number of Trending Videos by Time Interval',
        xlabel='Time Interval',
        ylabel='Number of Trending Videos'
    )

def plot_weekend_vs_weekdays_bar_chart(weekend_counts):
    plot_bar_chart(
        data=weekend_counts,
        title='Number of Trending Videos by Weekdays vs Weekends',
        xlabel='Day Type',
        ylabel='Number of Trending Videos'
    )

def plot_trending_weekday_bar_chart(trending_weekday_counts):
    plot_bar_chart(
        data=trending_weekday_counts,
        title='Number of Trending Videos by Day of the Week',
        xlabel='Day of the Week',
        ylabel='Number of Trending Videos'
    )

# Plotting the distribution of views, likes, dislikes, and comment counts
def plot_categorical_distribution(data, column, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    data[column].value_counts().sort_index().plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot to the specified directory
    save_path = os.path.join('..', 'graphs', 'histograms', f'{title}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# Perform Mann-Whitney U Test and print results
def perform_mannwhitneyu_test(group1, group2, variable, group1_label, group2_label):
    p_val = mannwhitneyu(group1, group2, alternative='two-sided').pvalue
    result = f'{variable}: Mann-Whitney U Test between {group1_label} and {group2_label}: p-value = {p_val}\n'
    print(result)

# Define the median calculation function
def calculate_median(group1, group2, group1_label, group2_label):
    median_group1 = group1.median()
    median_group2 = group2.median()
    return {group1_label: median_group1, group2_label: median_group2}

# Get the data
current_dir = os.path.dirname(os.path.abspath(__file__))
old_data = pd.read_csv(os.path.join(current_dir, '..', 'data', 'CAvideos.csv'), parse_dates=['publish_time'])
new_data = pd.read_csv(os.path.join(current_dir, '..', 'data', 'CAvideos_2020-2024.csv'), parse_dates=['publishedAt'])

new_data.rename(columns={'view_count': 'views'}, inplace=True)
new_data.rename(columns={'publishedAt': 'publish_time'}, inplace=True)

# Filter data
old_data['year'] = old_data['publish_time'].dt.year
new_data['year'] = new_data['publish_time'].dt.year
old_data = old_data[old_data['year'] >= 2017]

# Calculate categories
old_data['tag_count'] = old_data['tags'].apply(split_tags).apply(len)
new_data['tag_count'] = new_data['tags'].apply(split_tags).apply(len)

# Apply transformations to the data to make the data more normal
old_data['log_views'] = np.log(old_data['views'] + 1)
old_data['log_likes'] = np.log(old_data['likes'] + 1)
old_data['log_dislikes'] = np.log(old_data['dislikes'] + 1)
old_data['log_comment_count'] = np.log(old_data['comment_count'] + 1)
new_data['log_views'] = np.log(new_data['views'] + 1)
new_data['log_likes'] = np.log(new_data['likes'] + 1)
new_data['log_dislikes'] = np.log(new_data['dislikes'] + 1)
new_data['log_comment_count'] = np.log(new_data['comment_count'] + 1)

# Convert trending date format
old_data['trending_date'] = old_data['trending_date'].apply(convert_dates)
new_data['trending_date'] = new_data['publish_time'].apply(convert_dates)

# Filter data to only include valid trending dates
old_data = old_data.dropna(subset=['trending_date'])
new_data = new_data.dropna(subset=['trending_date'])

# Extract the day of the week from the trending date
old_data['trending_day_of_week'] = old_data['trending_date'].dt.day_name()
new_data['trending_day_of_week'] = new_data['trending_date'].dt.day_name()

# Combine old_data and new_data
combined_data = pd.concat([old_data, new_data])

# # Compare transformed variables for ANOVA and Post-Hoc Analysis
# compare_years('log_views')
# compare_years('log_likes')
# compare_years('log_dislikes')
# compare_years('log_comment_count')

# # Also compare the raw variables without any transformations
# new_data.rename(columns={'view_count': 'views'}, inplace=True)
# compare_years('views')
# compare_years('likes')
# compare_years('dislikes')
# compare_years('comment_count')
# compare_years('tag_count')

combined_data['times_trending'] = combined_data['video_id'].map(combined_data['video_id'].value_counts())
# compare_years('times_trending')

# combined_data['description_length'] = combined_data['description'].apply(count_words)
# compare_years('description_length')

# Perform regression analysis on old_data
combined_data['tags'] = combined_data['tags'].apply(split_tags)
all_tags = combined_data['tags'].explode().value_counts()
top_15_tags = all_tags.head(15).index.tolist()

# # Create dummy variables for the top 15 tags
# for tag in top_15_tags:
#     combined_data[f'tag: {tag}'] = combined_data['tags'].apply(lambda tags: 1 if tag in tags else 0)

# tag_columns = [f'tag: {tag}' for tag in top_15_tags]

# features = ['views', 'log_views', 'likes', 'log_likes', 'dislikes', 'log_dislikes', 'comment_count', 'log_comment_count', 'tag_count', 'times_trending', 'description_length']

# # plot a histogram of the views, likes, dislikes, and comment counts to check for normality, CLT is satisfied since we have >40 data points
# perform_histogram_tests(combined_data, features)

# # Use the transformed features for OLS regression
# transformed_x_vars = ['log_likes', 'log_dislikes', 'log_comment_count'] + tag_columns

# perform_ols_regression(combined_data, 'log_views', transformed_x_vars)

# # Define bins and labels for the numerical variables, used for help: https://stackoverflow.com/questions/32633977/how-to-create-categorical-variable-based-on-a-numerical-variable
# views_bins = [0, 100000, 200000, 300000, 400000, 500000, 750000, 1000000, 1500000, 2000000, 2500000, 5000000, 10000000, 50000000, 100000000, np.inf]
# likes_bins = [0, 1000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, np.inf]
# dislikes_bins = [0, 100, 500, 1000, 5000, 10000, 20000, 50000, np.inf]
# comment_count_bins = [0, 100, 500, 1000, 5000, 10000, 20000, 50000, np.inf]

# views_labels = [
#     '0 - 100K', 
#     '100K - 200K', 
#     '200K - 300K', 
#     '300K - 400K', 
#     '400K - 500K', 
#     '500K - 750K', 
#     '750K - 1M', 
#     '1M - 1.5M', 
#     '1.5M - 2M', 
#     '2M - 2.5M', 
#     '2.5M - 5M', 
#     '5M - 10M', 
#     '10M - 50M', 
#     '50M - 100M', 
#     '100M - infinity'
# ]

# likes_labels = ['0-1K', '1K-5K', '5K-10K', '10K-20K', '20K-50K', '50K-100K', '100K-200K', '200K-500K', '500K+']
# dislikes_labels = ['0-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K-20K', '20K-50K', '50K+']
# comment_count_labels = ['0-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K-20K', '20K-50K', '50K+']

# # Convert numerical variables to categorical for chi-squared test
# combined_data = convert_to_categorical(combined_data, 'views', views_bins, views_labels)
# combined_data = convert_to_categorical(combined_data, 'likes', likes_bins, likes_labels)
# combined_data = convert_to_categorical(combined_data, 'dislikes', dislikes_bins, dislikes_labels)
# combined_data = convert_to_categorical(combined_data, 'comment_count', comment_count_bins, comment_count_labels)

# # Plot histograms of the views categories, likes, dislikes, and comment count categories before performing chi-sqaured test to see if distrubiton is evenly distributed in the current bins
# plot_categorical_distribution(combined_data, 'views_category', 'Distribution of Views Categories', 'Views Category', 'Frequency')
# plot_categorical_distribution(combined_data, 'likes_category', 'Distribution of Likes Categories', 'Likes Category', 'Frequency')
# plot_categorical_distribution(combined_data, 'dislikes_category', 'Distribution of Dislikes Categories', 'Dislikes Category', 'Frequency')
# plot_categorical_distribution(combined_data, 'comment_count_category', 'Distribution of Comment Count Categories', 'Comment Count Category', 'Frequency')

# # Perform Chi-Squared test for views and tags
# for tag in top_15_tags:
#     perform_chi_squared(combined_data, f'tag: {tag}', 'views_category')

# # Perform Chi-Squared test for views and category_id
# perform_chi_squared(combined_data, 'category_id', 'views_category')

# # Perform Chi-Squared test for likes, dislikes, and comment_count categories with views
# perform_chi_squared(combined_data, 'likes_category', 'views_category')
# perform_chi_squared(combined_data, 'dislikes_category', 'views_category')
# perform_chi_squared(combined_data, 'comment_count_category', 'views_category')

# # Perform Chi-Squared test on trending month and category_id
# combined_data['trending_month'] = combined_data['publish_time'].dt.month
# perform_chi_squared(combined_data, 'trending_month', 'views_category')

# # Perform chi squared test on the day of the week and the views categorys
# combined_data['day_of_week'] = combined_data['publish_time'].dt.dayofweek
# perform_chi_squared(combined_data, 'day_of_week', 'views_category')

# # explode and count the the tags
# tag_counts = combined_data['tags'].explode().value_counts()

# # Filter for the top 15 tags
# top_15_tags_counts = tag_counts.head(15)

# # Plot the bar charts for the counts of the top 15 tags
# plot_top_tags_bar_chart(top_15_tags_counts)

# Category names
combined_data['category_name'] = combined_data['category_id'].map(category_names)

# # Get the count of trending videos for each category
# category_counts = combined_data['category_name'].value_counts()

# # Plot the number of trending videos in each category
# plot_category_bar_chart(category_counts)

# # Count the number of trending videos for each month of the year
# trending_monthly_counts = combined_data['trending_month'].value_counts().sort_index()

# month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# # Plot the number of trending videos in each month
# plot_trending_month_bar_chart(trending_monthly_counts, month_names)

# Define time intervals
time_bins = [
    (6, 10, '6AM-10AM'),
    (10, 14, '10AM-2PM'),
    (14, 18, '2PM-6PM'),
    (18, 22, '6PM-10PM'),
    (22, 26, '10PM-2AM'),
    (2, 6, '2AM-6AM')
]

combined_data['publish_hour'] = combined_data['publish_time'].dt.hour
combined_data['publish_time_interval'] = combined_data['publish_hour'].apply(map_hour_to_interval)

# # Count the number of trending videos for each time interval of the day
# trending_time_interval_counts = combined_data['publish_time_interval'].value_counts().reindex(
#     ['6AM-10AM', '10AM-2PM', '2PM-6PM', '6PM-10PM', '10PM-2AM', '2AM-6AM']
# )

# # Plot the number of trending videos in each time interval above
# plot_trending_time_interval_bar_chart(trending_time_interval_counts)

# # Extract day of the week and weekend flag
# combined_data['publish_day'] = combined_data['publish_time'].dt.dayofweek
# combined_data['weekend'] = combined_data['publish_day'].apply(lambda day: 1 if day >= 5 else 0)

# # Count the number of trending videos for weekdays and weekends
# weekend_counts = combined_data['weekend'].value_counts().reindex([0, 1])
# weekend_counts.index = ['Weekdays', 'Weekends']

# # Plot the number of trending videos on weekends vs weekdays
# plot_weekend_vs_weekdays_bar_chart(weekend_counts)

# # Count the number of trending videos for each day of the week
# trending_weekday_counts = combined_data['trending_day_of_week'].value_counts().reindex(
#     ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
# )

# # Plot the number of trending videos on each weekday
# plot_trending_weekday_bar_chart(trending_weekday_counts)

# # Perform Mann-Whitney U Tests
# median_likes = combined_data['likes'].median()

# perform_mannwhitneyu_test(
#     combined_data[combined_data['likes'] >= median_likes]['views'], 
#     combined_data[combined_data['likes'] < median_likes]['views'], 
#     'Views and Likes', 'High Likes', 'Low Likes'
# )

# # Calculate the median views of videos with high likes vs videos with low likes
# median_views_likes = calculate_median(
#     combined_data[combined_data['likes'] >= combined_data['likes'].median()]['views'], 
#     combined_data[combined_data['likes'] < combined_data['likes'].median()]['views'], 
#     'High Likes', 
#     'Low Likes'
# )

# # Create the 'funny' column based on whether 'funny' exists in the tags
# combined_data['funny'] = combined_data['tags'].apply(lambda tags: 1 if 'funny' in tags else 0)

# perform_mannwhitneyu_test(
#     combined_data[combined_data['funny'] == 1]['likes'], 
#     combined_data[combined_data['funny'] == 0]['likes'], 
#     'Likes and "funny" Tag', 'Funny Tag', 'No Funny Tag'
# )

# # Calculate the median likes of videos with the funny tag vs. without the funny tag
# median_likes_funny = calculate_median(
#     combined_data[combined_data['funny'] == 1]['likes'], 
#     combined_data[combined_data['funny'] == 0]['likes'], 
#     'Funny Tag', 'No Funny Tag'
# )

# # Create the 'fortnite' column based on whether 'fortnite' exists in the tags
# combined_data['fortnite'] = combined_data['tags'].apply(lambda tags: 1 if 'fortnite' in tags else 0)

# perform_mannwhitneyu_test(
#     combined_data[combined_data['fortnite'] == 1]['views'], 
#     combined_data[combined_data['fortnite'] == 0]['views'], 
#     'Views and "fortnite" Tag', 'fortnite Tag', 'No fortnite Tag'
# )

# # Calculate the median views of videos with the fortnite tag vs. without the fortnite tag
# median_views_fortnite = calculate_median(
#     combined_data[combined_data['fortnite'] == 1]['views'], 
#     combined_data[combined_data['fortnite'] == 0]['views'], 
#     'fortnite Tag', 'No fortnite Tag'
# )

# # Create the 'vlog' column based on whether 'vlog' exists in the tags
# combined_data['vlog'] = combined_data['tags'].apply(lambda tags: 1 if 'vlog' in tags else 0)

# perform_mannwhitneyu_test(
#     combined_data[combined_data['vlog'] == 1]['vlog'], 
#     combined_data[combined_data['vlog'] == 0]['vlog'], 
#     'Views and "vlog" Tag', 'Vlog Tag', 'No Vlog Tag'
# )

# # Calculate the median views of videos with the vlog tag vs. without the vlog tag
# median_views_vlog = calculate_median(
#     combined_data[combined_data['vlog'] == 1]['views'], 
#     combined_data[combined_data['vlog'] == 0]['views'], 
#     'Vlog Tag', 
#     'No Vlog Tag'
# )

# # Create the 'publish hour' column based on the publish time
# combined_data['publish_hour'] = combined_data['publish_time'].dt.hour
# combined_data['morning'] = combined_data['publish_hour'].apply(lambda hour: 1 if 0 <= hour < 12 else 0)

# perform_mannwhitneyu_test(
#     combined_data[combined_data['morning'] == 1]['views'], 
#     combined_data[combined_data['morning'] == 0]['views'], 
#     'Views and Publish Time', 'Morning', 'Evening/Night'
# )

# # Calculate the median views of videos posted from 12AM to 12PM (morning) vs. 12PM to 12AM (Evening/Night)
# median_views_publish_time = calculate_median(
#     combined_data[combined_data['morning'] == 1]['views'], 
#     combined_data[combined_data['morning'] == 0]['views'], 
#     'Morning', 'Evening/Night'
# )

# perform_mannwhitneyu_test(
#     combined_data[combined_data['weekend'] == 1]['views'], 
#     combined_data[combined_data['weekend'] == 0]['views'], 
#     'Views and Publish Day', 'Weekend', 'Weekdays'
# )

# # Calculate the median views of videos posted on weekdays vs. weekends
# median_views_publish_day = calculate_median(
#     combined_data[combined_data['weekend'] == 1]['views'], 
#     combined_data[combined_data['weekend'] == 0]['views'], 
#     'Weekend', 'Weekdays'
# )

# median_results = {
#     "Views and Likes": median_views_likes,
#     "Likes and 'funny' Tag": median_likes_funny,
#     "Views and 'fortnite' Tag": median_views_fortnite,
#     "Views and 'vlog' Tag": median_views_vlog,
#     "Views and Publish Time": median_views_publish_time,
#     "Views and Publish Day": median_views_publish_day,
# }

# # Print out the results
# for test, medians in median_results.items():
#     print(f'{test} Medians:')
#     for group, median in medians.items():
#         print(f'  {group}: {median}')
#     print()


separated_data = {category: combined_data[combined_data['category_name'] == category] for category in category_names.values()}

# not normal enough, even after transformation
for category, data in separated_data.items():
    values = data['times_trending']
    plt.hist(np.log(values))
    plt.title(category)
    plt.savefig(f'graphs/days_by_cat/{category}.png')
    plt.close()

# Use Kruskal-Willlis H-Test for categories, top 15 tags, time interval, and day of the week
grouped_data = []
for cat in categories:
    grouped_data.append(combined_data[combined_data['category_id'] == cat]['times_trending'])

cat_kruskal = kruskal(*grouped_data).pvalue
print(f"Category vs Time Trending p-value: {cat_kruskal}")

if(cat_kruskal < 0.05):
    # not enough data for category 30
    filtered_data = combined_data[combined_data['category_id'] != 30]
    tukey_data = filtered_data[filtered_data['category_id'].isin(categories)]
    tukey_data = tukey_data[['category_id', 'times_trending']]
    tukey_data['category_name'] = tukey_data['category_id'].map(category_names)

    posthoc = pairwise_tukeyhsd(tukey_data['times_trending'], tukey_data['category_name'], alpha=0.05)
    print(posthoc.summary())
    fig = posthoc.plot_simultaneous(figsize=(10, 5), xlabel="Days Trending", ylabel='Category Name',)
    fig.tight_layout()
    fig.savefig("graphs/cat_by_days_trending.png")

exploded_data = combined_data.explode('tags')
grouped_data = []
for tag in top_15_tags:
    grouped_data.append(exploded_data[exploded_data['tags'] == tag]['times_trending'])

tags_kruskal = kruskal(*grouped_data).pvalue
print(f"Tag vs Time Trending p-value: {tags_kruskal}")

if(tags_kruskal < 0.05):
    posthoc_data = exploded_data[exploded_data['tags'].isin(top_15_tags)]
    posthoc = pairwise_tukeyhsd(posthoc_data['times_trending'], posthoc_data['tags'], alpha=0.05)
    print(posthoc.summary())
    fig = posthoc.plot_simultaneous(figsize=(10, 5), xlabel="Days Trending", ylabel='Tag Name',)
    fig.tight_layout()
    fig.savefig("graphs/tag_by_days_trending.png")

# Extract interval labels from time_bins
interval_labels = [label for _, _, label in time_bins]

grouped_data = []
for _, _, interval in time_bins:
    grouped_data.append(combined_data[combined_data['publish_time_interval'] == interval]['times_trending'])

time_kruskal = kruskal(*grouped_data).pvalue
print(f"Time Interval vs Time Trending p-value: {time_kruskal}")

if(time_kruskal < 0.05):
    posthoc_data = combined_data[combined_data['publish_time_interval'].isin(interval_labels)]
    posthoc = pairwise_tukeyhsd(posthoc_data['times_trending'], posthoc_data['publish_time_interval'], alpha=0.05)
    print(posthoc.summary())
    fig = posthoc.plot_simultaneous(figsize=(10, 5), xlabel="Days Trending", ylabel='Time Interval',)
    fig.tight_layout()
    fig.savefig("graphs/time_by_days_trending.png")

# TODO: posthoc: day of the week, description length? add variables to model: day of the week, time_interval. encode: category_id, year
# if we had more time: more variables in the model to make it more accurate, e.g. tags in combanation with the categories based on the exploratory 
# analysis we did
combined_data['trending_day_of_week']
grouped_data = []
for _, _, interval in time_bins:
    grouped_data.append(combined_data[combined_data['publish_time_interval'] == interval]['times_trending'])

time_kruskal = kruskal(*grouped_data).pvalue
print(f"Time Interval vs Time Trending p-value: {time_kruskal}")

if(time_kruskal < 0.05):
    posthoc_data = combined_data[combined_data['publish_time_interval'].isin(interval_labels)]
    posthoc = pairwise_tukeyhsd(posthoc_data['times_trending'], posthoc_data['publish_time_interval'], alpha=0.05)
    print(posthoc.summary())
    fig = posthoc.plot_simultaneous(figsize=(10, 5), xlabel="Days Trending", ylabel='Time Interval',)
    fig.tight_layout()
    fig.savefig("graphs/time_by_days_trending.png")