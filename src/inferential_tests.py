from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from scipy.stats import f_oneway, chi2_contingency
from scipy.stats import mannwhitneyu

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
    chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)
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

# Compare transformed variables for ANOVA and Post-Hoc Analysis
compare_years('log_views')
compare_years('log_likes')
compare_years('log_dislikes')
compare_years('log_comment_count')

# Also compare the raw variables without any transformations
new_data.rename(columns={'view_count': 'views'}, inplace=True)
compare_years('views')
compare_years('likes')
compare_years('dislikes')
compare_years('comment_count')
compare_years('tag_count')

combined_data['times_trending'] = combined_data['video_id'].map(combined_data['video_id'].value_counts())
compare_years('times_trending')

combined_data['description_length'] = combined_data['description'].apply(count_words)
compare_years('description_length')

# Perform regression analysis on old_data
combined_data['tags'] = combined_data['tags'].apply(split_tags)
all_tags = combined_data['tags'].explode().value_counts()
top_15_tags = all_tags.head(15).index.tolist()

# Create dummy variables for the top 15 tags
for tag in top_15_tags:
    combined_data[f'tag: {tag}'] = combined_data['tags'].apply(lambda tags: 1 if tag in tags else 0)

tag_columns = [f'tag: {tag}' for tag in top_15_tags]

features = ['views', 'log_views', 'likes', 'log_likes', 'dislikes', 'log_dislikes', 'comment_count', 'log_comment_count', 'tag_count', 'times_trending', 'description_length']

# plot a histogram of the views, likes, dislikes, and comment counts to check for normality, CLT is satisfied since we have >40 data points
perform_histogram_tests(combined_data, features)

# Use the transformed features for OLS regression
transformed_x_vars = ['log_likes', 'log_dislikes', 'log_comment_count'] + tag_columns

perform_ols_regression(combined_data, 'log_views', transformed_x_vars)

# Define bins and labels for the numerical variables, used for help: https://stackoverflow.com/questions/32633977/how-to-create-categorical-variable-based-on-a-numerical-variable
views_bins = [0, 10000, 100000, 1000000, 10000000, np.inf]
likes_bins = [0, 1000, 10000, 100000, np.inf]
dislikes_bins = [0, 1000, 10000, 100000, np.inf]
comment_count_bins = [0, 100, 1000, 10000, np.inf]

views_labels = ['0 - 10000', '10000 - 100000', '100000 - 1000000', '1000000 - 10000000', '10000000 - infinity']
likes_labels = ['0 - 1000', '1000 - 10000', '10000 - 100000', '100000 - infinity']
dislikes_labels = ['0 - 1000', '1000 - 10000', '10000 - 100000', '100000 - infinity']
comment_count_labels = ['0 - 100', '100 - 1000', '1000 - 10000', '10000 - infinity']

# Convert numerical variables to categorical for chi-squared test
combined_data = convert_to_categorical(combined_data, 'views', views_bins, views_labels)
combined_data = convert_to_categorical(combined_data, 'likes', likes_bins, likes_labels)
combined_data = convert_to_categorical(combined_data, 'dislikes', dislikes_bins, dislikes_labels)
combined_data = convert_to_categorical(combined_data, 'comment_count', comment_count_bins, comment_count_labels)

# Perform Chi-Squared test for views and tags
for tag in top_15_tags:
    perform_chi_squared(combined_data, f'tag: {tag}', 'views_category')

# Perform Chi-Squared test for views and category_id
perform_chi_squared(combined_data, 'category_id', 'views_category')

# Perform Chi-Squared test for likes, dislikes, and comment_count categories with views
perform_chi_squared(combined_data, 'likes_category', 'views_category')
perform_chi_squared(combined_data, 'dislikes_category', 'views_category')
perform_chi_squared(combined_data, 'comment_count_category', 'views_category')

# Perform Chi-Squared test on trending month and category_id
combined_data['trending_month'] = combined_data['publish_time'].dt.month
perform_chi_squared(combined_data, 'trending_month', 'views_category')

# Perform chi squared test on the day of the week and the views categorys
combined_data['day_of_week'] = combined_data['publish_time'].dt.dayofweek
perform_chi_squared(combined_data, 'day_of_week', 'views_category')

# explode and count the the tags
tag_counts = combined_data['tags'].explode().value_counts()

# Filter for the top 15 tags
top_15_tags_counts = tag_counts.head(15)

# Plot the results of the counts of the top 15 tags that are in trending videos
plt.figure(figsize=(12, 8))
barplot = top_15_tags_counts.plot(kind='bar', color='skyblue')
plt.title('Top 15 Tags by Count of Trending Videos')
plt.xlabel('Tags')
plt.ylabel('Count of Trending Videos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Annotate bars with the counts for each tag
for index, value in enumerate(top_15_tags_counts):
    barplot.annotate(str(value), xy=(index, value), ha='center', va='bottom')

# Save the plot
save_path = os.path.join(current_dir, '..', 'graphs', 'top_15_tags_trending_videos.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.close()

# Get the count of trending videos for each category
category_counts = combined_data['category_id'].value_counts()

# Plot the results
plt.figure(figsize=(12, 8))
barplot = category_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Trending Videos by Category')
plt.xlabel('Category ID')
plt.ylabel('Count of Trending Videos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Annotate bars with the counts for each category
for index, value in enumerate(category_counts):
    barplot.annotate(str(value), xy=(index, value), ha='center', va='bottom')

# Save the plot
save_path = os.path.join(current_dir, '..', 'graphs', 'trending_videos_by_category.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.close()

# Count the number of trending videos for each month of the year
trending_monthly_counts = combined_data['trending_month'].value_counts().sort_index()

month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Plot the results of the counts of trending videos by month
plt.figure(figsize=(10, 6))
monthly_barplot = trending_monthly_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Trending Videos by Month')
plt.xlabel('Month')
plt.ylabel('Number of Trending Videos')
monthly_barplot.set_xticklabels(month_names, rotation=45)
plt.tight_layout()

# Annotate bars with the counts for each month
for index, value in enumerate(trending_monthly_counts):
    monthly_barplot.annotate(str(value), xy=(index, value), ha='center', va='bottom')

# Save the plot
save_path = os.path.join(current_dir, '..', 'graphs', 'trending_by_month.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.close()

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

# Count the number of trending videos for each time interval of the day
trending_time_interval_counts = combined_data['publish_time_interval'].value_counts().reindex(
    ['6AM-10AM', '10AM-2PM', '2PM-6PM', '6PM-10PM', '10PM-2AM', '2AM-6AM']
)

# Plot the results of the counts of trending videos by time interval
plt.figure(figsize=(10, 6))
time_interval_barplot = trending_time_interval_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Trending Videos by Time Interval')
plt.xlabel('Time Interval')
plt.ylabel('Number of Trending Videos')
plt.xticks(rotation=45)
plt.tight_layout()
    
# Annotate bars with the counts for each time interval
for index, value in enumerate(trending_time_interval_counts):
    time_interval_barplot.annotate(str(value), xy=(index, value), ha='center', va='bottom')

# Save the plot
save_path = os.path.join(current_dir, '..', 'graphs', 'trending_by_time_interval.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.close()

# Extract day of the week and weekend flag
combined_data['publish_day'] = combined_data['publish_time'].dt.dayofweek
combined_data['weekend'] = combined_data['publish_day'].apply(lambda day: 1 if day >= 5 else 0)

# Count the number of trending videos for weekdays and weekends
weekend_counts = combined_data['weekend'].value_counts().reindex([0, 1])
weekend_counts.index = ['Weekdays', 'Weekends']

# Plot the results of the counts of trending videos by weekdays vs weekends
plt.figure(figsize=(10, 6))
weekend_barplot = weekend_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Trending Videos by Weekdays vs Weekends')
plt.xlabel('Day Type')
plt.ylabel('Number of Trending Videos')
plt.tight_layout()

# Annotate bars with the counts
for index, value in enumerate(weekend_counts):
    plt.text(index, value + 5, str(value), ha='center')

# Save the plot
save_path = os.path.join(current_dir, '..', 'graphs', 'Trending videos on weekends vs. weekdays.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)

# Count the number of trending videos for each day of the week
trending_weekday_counts = combined_data['trending_day_of_week'].value_counts().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

# Plot the results of the counts of trending videos by day
plt.figure(figsize=(10, 6))
barplot = trending_weekday_counts.plot(kind='bar', color='skyblue')
plt.title('Number of Trending Videos by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Trending Videos')
plt.xticks(rotation=45)
plt.tight_layout()

# Annotate bars with the counts for each day of the week
for index, value in enumerate(trending_weekday_counts):
    barplot.text(index, value + 5, str(value), ha='center') # got help from: https://www.geeksforgeeks.org/adding-value-labels-on-a-matplotlib-bar-chart/

# Save the plot
save_path = os.path.join(current_dir, '..', 'graphs', 'trending_by_day_of_week.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
#plt.show()

# Plotting the distribution of the categorical views
plt.figure(figsize=(10, 6))
combined_data['views_category'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Views Categories')
plt.xlabel('Views Category')
plt.xticks(rotation=0, ha='right')
plt.ylabel('Frequency')
save_path = os.path.join('..', 'graphs', 'Distribution of Views Categories.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.close()

# Views and Likes
median_likes = combined_data['likes'].median()
group1_views = combined_data[combined_data['likes'] >= median_likes]['views']
group2_views = combined_data[combined_data['likes'] < median_likes]['views']
u_stat, p_val = mannwhitneyu(group1_views, group2_views, alternative='two-sided')
print(f'Views and Likes Mann-Whitney U Test: p-value={p_val}')

# Tags and Likes ("funny" tag)
combined_data['funny'] = combined_data['tags'].apply(lambda tags: 1 if 'funny' in tags else 0)
group1_likes = combined_data[combined_data['funny'] == 1]['likes']
group2_likes = combined_data[combined_data['funny'] == 0]['likes']
u_stat, p_val = mannwhitneyu(group1_likes, group2_likes, alternative='two-sided')
print(f'Likes and "funny" Tag Mann-Whitney U Test: p-value={p_val}')

# Tags and Views ("video" tag)
combined_data['video'] = combined_data['tags'].apply(lambda tags: 1 if 'video' in tags else 0)
group1_views = combined_data[combined_data['video'] == 1]['views']
group2_views = combined_data[combined_data['video'] == 0]['views']
u_stat, p_val = mannwhitneyu(group1_views, group2_views, alternative='two-sided')
print(f'Views and "video" Tag Mann-Whitney U Test: p-value={p_val}')

# Views and Publish Time (Morning vs Evening)
combined_data['publish_hour'] = combined_data['publish_time'].dt.hour
combined_data['morning'] = combined_data['publish_hour'].apply(lambda hour: 1 if 0 <= hour < 12 else 0)
group1_views = combined_data[combined_data['morning'] == 1]['views']  # Morning (12AM to 12PM)
group2_views = combined_data[combined_data['morning'] == 0]['views']  # Evening/Night (12PM to 12AM)
u_stat, p_val = mannwhitneyu(group1_views, group2_views, alternative='two-sided')
print(f'Views and Publish Time (Morning vs Evening) Mann-Whitney U Test: p-value={p_val}')

# Views and Publish Day (Weekdays vs Weekends)
group1_views = combined_data[combined_data['weekend'] == 1]['views']  # Weekend
group2_views = combined_data[combined_data['weekend'] == 0]['views']  # Weekdays
u_stat, p_val = mannwhitneyu(group1_views, group2_views, alternative='two-sided')
print(f'Views and Publish Day (Weekdays vs Weekends) Mann-Whitney U Test: p-value={p_val}')