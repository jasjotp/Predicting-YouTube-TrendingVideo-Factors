from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import statsmodels.api as sm
from scipy.stats import f_oneway, chi2_contingency
from scipy.stats import mannwhitneyu

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

# Perform Chi-Squared test
def perform_chi_squared(data, group_var, condition_var):
    contingency_table = pd.crosstab(data[group_var], data[condition_var])
    chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)
    print(f'Chi-Squared results for {group_var} and {condition_var}: chi2_statistic = {chi2_stat}, p-value = {p_value}')

# Display histogram to check for Normality
def perform_histogram_tests(data, features):
    for feature in features:
        # Replace infinite values with NaN
        data[feature].replace([np.inf, -np.inf], np.nan, inplace=True)

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

# Get the data
current_dir = os.path.dirname(os.path.abspath(__file__))
old_data = pd.read_csv(os.path.join(current_dir, '..', 'data', 'CAvideos.csv'), parse_dates=['publish_time'])
new_data = pd.read_csv(os.path.join(current_dir, '..', 'data', 'CAvideos_2020-2024.csv'), parse_dates=['publishedAt'])

# Filter data
old_data['year'] = old_data['publish_time'].dt.year
new_data['year'] = new_data['publishedAt'].dt.year
old_data = old_data[old_data['year'] >= 2017]

# Calculate categories
old_data['tag_count'] = old_data['tags'].apply(split_tags).apply(len)
new_data['tag_count'] = new_data['tags'].apply(split_tags).apply(len)

new_data.rename(columns={'view_count': 'views'}, inplace=True)

# Apply transformations to the data to make the data more normal
old_data['log_views'] = np.log(old_data['views'] + 1)
old_data['log_likes'] = np.log(old_data['likes'] + 1)
old_data['log_dislikes'] = np.log(old_data['dislikes'] + 1)
old_data['log_comment_count'] = np.log(old_data['comment_count'] + 1)
new_data['log_views'] = np.log(new_data['views'] + 1)
new_data['log_likes'] = np.log(new_data['likes'] + 1)
new_data['log_dislikes'] = np.log(new_data['dislikes'] + 1)
new_data['log_comment_count'] = np.log(new_data['comment_count'] + 1)

# Combine old_data and new_data
combined_data = pd.concat([old_data, new_data])

# Compare transformed variables for ANOVA and Post-Hoc Analysis
compare_years('log_views')
compare_years('log_likes')
compare_years('log_dislikes')
compare_years('log_comment_count')

# Also compare the raw variables without any transformations
compare_years('views')
compare_years('likes')
compare_years('dislikes')
compare_years('comment_count')
compare_years('tag_count')

new_data.rename(columns={'view_count': 'views'}, inplace=True)
compare_years('views')

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

features = ['views', 'log_views', 'likes', 'log_likes', 'dislikes', 'log_dislikes', 'comment_count', 'log_comment_count']

# plot a histogram of the views, likes, dislikes, and comment counts to check for normality, CLT is satisfied since we have >40 data points
perform_histogram_tests(combined_data, features)

# Use the transformed features for OLS regression
transformed_x_vars = ['log_likes', 'log_dislikes', 'log_comment_count'] + tag_columns

perform_ols_regression(combined_data, 'log_views', transformed_x_vars)

# Perform Chi-Squared test on trending month and category_id
combined_data['trending_month'] = combined_data['publish_time'].dt.month
perform_chi_squared(combined_data, 'trending_month', 'category_id')

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

# Plotting the distribution of the categorical views
plt.figure(figsize=(10, 6))
combined_data['views_category'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Views Categories')
plt.xlabel('Views Category')
plt.ylabel('Frequency')
save_path = os.path.join('..', 'graphs', 'compare_years', 'Distribution of Views Categories.png')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path)
plt.close()

# Fit the OLS model
X = combined_data[transformed_x_vars]
y = combined_data['views']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Views and Likes
median_likes = combined_data['likes'].median()
group1_views = combined_data[combined_data['likes'] >= median_likes]['views']
group2_views = combined_data[combined_data['likes'] < median_likes]['views']
u_stat, p_val = mannwhitneyu(group1_views, group2_views, alternative='two-sided')
print(f'Views and Likes Mann-Whitney U Test: U-stat={u_stat}, p-value={p_val}')

# Tags and Likes ("funny" tag)
combined_data['funny'] = combined_data['tags'].apply(lambda tags: 1 if 'funny' in tags else 0)
group1_likes = combined_data[combined_data['funny'] == 1]['likes'].dropna()
group2_likes = combined_data[combined_data['funny'] == 0]['likes'].dropna()
u_stat, p_val = mannwhitneyu(group1_likes, group2_likes, alternative='two-sided')
print(f'Likes and "funny" Tag Mann-Whitney U Test: U-stat={u_stat}, p-value={p_val}')

# Tags and Views ("video" tag)
combined_data['video'] = combined_data['tags'].apply(lambda tags: 1 if 'video' in tags else 0)
group1_views = combined_data[combined_data['video'] == 1]['views'].dropna()
group2_views = combined_data[combined_data['video'] == 0]['views'].dropna()
u_stat, p_val = mannwhitneyu(group1_views, group2_views, alternative='two-sided')
print(f'Views and "video" Tag Mann-Whitney U Test: U-stat={u_stat}, p-value={p_val}')