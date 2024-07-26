import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import os
import statsmodels.api as sm
from scipy.stats import f_oneway
from scipy.stats import chi2_contingency

def plot_characteristic(name, column):
    plt.plot(data_2017_2018['trending_date'], column)
    plt.title(name)
    plt.xlabel('Trending Date')
    plt.ylabel('Amount')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()
    plt.clf()


def date_to_month(d):
    # You may need to modify this function, depending on your data_2017_2018 types.
    return '%04i-%02i' % (d.year, d.month)

# Function to split tags 
def split_tags(tags):
    tags_list = tags.split('|')
    return [tag.strip().lower() for tag in tags_list]

# Fuction to create a barplot with the top 15 tags for each year, got help from: https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
def plot_top15_tags(tag_data, year):
    tags_df = pd.DataFrame({'tag': tag_data.index, 'count': tag_data.values})
    plt.figure(figsize=(10, 8))
    sns.barplot(x='count', y='tag', data=tags_df, hue=tag_data.index, palette='viridis')
    plt.title(f'Top 15 Most Common Tags in {year}')
    plt.xlabel('Occurrences')
    plt.ylabel('Tags')
    plt.tight_layout()
    plt.show()

# Function to seperate tags by year, explode them to retrive each tag, and count their values. got help from: got help from: https://www.datacamp.com/tutorial/pandas-explode
def seperate_explode_count(data, published_column, tags_column, year):
    data_year = data[data[published_column].dt.year == int(year)].copy()  # Separate data by year
    exploded_tags = data_year.explode(tags_column)  # Explode the tags so each tag gets its own row for counting purposes
    tag_counts = exploded_tags[tags_column].value_counts()  # Count each occurrence of each tag

    return tag_counts

# Function to one-hot encode tags: 1 if video has a top 15 tag, 0 otherwise, got help from: https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
def one_hot_encode_tags(data, tags):
    for tag in tags:
        data.loc[:, f'tag: {tag}'] = data['tags'].apply(lambda tags: 1 if tag in tags else 0)
    return data

# Fetch the data_2017_2018 from the csv file
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path_2017_2018 = os.path.join(current_dir, '..', 'data', 'CAvideos.csv')
csv_path_2020_2024 = os.path.join(current_dir, '..', 'data', 'CAvideos_2020-2024.csv')

data_2017_2018 = pd.read_csv(csv_path_2017_2018, parse_dates=['publish_time'])
data_2020_2024 = pd.read_csv(csv_path_2020_2024, parse_dates=['publishedAt'])

# Function to plot heatmaps for each year
def plot_heatmap(data, features, year):
    plt.figure(figsize=(18, 16))
    correlation_matrix = data[features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title(f'Correlation Matrix of Trending Video Metrics and Top 15 Most Common Tags in {year}')
    plt.tight_layout()
    plt.show()
    
# Make sure the data_2017_2018 frame only includes videos from the years 2017 - 2018, since most of the videos are from 2017-18
year_range_2017_2018 = np.arange(2017, 2019)
data_2017_2018 = data_2017_2018[data_2017_2018['publish_time'].dt.year.isin(year_range_2017_2018)]

year_range_2020_2024 = np.arange(2020, 2025)
data_2020_2024 = data_2020_2024[data_2020_2024['publishedAt'].dt.year.isin(year_range_2020_2024)]

# Convert trending date format
data_2017_2018['trending_date'] = data_2017_2018['trending_date'].apply(lambda x: datetime.strptime(x, '%y.%d.%m'))
data_2020_2024['trending_date'] = data_2020_2024['trending_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))

# Rename publishedAt to publish_time, categoryID to category_id and view _count to views so column names are consistent 
data_2020_2024 = data_2020_2024.rename(columns={'publishedAt': 'publish_time', 'categoryId': 'category_id', 'view_count': 'views'})

# Exclude music videos
data_2017_2018 = data_2017_2018[data_2017_2018['category_id'] != 10]
data_2020_2024 = data_2020_2024[data_2020_2024['category_id'] != 10]

# Split up tags to extract tags
data_2017_2018['tags'] = data_2017_2018['tags'].apply(split_tags)
data_2020_2024['tags'] = data_2020_2024['tags'].apply(split_tags)

# seperate the data by year, and explode/count each year's tags 
tag_counts_2017 = seperate_explode_count(data_2017_2018, 'publish_time', 'tags', '2017')
tag_counts_2018 = seperate_explode_count(data_2017_2018, 'publish_time', 'tags', '2018')
tag_counts_2020 = seperate_explode_count(data_2020_2024, 'publish_time', 'tags', '2020')
tag_counts_2021 = seperate_explode_count(data_2020_2024, 'publish_time', 'tags', '2021')
tag_counts_2022 = seperate_explode_count(data_2020_2024, 'publish_time', 'tags', '2022')
tag_counts_2023 = seperate_explode_count(data_2020_2024, 'publish_time', 'tags', '2023')
tag_counts_2024 = seperate_explode_count(data_2020_2024, 'publish_time', 'tags', '2024')

# Get top 15 highest frequency tags for each year
top_15_tags_2017 = tag_counts_2017.head(15)
top_15_tags_2018 = tag_counts_2018.head(15)
top_15_tags_2020 = tag_counts_2020.head(15)
top_15_tags_2021 = tag_counts_2021.head(15)
top_15_tags_2022 = tag_counts_2022.head(15)
top_15_tags_2023 = tag_counts_2023.head(15)
top_15_tags_2024 = tag_counts_2024.head(15)

data_2017_2018 = one_hot_encode_tags(data_2017_2018, top_15_tags_2017.index)
data_2017_2018 = one_hot_encode_tags(data_2017_2018, top_15_tags_2018.index)
data_2020_2024 = one_hot_encode_tags(data_2020_2024, top_15_tags_2020.index)
data_2020_2024 = one_hot_encode_tags(data_2020_2024, top_15_tags_2021.index)
data_2020_2024 = one_hot_encode_tags(data_2020_2024, top_15_tags_2022.index)
data_2020_2024 = one_hot_encode_tags(data_2020_2024, top_15_tags_2023.index)
data_2020_2024 = one_hot_encode_tags(data_2020_2024, top_15_tags_2024.index)

features_2017 = ['views', 'likes', 'dislikes', 'comment_count'] + [f'tag: {tag}' for tag in top_15_tags_2017.index]
features_2018 = ['views', 'likes', 'dislikes', 'comment_count'] + [f'tag: {tag}' for tag in top_15_tags_2018.index]
features_2020 = ['views', 'likes', 'dislikes', 'comment_count'] + [f'tag: {tag}' for tag in top_15_tags_2020.index]
features_2021 = ['views', 'likes', 'dislikes', 'comment_count'] + [f'tag: {tag}' for tag in top_15_tags_2021.index]
features_2022 = ['views', 'likes', 'dislikes', 'comment_count'] + [f'tag: {tag}' for tag in top_15_tags_2022.index]
features_2023 = ['views', 'likes', 'dislikes', 'comment_count'] + [f'tag: {tag}' for tag in top_15_tags_2023.index]
features_2024 = ['views', 'likes', 'dislikes', 'comment_count'] + [f'tag: {tag}' for tag in top_15_tags_2024.index]

# Plot heatmap with a correlation matrix of views, likes, dislikes, comment count and tag frequency (see if popular tags impact video metrics) for each year
plot_heatmap(data_2017_2018, features_2017, '2017')
plot_heatmap(data_2017_2018, features_2018, '2018')
plot_heatmap(data_2020_2024, features_2020, '2020')
plot_heatmap(data_2020_2024, features_2021, '2021')
plot_heatmap(data_2020_2024, features_2022, '2022')
plot_heatmap(data_2020_2024, features_2023, '2023')
plot_heatmap(data_2020_2024, features_2024, '2024')

# Plotting the 15 most common tags and their occurrences from 2017-2018 and 2020-2024
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

sns.barplot(ax=ax1, x=top_15_tags_2017.values, y=top_15_tags_2017.index, hue=top_15_tags_2017.index, palette='viridis')
ax1.set_title('Top 15 Most Common Tags in 2017')
ax1.set_xlabel('Occurrences')
ax1.set_ylabel('Tags')

sns.barplot(ax=ax2, x=top_15_tags_2018.values, y=top_15_tags_2018.index, hue=top_15_tags_2018.index, palette='viridis')
ax2.set_title('Top 15 Most Common Tags in 2018')
ax2.set_xlabel('Occurrences')
ax2.set_ylabel('Tags')

plt.tight_layout()
plt.show()

# Plot the top 15 tags from 2017-2018 and 2020-2024
plot_top15_tags(top_15_tags_2017, 2017)
plot_top15_tags(top_15_tags_2018, 2018)
plot_top15_tags(top_15_tags_2020, 2020)
plot_top15_tags(top_15_tags_2021, 2021)
plot_top15_tags(top_15_tags_2022, 2022)
plot_top15_tags(top_15_tags_2023, 2023)
plot_top15_tags(top_15_tags_2024, 2024)


# Plot likes and dislikes
plt.plot(data_2017_2018['trending_date'], data_2017_2018['likes'], color='blue', label='Likes')
plt.plot(data_2017_2018['trending_date'], data_2017_2018['dislikes'], color='red', label='Dislikes')
plt.title('Likes vs. Dislikes OVer Time')
plt.xlabel('Trending Date')
plt.ylabel('Amount')
plt.legend(loc='upper right')
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()
plt.clf()

# Plot views, comment and tag count
plot_characteristic('Views', data_2017_2018['views'])
plot_characteristic('Comment Count', data_2017_2018['comment_count'])
data_2017_2018['tag_count'] = data_2017_2018['tags'].apply(lambda x: x.count('|') + 1)
plot_characteristic('Tag Count', data_2017_2018['tag_count'])

# Bot plot for views, likes, and dislikes
plt.figure(figsize = (12,6))
sns.boxplot(data = data_2017_2018, x = 'category_id', y = 'views')
plt.title('Views and their categories')
plt.xlabel('Category ID')
plt.ylabel('Views')
plt.tight_layout()
plt.show()

plt.figure(figsize = (12,6))
sns.boxplot(data = data_2017_2018, x = 'category_id', y = 'likes')
plt.title('Likes by their categories')
plt.xlabel('Category ID')
plt.ylabel('Likes')
plt.tight_layout()
plt.show()

plt.figure(figsize = (12,6))
sns.boxplot(data = data_2017_2018, x = 'category_id', y = 'dislikes')
plt.title('Dislikes by their categories')
plt.xlabel('Category ID')
plt.ylabel('Dislikes')
plt.tight_layout()
plt.show()

# Plot the video category
plt.hist(data_2017_2018['category_id'], bins=range(1, 45))
plt.title('Category ID')
plt.xlabel('ID')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.clf()

# data_2017_2018['count'] = data_2017_2018.groupby(['category_id', 'publish_time'.month, 'publish_time'.year]).sum()
# gaming = data_2017_2018[data_2017_2018['category_id'] == 20]
# sorted = data_2017_2018.sort_values('publish_time')

# Trend vs publish date
data_2017_2018['publish_time'] = data_2017_2018['publish_time'].apply(lambda x: datetime.strptime(x.strftime('%y.%d.%m'), '%y.%d.%m'))
data_2017_2018['days_since_published'] = (data_2017_2018['trending_date'] - data_2017_2018['publish_time']).dt.days
plot_characteristic('Days Between Publish and Trending', data_2017_2018['days_since_published'])

#Categories over time
data_2017_2018["month"] = data_2017_2018["publish_time"].apply(date_to_month)
grouped_data_2017_2018 = data_2017_2018.groupby(["month", "category_id"]).agg({'views': 'sum'})
sns.lineplot(data=grouped_data_2017_2018, x='month', y='views', style='category_id', hue='category_id', palette='bright', errorbar=None)
plt.legend(title='Category ID', loc='upper center', ncol=8)
plt.xticks(rotation=25)
plt.title('Category Popularity Over Time')
plt.xlabel('Month')
plt.ylabel('Total View Count')
plt.show()