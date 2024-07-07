import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import os


def plot_characteristic(name, column):
    plt.plot(data['trending_date'], column)
    plt.title(name)
    plt.xlabel('Trending Date')
    plt.ylabel('Amount')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()
    plt.clf()

def date_to_month(d):
    # You may need to modify this function, depending on your data types.
    return '%04i-%02i' % (d.year, d.month)

# Function to split tags 
def split_tags(tags):
    tags_list = tags.split('|')
    return [tag.strip().lower() for tag in tags_list]

# Fetch the data from the csv file
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'data', 'CAvideos.csv')
data = pd.read_csv(csv_path, parse_dates=['publish_time'])

# Make sure the data frame only includes videos from the years 2017 - 2018, since most of the videos are from 2017-18
year_range = np.arange(2017, 2019)
data = data[data['publish_time'].dt.year.isin(year_range)]

# Convert trending date format
data['trending_date'] = data['trending_date'].apply(lambda x: datetime.strptime(x, '%y.%d.%m'))

# Exclude music videos
data = data[data['category_id'] != 10]

# Split up tags to extract tags
data['tags'] = data['tags'].apply(split_tags)

# Separate data for 2017 and 2018
data_2017 = data[data['publish_time'].dt.year == 2017].copy()
data_2018 = data[data['publish_time'].dt.year == 2018].copy()

# Explode the tags
exploded_tags_2017 = data_2017.explode('tags')
exploded_tags_2018 = data_2018.explode('tags')

# Frequency encoding for tags
tag_counts_2017 = exploded_tags_2017['tags'].value_counts()
tag_counts_2018 = exploded_tags_2018['tags'].value_counts()

# Get top 15 highest frequency tags for each year
top_15_tags_2017 = tag_counts_2017.head(15)
top_15_tags_2018 = tag_counts_2018.head(15)

# Plotting the 15 most common tags and their occurrences for 2017 and 2018
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

sns.barplot(ax=ax1, x=top_15_tags_2017.values, y=top_15_tags_2017.index, hue=top_15_tags_2017.index, palette='bright')
ax1.set_title('Top 15 Most Common Tags in 2017')
ax1.set_xlabel('Occurrences')
ax1.set_ylabel('Tags')

sns.barplot(ax=ax2, x=top_15_tags_2018.values, y=top_15_tags_2018.index, hue=top_15_tags_2018.index, palette='bright')
ax2.set_title('Top 15 Most Common Tags in 2018')
ax2.set_xlabel('Occurrences')
ax2.set_ylabel('Tags')

plt.tight_layout()
plt.show()


# One-hot encode the tags for each the 2017 and 2018 data: 1 if video has a top 20 tag, 0 otherwise, got help from: https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
for tag in top_15_tags_2017.index:
    data_2017.loc[:, f'tag: {tag}'] = data_2017['tags'].apply(lambda tags: 1 if tag in tags else 0)

for tag in top_15_tags_2018.index:
    data_2018.loc[:, f'tag: {tag}'] = data_2018['tags'].apply(lambda tags: 1 if tag in tags else 0)

features_2017 = ['views', 'likes', 'dislikes', 'comment_count'] + [f'tag: {tag}' for tag in top_15_tags_2017.index]
features_2018 = ['views', 'likes', 'dislikes', 'comment_count'] + [f'tag: {tag}' for tag in top_15_tags_2018.index]

# Plot a heatmap with a correlation matrix of views, likes, dislikes, comment count and tag frequency (see if popular tags impact video metrics) for 2017 and 2018
plt.figure(figsize=(14, 10))
correlation_matrix_2017 = data_2017[features_2017].corr()
sns.heatmap(correlation_matrix_2017, annot=True)
plt.title('Correlation Matrix of Trending Video Metrics and Top 15 Most Common Tags in 2017')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
correlation_matrix_2018 = data_2018[features_2018].corr()
sns.heatmap(correlation_matrix_2018, annot=True)
plt.title('Correlation Matrix of Trending Video Metrics and Top 15 Most Common Tags in 2018')
plt.tight_layout()
plt.show()


# Plot likes and dislikes
plt.plot(data['trending_date'], data['likes'], color='blue', label='Likes')
plt.plot(data['trending_date'], data['dislikes'], color='red', label='Dislikes')
plt.title('Likes vs. Dislikes OVer Time')
plt.xlabel('Trending Date')
plt.ylabel('Amount')
plt.legend(loc='upper right')
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()
plt.clf()

# Plot views, comment and tag count
plot_characteristic('Views', data['views'])
plot_characteristic('Comment Count', data['comment_count'])

data['tag_count'] = data['tags'].apply(lambda x: x.count('|') + 1)
plot_characteristic('Tag Count', data['tag_count'])

# Bot plot for views, likes, and dislikes
plt.figure(figsize = (12,6))
sns.boxplot( data = data, x = 'category_id', y = 'views')
plt.title('Views and their categories')
plt.xlabel('Category ID')
plt.ylabel('Views')
plt.tight_layout()
plt.show()

plt.figure(figsize = (12,6))
sns.boxplot(data = data, x = 'category_id', y = 'likes')
plt.title('Likes by their categories')
plt.xlabel('Category ID')
plt.ylabel('Likes')
plt.tight_layout()
plt.show()

plt.figure(figsize = (12,6))
sns.boxplot(data = data, x = 'category_id', y = 'dislikes')
plt.title('Dislikes by their categories')
plt.xlabel('Category ID')
plt.ylabel('Dislikes')
plt.tight_layout()
plt.show()

# Plot the video category
plt.hist(data['category_id'], bins=range(1, 45))
plt.title('Category ID')
plt.xlabel('ID')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.clf()

# data['count'] = data.groupby(['category_id', 'publish_time'.month, 'publish_time'.year]).sum()
# gaming = data[data['category_id'] == 20]
# sorted = data.sort_values('publish_time')

# Trend vs publish date
data['publish_time'] = data['publish_time'].apply(lambda x: datetime.strptime(x.strftime('%y.%d.%m'), '%y.%d.%m'))
data['days_since_published'] = (data['trending_date'] - data['publish_time']).dt.days
plot_characteristic('Days Between Publish and Trending', data['days_since_published'])

#Categories over time
data["month"] = data["publish_time"].apply(date_to_month)
grouped_data = data.groupby(["month", "category_id"]).agg({'views': 'sum'})
sns.lineplot(data=grouped_data, x='month', y='views', style='category_id', hue='category_id', palette='bright', errorbar=None)
plt.legend(title='Category ID', loc='upper center', ncol=8)
plt.xticks(rotation=25)
plt.title('Category Popularity Over Time')
plt.xlabel('Month')
plt.ylabel('Total View Count')
plt.show()