import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
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

# Esxplode the tags (have one tag per row). got help from: https://www.datacamp.com/tutorial/pandas-explode
exploded_tags = data.explode('tags')

# Frequency encoding for tags
tag_counts = exploded_tags['tags'].value_counts()

# Focus on top 15 highest frequency tags to see which tags impact video performance 
top_15_tags = tag_counts.head(15)

# One-hot encode the tags: 1 if video has a top 20 tag, 0 otherwise, got help from: https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
for tag in top_15_tags.index:
    data[f'tag: {tag}'] = data['tags'].apply(lambda tags: 1 if tag in tags else 0)

features = ['views', 'likes', 'dislikes', 'comment_count'] + [f'tag: {tag}' for tag in top_15_tags.index]

# Plot a heatmap with a correlation matrix of viws, likes, dislikes, comment count and tag frequency (see if popular tags impact video metrics)
plt.figure(figsize=(14, 10))
correlation_matrix = data[features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Trending Video Metrics and Top 15 Most Common Tags')
plt.tight_layout()
plt.show()

# Plotting the 15 most common tags and their occurrences
# got help from: https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
plt.figure(figsize=(12, 6))
sns.barplot(x=top_15_tags.values, y=top_15_tags.index, hue=top_15_tags.index)
plt.title('Top 15 Most Common Tags')
plt.xlabel('Occurrences')
plt.ylabel('Tags')
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