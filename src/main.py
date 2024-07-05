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


# Fetch the data from the csv file
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, '..', 'data', 'CAvideos.csv')
data = pd.read_csv(csv_path, parse_dates=['publish_time'])

# Make sure the data frame only includes videos from the years 2012 - 2018
year_range = np.arange(2017, 2019)
data = data[data['publish_time'].dt.year.isin(year_range)]

# Convert trending date format
data['trending_date'] = data['trending_date'].apply(lambda x: datetime.strptime(x, '%y.%d.%m'))

# Exclude music videos
data = data[data['category_id'] != 10]

# Plot likes and dislikes
plt.plot(data['trending_date'], data['likes'], color='blue', label='Likes')
plt.plot(data['trending_date'], data['dislikes'], color='red', label='Dislikes')
plt.title('Likes vs. Dislikes')
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