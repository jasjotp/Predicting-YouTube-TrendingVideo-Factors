import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import os

# no 10, 18, 21, 31-42, 44 + 43 contains tamil, not supported
categories = [1, 2, 15, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            #   , 43]
category_names = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music", 15: "Pets & Animals", 17: "Sports", 18: "Short Movies", 19: "Travel & Events", 20: "Gaming", 21: "Videoblogging",
    22: "People & Blogs", 23: "Comedy", 24: "Entertainment", 25: "News & Politics", 26: "Howto & Style", 27: "Education", 28: "Science & Technology", 29: "Nonprofits & Activism",
    30: "Movies", 31: "Anime/Animation", 32: "Action/Adventure", 33: "Classics", 34: 'Comedy', 35: "Documentary", 36: "Drama", 37: "Family", 38: "Foreign", 39: "Horror",
    40: "Sci-Fi/Fantasy", 41: "Thriller", 42: "Shorts", 43: "Shows", 44: "Trailers"
}

def plot_characteristic(data, name, column):
    plt.plot(data['trending_date'], column)
    plt.title(name)
    plt.xlabel('Trending Date')
    plt.ylabel('Amount')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(f'graphs/characteristics/{name}.png')
    plt.clf()

def date_to_month(d):
    return '%04i-%02i' % (d.year, d.month)

# Function to split tags 
def split_tags(tags):
    return [tag.strip().lower() for tag in tags.split('|')]

# Fuction to create a barplot with the top 15 tags for each year, got help from: https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
def plot_top15_tags(tag_data, year):
    tags_df = pd.DataFrame({'tag': tag_data.index, 'count': tag_data.values})
    plt.figure(figsize=(10, 8))
    sns.barplot(x='count', y='tag', data=tags_df, hue=tag_data.index, palette='viridis')
    plt.title(f'Top 15 Most Common Tags in {year}')
    plt.xlabel('Occurrences')
    plt.ylabel('Tags')
    plt.tight_layout()
    plt.savefig(f'graphs/tags_by_year/{year}.png')
    plt.close()

def plot_top10_by_category(tag_data, category_name):
    sns.barplot(x='count', y='tags', data=tag_data, hue=tag_data.index, palette='viridis', legend=False)
    plt.title(f'Top 10 Most Common Tags in {category_name}')
    plt.xlabel('Occurrences')
    plt.ylabel('Tags')
    plt.tight_layout()
    plt.savefig(f'graphs/tags_by_cat/{category_name}.png')
    plt.close()

def get_top_tags_for_category(df, category):
    return df[df['category_id'] == category].iloc[:10]

# Function to seperate tags by year, explode them to retrive each tag, and count their values. got help from: got help from: https://www.datacamp.com/tutorial/pandas-explode
def separate_explode_count(data, published_column, tags_column, year):
    data_year = data[data[published_column].dt.year == int(year)].copy()  # Separate data by year
    exploded_tags = data_year.explode(tags_column)  # Explode the tags so each tag gets its own row for counting purposes
    tag_counts = exploded_tags[tags_column].value_counts()  # Count each occurrence of each tag
    return tag_counts

# Function to one-hot encode tags: 1 if video has a top 15 tag, 0 otherwise, got help from: https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list
def one_hot_encode_tags(data, tags):
    for tag in tags:
        data.loc[:, f'tag: {tag}'] = data['tags'].apply(lambda tags: 1 if tag in tags else 0)
    return data

# Function to plot heatmaps for each year
def plot_heatmap(data, features, year):
    plt.figure(figsize=(14, 10))
    correlation_matrix = data[features].corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.title(f'Correlation Matrix of Trending Video Metrics and Top 15 Most Common Tags in {year}')
    plt.tight_layout()
    plt.savefig(f'graphs/heatmaps/{year}.png')
    plt.close()

def plot_boxplot(data, column, title, save_path):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='category_id', y=column)
    plt.title(title)
    plt.xlabel('Category ID')
    plt.ylabel(column.capitalize())
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{column}.png'))
    plt.close()

def preprocess_data(data, year_range):
    data = data[data['publish_time'].dt.year.isin(year_range)]
    data = data[data['category_id'] != 10]
    data['tags'] = data['tags'].apply(split_tags)
    return data

def main():
    # Fetch the data_2017_2018 from the csv file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path_2017_2018 = os.path.join(current_dir, '..', 'data', 'CAvideos.csv')
    csv_path_2020_2024 = os.path.join(current_dir, '..', 'data', 'CAvideos_2020-2024.csv')

    data_2017_2018 = pd.read_csv(csv_path_2017_2018, parse_dates=['publish_time'])
    data_2020_2024 = pd.read_csv(csv_path_2020_2024, parse_dates=['publishedAt'])

    # Rename publishedAt to publish_time, categoryID to category_id and view _count to views so column names are consistent 
    data_2020_2024 = data_2020_2024.rename(columns={'publishedAt': 'publish_time', 'categoryId': 'category_id', 'view_count': 'views'})

    combined_data = pd.concat([data_2017_2018, data_2020_2024])

    # Make sure the data_2017_2018 frame only includes videos from the years 2017 - 2018, since most of the videos are from 2017-18
    # Exclude music videos
    # Split up tags to extract tags
    data_2017_2018 = preprocess_data(data_2017_2018, np.arange(2017, 2019))
    data_2020_2024 = preprocess_data(data_2020_2024, np.arange(2020, 2025))
    
    # Convert trending date format
    data_2017_2018['trending_date'] = data_2017_2018['trending_date'].apply(lambda x: datetime.strptime(x, '%y.%d.%m'))
    data_2020_2024['trending_date'] = data_2020_2024['trending_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ'))

    # Top tags per category
    exploded_tags = data_2017_2018.explode('tags')
    tag_counts = exploded_tags.groupby(['category_id', 'tags']).size().reset_index(name='count')
    top_tags = tag_counts.sort_values(['category_id', 'count'], ascending=[True, False])

    # Seperate by category and plot
    for category in categories:
        top_tags_for_category = get_top_tags_for_category(top_tags, category)
        plot_top10_by_category(top_tags_for_category, category_names[category])

    # seperate the data by year, and explode/count each year's tags 
    year_data = {
        '2017': separate_explode_count(data_2017_2018, 'publish_time', 'tags', '2017'),
        '2018': separate_explode_count(data_2017_2018, 'publish_time', 'tags', '2018'),
        '2020': separate_explode_count(data_2020_2024, 'publish_time', 'tags', '2020'),
        '2021': separate_explode_count(data_2020_2024, 'publish_time', 'tags', '2021'),
        '2022': separate_explode_count(data_2020_2024, 'publish_time', 'tags', '2022'),
        '2023': separate_explode_count(data_2020_2024, 'publish_time', 'tags', '2023'),
        '2024': separate_explode_count(data_2020_2024, 'publish_time', 'tags', '2024'),
    }
   
    # Get top 15 highest frequency tags for each year
    top_15_tags = {year: tags.head(15) for year, tags in year_data.items()}

    # Plotting the 15 most common tags and their occurrences from 2017-2018 and 2020-2024
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    sns.barplot(ax=ax1, x=top_15_tags['2017'], y=top_15_tags['2017'].index, hue=top_15_tags['2017'].index, palette='viridis')
    ax1.set_title('Top 15 Most Common Tags in 2017')
    ax1.set_xlabel('Occurrences')
    ax1.set_ylabel('Tags')

    sns.barplot(ax=ax2, x=top_15_tags['2018'].values, y=top_15_tags['2018'].index, hue=top_15_tags['2018'].index, palette='viridis')
    ax2.set_title('Top 15 Most Common Tags in 2018')
    ax2.set_xlabel('Occurrences')
    ax2.set_ylabel('Tags')

    plt.tight_layout()
    save_path = os.path.join(current_dir, '..', 'graphs', 'tags_2017_2018.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    # Plot the top 15 tags from 2017-2018 and 2020-2024
    for year, tags in top_15_tags.items():
        plot_top15_tags(tags, year)

    for year, tags in top_15_tags.items():
        if year in ['2017', '2018']:
            data_2017_2018 = one_hot_encode_tags(data_2017_2018, tags.index)
        else:
            data_2020_2024 = one_hot_encode_tags(data_2020_2024, tags.index)

    base_features = ['views', 'likes', 'dislikes', 'comment_count']
    features = {year: base_features + [f'tag: {tag}' for tag in tags.index] for year, tags in top_15_tags.items()}

    # Plot heatmap with a correlation matrix of views, likes, dislikes, comment count and tag frequency (see if popular tags impact video metrics) for each year
    for year in top_15_tags.keys():
        if year in ['2017', '2018']:
            plot_heatmap(data_2017_2018, features[year], year)
        else:
            plot_heatmap(data_2020_2024, features[year], year)

    # Plot likes and dislikes
    plt.figure(figsize=(10, 8))
    plt.plot(data_2017_2018['trending_date'], data_2017_2018['likes'], color='blue', label='Likes')
    plt.plot(data_2017_2018['trending_date'], data_2017_2018['dislikes'], color='red', label='Dislikes')
    plt.title('Likes vs. Dislikes OVer Time')
    plt.xlabel('Trending Date')
    plt.ylabel('Amount')
    plt.legend(loc='upper right')
    plt.xticks(rotation=25)
    plt.tight_layout()
    save_path = os.path.join(current_dir, '..', 'graphs', 'likes_vs_dislikes.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()

    # Plot views, comment and tag count
    plot_characteristic(data_2017_2018, 'Views', data_2017_2018['views'])
    plot_characteristic(data_2017_2018, 'Comment Count', data_2017_2018['comment_count'])
    data_2017_2018['tag_count'] = data_2017_2018['tags'].apply(lambda x: x.count('|') + 1)
    plot_characteristic(data_2017_2018, 'Tag Count', data_2017_2018['tag_count'])

    # Bot plot for views, likes, and dislikes
    save_path = os.path.join(current_dir, '..', 'graphs')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_boxplot(data_2017_2018, 'views', 'Views and their categories', save_path)
    plot_boxplot(data_2017_2018, 'likes', 'Likes by their categories', save_path)
    plot_boxplot(data_2017_2018, 'dislikes', 'Dislikes by their categories', save_path)

    # Plot the video category
    plt.hist(data_2017_2018['category_id'], bins=range(1, 45))
    plt.title('Category ID')
    plt.xlabel('ID')
    plt.ylabel('Frequency')
    plt.tight_layout()
    save_path = os.path.join(current_dir, '..', 'graphs', 'categories.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.clf()

    # Trend vs publish date
    data_2017_2018['publish_time'] = data_2017_2018['publish_time'].apply(lambda x: datetime.strptime(x.strftime('%y.%d.%m'), '%y.%d.%m'))
    data_2017_2018['days_since_published'] = (data_2017_2018['trending_date'] - data_2017_2018['publish_time']).dt.days
    plot_characteristic(data_2017_2018, 'Days Between Publish and Trending', data_2017_2018['days_since_published'])

    #Categories over time
    combined_data["month"] = combined_data["publish_time"].apply(date_to_month)
    grouped_data_2017_2018 = combined_data.groupby(["month", "category_id"]).agg({'views': 'sum'})
    sns.lineplot(data=grouped_data_2017_2018, x='month', y='views', style='category_id', hue='category_id', palette='bright', errorbar=None)
    plt.legend(title='Category ID', loc='upper center', ncol=8)
    plt.xticks(rotation=25)
    plt.title('Category Popularity Over Time')
    plt.xlabel('Month')
    plt.ylabel('Total View Count')
    save_path = os.path.join(current_dir, '..', 'graphs', 'cats_over_time.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

if __name__ == '__main__':
    main()