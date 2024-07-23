from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# Get the list of tags
def split_tags(tags):
    tags_list = tags.split('|')
    return [tag.strip().lower() for tag in tags_list]


# Get the number of words ina  string
def count_words(text):
    return len((str(text)).split())


# Compare a category over the years
def compare_years(title):

    # Sort the category by year
    old_years = old_data.groupby('year')[title].mean().reset_index()
    old_years.loc[2] = {'year': 2019, title: np.nan}
    new_years = new_data.groupby('year')[title].mean().reset_index()
    combined = pd.concat([old_years, new_years])

    # Post-Hoc analysis
    post_hoc_data = pd.concat([old_data, new_data])
    tukey = pairwise_tukeyhsd(endog=post_hoc_data[title], groups=post_hoc_data['year'], alpha=0.05)
    print('\n\n==========', title, '==========')
    print(tukey.summary())

    # Plot results
    plt.plot(combined['year'], combined[title], marker='o')
    plt.title(f'{title} By Year (check output for post-hoc summary)')
    plt.xlabel('Year')
    plt.ylabel(title)
    plt.tight_layout()
    plt.show()


# Get the data
current_dir = os.path.dirname(os.path.abspath(__file__))
old_data = pd.read_csv(os.path.join(current_dir, '..', 'data', 'CAvideos.csv'), parse_dates=['publish_time'])
new_data = pd.read_csv(os.path.join(current_dir, '..', 'data', 'CAvideos2020.csv'), parse_dates=['publishedAt'])

# filter data
old_data['year'] = old_data['publish_time'].dt.year
new_data['year'] = new_data['publishedAt'].dt.year
old_data = old_data[old_data['year'] >= 2017]

# Calculate categories
old_data['tag_count'] = old_data['tags'].apply(split_tags).apply(len)
new_data['tag_count'] = new_data['tags'].apply(split_tags).apply(len)
compare_years('tag_count')

compare_years('likes')
compare_years('dislikes')
compare_years('comment_count')

new_data.rename(columns={'view_count': 'views'}, inplace=True)
compare_years('views')

old_data['times_trending'] = old_data['video_id'].map(old_data['video_id'].value_counts())
new_data['times_trending'] = new_data['video_id'].map(new_data['video_id'].value_counts())
compare_years('times_trending')

old_data['description_length'] = old_data['description'].apply(count_words)
new_data['description_length'] = new_data['description'].apply(count_words)
compare_years('description_length')
