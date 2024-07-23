import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import os


# Get the list of tags
def split_tags(tags):
    tags_list = tags.split('|')
    return [tag.strip().lower() for tag in tags_list]


current_dir = os.path.dirname(os.path.abspath(__file__))
old_data = pd.read_csv(os.path.join(current_dir, '..', 'data', 'CAvideos.csv'), parse_dates=['publish_time'])
new_data = pd.read_csv(os.path.join(current_dir, '..', 'data', 'CAvideos2020.csv'), parse_dates=['publishedAt'])

old_data['year'] = old_data['publish_time'].dt.year
new_data['year'] = new_data['publishedAt'].dt.year
old_data = old_data[old_data['year'] >= 2017]

old_data = old_data[old_data['comments_disabled'] == False]
new_data = new_data[new_data['comments_disabled'] == False]
old_data = old_data[old_data['ratings_disabled'] == False]
new_data = new_data[new_data['ratings_disabled'] == False]

old_data = old_data.dropna(subset=['views'])
new_data = new_data.dropna(subset=['view_count'])

old_data['times_trending'] = old_data['video_id'].map(old_data['video_id'].value_counts())
new_data['times_trending'] = new_data['video_id'].map(new_data['video_id'].value_counts())
old_data['tag_count'] = old_data['tags'].apply(split_tags).apply(len)
new_data['tag_count'] = new_data['tags'].apply(split_tags).apply(len)

# Couldn't use view counts for some reason (Nan values)
old_data2 = pd.DataFrame(old_data, columns=['times_trending', 'tag_count', 'dislikes', 'comment_count'])
new_data2 = pd.DataFrame(new_data, columns=['times_trending', 'tag_count', 'likes', 'dislikes', 'comment_count'])

X = pd.concat([old_data2, new_data2])
y = pd.concat([old_data['year'], new_data['year']])

nan_indices = X[X.isna().any(axis=1)].index
X = X.drop(nan_indices)
y = y.drop(nan_indices)

X = X.values
y = y.values

X_train, X_valid, y_train, y_valid = train_test_split(X, y)

# SVC was taking too long so I removed it
model = VotingClassifier([
    ('nb', GaussianNB()),
    ('knn', KNeighborsClassifier(10)),
    ('tree1', DecisionTreeClassifier(max_depth=100)),
    ('tree2', DecisionTreeClassifier(min_samples_leaf=20)),
])
model.fit(X_train, y_train)
print('VotingClassifier score on predicting years:', model.score(X_valid, y_valid))

X2 = pd.concat([old_data2, new_data2])
y2 = pd.concat([old_data['category_id'], new_data['categoryId']])

X2 = X2.drop(nan_indices)
y2 = y2.drop(nan_indices)

X2 = X2.values
y2 = y2.values

X_train2, X_valid2, y_train2, y_valid2 = train_test_split(X2, y2)

model.fit(X_train2, y_train2)
print('VotingClassifier score on predicting category ID:', model.score(X_valid2, y_valid2))
