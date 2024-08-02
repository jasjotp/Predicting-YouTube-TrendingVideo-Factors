import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import os

# All the numerical data columns
x_columns = ['times_trending', 'tag_count', 'likes', 'views', 'dislikes', 'comment_count', 'category_id',
             'description_length', 'year_2017', 'year_2018', 'year_2020', 'year_2021', 'year_2022', 'year_2023', 'year_2024']


# Get the list of tags
def split_tags(tags):
    tags_list = tags.split('|')
    return [tag.strip().lower() for tag in tags_list]


def main():
    # Parse the data into frames
    current_dir = os.path.dirname(os.path.abspath(__file__))
    old_data = pd.read_csv(os.path.join(current_dir, '..', 'data', 'CAvideos.csv'), parse_dates=['publish_time'])
    new_data = pd.read_csv(os.path.join(current_dir, '..', 'data', 'CAvideos_2020-2024.csv'), parse_dates=['publishedAt'])

    # Clean the data
    data = combine_data(old_data, new_data)
    data = add_data_columns(data)
    data = filter_data(data)
    data = encode(data)

    # Predictions
    predict_columns(data)
    predict_mock_data(data)


# Combine the old and new datasets into one
def combine_data(old_data, new_data):
    # Remove columns that don't appear in both sets
    old_data = old_data.drop('video_error_or_removed', axis=1)
    new_data = new_data.drop('channelId', axis=1)

    # Rename the new columns to match the old ones
    new_data.columns = ['video_id', 'title', 'publish_time', 'channel_title', 'category_id', 'trending_date', 'tags',
                        'views', 'likes', 'dislikes', 'comment_count', 'thumbnail_link', 'comments_disabled',
                        'ratings_disabled', 'description']

    # Return the concatenated data
    return pd.concat([old_data, new_data], ignore_index=True)


# Add new insightful columns to the data frame
def add_data_columns(data):
    # Store the year value separately
    data['year'] = data['publish_time'].dt.year

    # Track how many times each video has trended
    data['times_trending'] = data['video_id'].map(data['video_id'].value_counts())

    # Add a tag count column
    data['tag_count'] = data['tags'].apply(split_tags).apply(len)

    # Count the length of the video's description in characters
    data['description_length'] = data['description'].apply(lambda x: len((str(x)).split()))

    return data


# Filter and clean the data of unwanted rows
def filter_data(data):
    # There is not enough data before 2017 to work with
    data = data[data['year'] >= 2017]

    # Remove rows where some values are hidden
    data = data[data['comments_disabled'] != True]
    data = data[data['ratings_disabled'] != True]

    # These values can be too large to work with, so we scale
    data['views'] = (data['views'] / 100000).astype(int)
    data['likes'] = (data['likes'] / 10000).astype(int)
    data['dislikes'] = (data['dislikes'] / 1000).astype(int)
    data['comment_count'] = (data['comment_count'] / 10000).astype(int)

    return data

def encode(data):
    categorical_columns = ['year']
    encoder = OneHotEncoder(sparse_output=False)
    one_hot_encoded  = encoder.fit_transform(data[categorical_columns])
    one_hot_data = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
    data_encoded = pd.concat([data.reset_index(drop=True), one_hot_data.reset_index(drop=True)], axis=1)
    data_encoded = data_encoded.drop(categorical_columns, axis=1)
    print(list(data_encoded.columns.values))
    return data_encoded


# Predict and score the accuracy of certain columns
def predict_columns(data):
    prediction_columns = ['times_trending', 'views']

    # Loop through all the columns we want to predict
    for y_column in prediction_columns:
        # Remove the label form the x values
        new_x_columns = x_columns[:]
        new_x_columns.remove(y_column)

        # Split the data into train and validate sets
        if y_column == 'views' or y_column == 'likes':
            # Views can run into memory issues, so a subsample of the dataset is needed
            X_train, X_valid, y_train, y_valid = get_data_splits(data.sample(n=100000), new_x_columns, y_column)
        else:
            X_train, X_valid, y_train, y_valid = get_data_splits(data, new_x_columns, y_column)

        # Create a random forest classifier model with a scaler
        model = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=50))

        # Train model and print score
        model.fit(X_train, y_train)
        print('RandomForestClassifier score on predicting', y_column, ':', model.score(X_valid, y_valid))


# Predict how many times a video with generated data will trend
def predict_mock_data(data):
    # Predict for times_trending
    new_x_columns = x_columns[:]
    new_x_columns.remove('times_trending')

    # Split the data into train and validate sets
    X_train, X_valid, y_train, y_valid = get_data_splits(data, new_x_columns, 'times_trending')
    model = make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=50))

    # Generate mock data
    mock_data = np.mean(X_train, axis=0).reshape(1, -1)

    # Train model and print prediction
    model.fit(X_train, y_train)
    print('Prediction on how many times the mock data will trend:', model.predict(mock_data))


# Split the data into train and validate sets
def get_data_splits(data, new_x_columns, y_column):
    # Sort X and y values
    X = pd.DataFrame(data, columns=new_x_columns)
    y = data[y_column]

    # Turn the pandas frames into numpy arrays
    X = X.values
    y = y.values

    return train_test_split(X, y)


main()
