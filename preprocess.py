import tensorflow as tf
import pandas as pd
from .constants import CSV_HEADER, TARGET_FEATURE_NAME, WEIGHT_COLUMN_NAME, NUMERIC_FEATURE_NAMES, COLUMN_DEFAULTS, CATEGORICAL_FEATURES_WITH_VOCABULARY


##Helper functions for preprocessing of data:

def process(features, target):
    for feature_name in features:
        if feature_name in CATEGORICAL_FEATURES_WITH_VOCABULARY:
            # Cast categorical feature values to string.
            features[feature_name] = tf.cast(features[feature_name], tf.dtypes.string)
    # Get the instance weight.
    weight = features.pop(WEIGHT_COLUMN_NAME)
    return features, target, weight


def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):

    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=False,
        shuffle=shuffle,
    ).map(process)

    return dataset

def create_max_values_map():
    max_values_map = {}
    for col in NUMERIC_FEATURE_NAMES:
        max_val = max(test_data[col])
        max_values_map["max_"+col] = max_val
    return max_values_map

def create_dropdown_default_values_map():
    dropdown_default_values_map = {}
    for col in CATEGORICAL_FEATURES_WITH_VOCABULARY.keys():
        max_val = test_data[col].max()
        dropdown_default_values_map["max_"+col] = max_val
    return dropdown_default_values_map

def load_test_data():
    
    test_data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz"
    test_data = pd.read_csv(test_data_url, header=None, names=CSV_HEADER)
    
    return test_data

def create_sample_test_data():
    
    test_data = load_test_data()
    
    test_data["income_level"] = test_data["income_level"].apply(
    lambda x: 0 if x == " - 50000." else 1)
    
    sample_df = test_data.loc[:20,:]
    sample_df_values = samp.values.tolist()
    
    return sample_df_values
    