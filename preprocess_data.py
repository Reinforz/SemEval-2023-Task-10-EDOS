import json
import os
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split


def generate_df(input_file_path: str, feature_name: str, train_split: int = 0.8):
    df = pd.read_csv(input_file_path,
                     delimiter=",", encoding='utf-8', lineterminator='\n')

    features = df.drop(columns=[feature_name]).copy()
    labels = df[feature_name]

    features_train, features_valid, labels_train, labels_valid = train_test_split(
        features, labels, train_size=train_split, stratify=labels)

    train_df = pd.concat([features_train, labels_train], axis=1)
    valid_df = pd.concat([features_valid, labels_valid], axis=1)

    return (train_df, valid_df)


def process_df(df, data_partition: str, data_folder: str, label_idx: Dict[str, int], feature_name: str):
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    feature_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        text = row['text'].strip('\r\n')
        features = {
            'idx': i,
            'text': text,
            'label': label_idx[row[feature_name]],
        }

        feature_list.append(features)
    df = pd.DataFrame(feature_list)
    json.dump(df.to_dict('records'), open(os.path.join(
        data_folder, data_partition+".json"), mode='w'))
    print('processed {}'.format(data_partition))


def process_data(label_idx: Dict[str, int], feature_name: str, input_file_path: str, train_split: int = 0.8):
    (train_df, valid_df) = generate_df(
        input_file_path, feature_name, train_split)
    process_df(train_df, f'{feature_name}_train',
               "data", label_idx, feature_name)
    process_df(valid_df, f'{feature_name}_valid',
               "data", label_idx, feature_name)
