import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

sexist_label_idx = {'not sexist': 0, 'sexist': 1}
sexist_idx_label = {0: 'not sexist', 1: 'sexist'}

data_folder = 'data'
# 80/20 train/dev split
train_split = 0.7

edos_df = pd.read_csv("data/train_all_tasks.csv",
                      delimiter=",", encoding='utf-8', lineterminator='\n')

edos_features = edos_df.drop(columns=["label_sexist"]).copy()
edos_labels = edos_df["label_sexist"]

edos_features_train, edos_features_valid, edos_labels_train, edos_labels_valid = train_test_split(
    edos_features, edos_labels, train_size=train_split, stratify=edos_labels)

edos_train_df = pd.concat([edos_features_train, edos_labels_train], axis=1)
edos_valid_df = pd.concat([edos_features_valid, edos_labels_valid], axis=1)

if not os.path.exists(data_folder):
    os.mkdir(data_folder)


def process_df(df, data_partition: str):
    feature_list = []
    for i in range(len(df)):
        row = df.iloc[i]
        text = row['text'].strip('\r\n')
        features = {
            'idx': i,
            'text': text,
            'label': sexist_label_idx[row['label_sexist']],
        }

        feature_list.append(features)
    df = pd.DataFrame(feature_list)
    json.dump(df.to_dict('records'), open(os.path.join(
        data_folder, data_partition+".json"), mode='w'))
    print('processed {}'.format(data_partition))


process_df(edos_train_df, "train")
process_df(edos_valid_df, "valid")
