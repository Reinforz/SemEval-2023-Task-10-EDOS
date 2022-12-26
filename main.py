from config import category_label_idx, sexist_label_idx, vector_label_idx
from preprocess_data import process_data

process_data(sexist_label_idx, "label_sexist", "data/train_all_tasks.csv")
process_data(category_label_idx, "label_category", "data/train_all_tasks.csv")
process_data(vector_label_idx, "label_vector", "data/train_all_tasks.csv")
