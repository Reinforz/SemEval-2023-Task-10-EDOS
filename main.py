from config import idx_label, label_idx
from predict import predict
from preprocess_data import process_data
from train import train


def start(task):
    process_data(label_idx[task], f'label_{task}', "data/train_all_tasks.csv")
    best_model_checkpoint = train(
        "distilbert-base-uncased", f'label_{task}', idx_label[task], label_idx[task])
    predict(best_model_checkpoint,
            f'data/dev_task_{task}.csv', 'task_{task}')


# start("sexist")
# start("category")
# start("vector")
