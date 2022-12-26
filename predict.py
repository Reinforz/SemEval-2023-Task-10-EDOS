import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict(model_name: str, input_csv_file_path: str, output_csv_file_name: str):
    dev_task_a = pd.read_csv(input_csv_file_path,
                             delimiter=",", encoding='utf-8', lineterminator='\n')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    task_result = pd.DataFrame()

    for index in tqdm(range(len(dev_task_a))):
        row = dev_task_a.iloc[index]
        with torch.no_grad():
            text = row["text"]
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs)
            predicted_class_id = np.argmax(outputs.logits.numpy()[0])
            task_result = task_result.append(
                {"rewire_id": row["rewire_id"], "label_pred": model.config.id2label[predicted_class_id]}, ignore_index=True)

    task_result.to_csv(f'{output_csv_file_name}.zip', index=False, compression={
        "method": 'zip', 'archive_name': f'{output_csv_file_name}.csv'})
