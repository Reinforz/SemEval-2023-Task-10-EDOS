import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

dev_task_a = pd.read_csv("data/dev_task_a_entries.csv",
                         delimiter=",", encoding='utf-8', lineterminator='\n')

model = AutoModelForSequenceClassification.from_pretrained(
    "results/checkpoint-6130")
tokenizer = AutoTokenizer.from_pretrained("results/checkpoint-6130")

dev_task_a_result = pd.DataFrame()

for index in tqdm(range(len(dev_task_a))):
    row = dev_task_a.iloc[index]
    with torch.no_grad():
        text = row["text"]
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        predicted_class_id = np.argmax(outputs.logits.numpy()[0])
        dev_task_a_result = dev_task_a_result.append(
            {"rewire_id": row["rewire_id"], "label_pred": model.config.id2label[predicted_class_id]}, ignore_index=True)


dev_task_a_result.to_csv('dev_task_a_entries.zip', index=False, compression={
                         "method": 'zip', 'archive_name': 'dev_task_a_entries.csv'})
