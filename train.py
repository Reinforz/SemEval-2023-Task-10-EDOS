import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

sexist_label_idx = {'not sexist': 0, 'sexist': 1}
sexist_idx_label = {0: 'not sexist', 1: 'sexist'}

category_label_idx = {'1. threats, plans to harm and incitement': 0,
                      '2. derogation': 1, '3. animosity': 2, '4. prejudiced discussions': 3, 'none': 4}
category_idx_label = {0: '1. threats, plans to harm and incitement',
                      1: '2. derogation', 2: '3. animosity', 3: '4. prejudiced discussions', 4: 'none'}
vector_label_idx = {'1.1 threats of harm': 0, '1.2 incitement and encouragement of harm': 1, '2.1 descriptive attacks': 2, '2.2 aggressive and emotive attacks': 3, '2.3 dehumanising attacks & overt sexual objectification': 4, '3.1 casual use of gendered slurs, profanities, and insults': 5,
                    '3.2 immutable gender differences and gender stereotypes': 6, '3.3 backhanded gendered compliments': 7, '3.4 condescending explanations or unwelcome advice': 8, '4.1 supporting mistreatment of individual women': 9, '4.2 supporting systemic discrimination against women as a group': 10, 'none': 11}
vector_idx_label = {0: '1.1 threats of harm', 1: '1.2 incitement and encouragement of harm', 2: '2.1 descriptive attacks', 3: '2.2 aggressive and emotive attacks', 4: '2.3 dehumanising attacks & overt sexual objectification', 5: '3.1 casual use of gendered slurs, profanities, and insults',
                    6: '3.2 immutable gender differences and gender stereotypes', 7: '3.3 backhanded gendered compliments', 8: '3.4 condescending explanations or unwelcome advice', 9: '4.1 supporting mistreatment of individual women', 10: '4.2 supporting systemic discrimination against women as a group', 11: 'none'}


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_folder = 'data'

edos = load_dataset("json", data_files={
    "train": "data/train.json", "validation": "data/valid.json"})


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


tokenized_edos = edos.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

f1 = evaluate.load("f1")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1.compute(predictions=predictions, references=labels, average="macro")


model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=sexist_idx_label, label2id=sexist_label_idx
)

training_args = TrainingArguments(
    output_dir="results",
    learning_rate=1e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=15,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_edos["train"],
    eval_dataset=tokenized_edos["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
