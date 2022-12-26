from typing import Dict

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)


def train(model_name: str, feature_name: str, idx_label: Dict[int, str], label_idx: Dict[str, int], output_dir: str = "results"):
    df = load_dataset("json", data_files={
        "train": f'preprocessed/{feature_name}_train.json', "validation": f'preprocessed/{feature_name}_valid.json'})
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_df = df.map(lambda examples: tokenizer(
        examples["text"], truncation=True), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1.compute(predictions=predictions, references=labels, average="macro")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=idx_label, label2id=label_idx
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-6,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=15,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_df["train"],
        eval_dataset=tokenized_df["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer.state.best_model_checkpoint
