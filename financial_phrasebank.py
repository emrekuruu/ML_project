from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# -------------------------------------
# Step 1: Load and split the dataset
# -------------------------------------
def prepare_datasets(tokenizer, max_length=128):
    dataset = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    df = pd.DataFrame(dataset['train'])

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['sentence'], df['label'], test_size=0.2, stratify=df['label'], random_state=42
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.1, stratify=train_labels, random_state=42
    )

    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)

    def to_dataset(texts, labels):
        return Dataset.from_dict({"sentence": texts, "labels": labels})

    train_ds = to_dataset(train_texts.tolist(), train_labels.tolist()).map(tokenize_fn, batched=True)
    val_ds = to_dataset(val_texts.tolist(), val_labels.tolist()).map(tokenize_fn, batched=True)
    test_ds = to_dataset(test_texts.tolist(), test_labels.tolist()).map(tokenize_fn, batched=True)

    for ds in [train_ds, val_ds, test_ds]:
        ds = ds.remove_columns(["sentence"])
        ds.set_format("torch")

    return train_ds, val_ds, test_ds

# -------------------------------------
# Step 2: Define compute_metrics
# -------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# -------------------------------------
# Step 3: Define model and tokenizer
# -------------------------------------
MODEL_NAME = "google-bert/bert-base-uncased"  # Swap this with any Modern BERT

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# -------------------------------------
# Step 4: Training args and trainer
# -------------------------------------
training_args = TrainingArguments(
    output_dir=f"./results/{MODEL_NAME.split('/')[-1]}",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=f"./logs/{MODEL_NAME.split('/')[-1]}",
    logging_steps=10,
    save_total_limit=2,
    save_strategy="epoch",
    report_to=["tensorboard"],
    load_best_model_at_end=True,
)

train_dataset, val_dataset, test_dataset = prepare_datasets(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# -------------------------------------
# Step 5: Predict on test set and save
# -------------------------------------
pred_output = trainer.predict(test_dataset)
pred_labels = np.argmax(pred_output.predictions, axis=1)

# test_dataset still has your raw sentences and labels
df = pd.DataFrame({
    "sentence":        test_dataset["sentence"],
    "true_label":      test_dataset["labels"],
    "predicted_label": pred_labels
})

df.to_csv("predictions.csv", index=False)
print("Saved predictions to predictions.csv")