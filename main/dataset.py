import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split

def prepare_datasets(task, tokenizer, max_length=128):
    if task != "financial":
        raise ValueError(f"Unknown task {task}")

    # 1. Load raw data
    raw = load_dataset(
        "financial_phrasebank",
        "sentences_50agree",
        trust_remote_code=True
    )["train"]
    df = pd.DataFrame(raw)

    # 2. Train/val/test split
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df["sentence"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=42
    )
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels,
        test_size=0.1, stratify=train_labels, random_state=42
    )

    # 3. Tokenization function
    def tokenize_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # 4. Build HF Datasets
    train_ds = Dataset.from_dict({
        "sentence": train_texts.tolist(),
        "labels":  train_labels.tolist()
    })
    val_ds = Dataset.from_dict({
        "sentence": val_texts.tolist(),
        "labels":  val_labels.tolist()
    })
    test_ds = Dataset.from_dict({
        "sentence": test_texts.tolist(),
        "labels":  test_labels.tolist()
    })

    # 5. Tokenize and set torch format
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds   = val_ds.map(tokenize_fn, batched=True)
    test_ds  = test_ds.map(tokenize_fn, batched=True)

    # 6. Remove raw sentence column and prepare for PyTorch
    for ds in (train_ds, val_ds, test_ds):
        ds = ds.remove_columns(["sentence"])
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )

    return train_ds, val_ds, test_ds
