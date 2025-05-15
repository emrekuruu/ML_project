import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


def prepare_datasets(task, tokenizer, max_length=128):
    """
    Prepare train/validation/test datasets for financial, emotion, or news tasks.
    All splits use 'sentence' for the raw text column and 'labels' for the label column.
    """
    def unify_columns(ds, text_col, label_col):
        # rename raw text field to 'sentence'
        if text_col != 'sentence':
            ds = ds.rename_column(text_col, 'sentence')
        # rename label field to 'labels'
        if label_col != 'labels':
            ds = ds.rename_column(label_col, 'labels')
        return ds

    # --- Task-specific loading and initial splits ---
    if task == 'financial':
        raw = load_dataset(
            'financial_phrasebank',
            'sentences_50agree',
            trust_remote_code=True
        )['train']
        df = pd.DataFrame(raw)

        # Train/val/test splits
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['sentence'], df['label'],
            test_size=0.2, stratify=df['label'], random_state=42
        )
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels,
            test_size=0.1, stratify=train_labels, random_state=42
        )

        train_ds = Dataset.from_dict({
            'sentence': train_texts.tolist(),
            'labels':  train_labels.tolist()
        })
        val_ds = Dataset.from_dict({
            'sentence': val_texts.tolist(),
            'labels':  val_labels.tolist()
        })
        test_ds = Dataset.from_dict({
            'sentence': test_texts.tolist(),
            'labels':  test_labels.tolist()
        })
        text_field = 'sentence'
        label_field = 'labels'

    elif task == 'emotion':
        raw = load_dataset('dair-ai/emotion')
        train_ds, val_ds, test_ds = raw['train'], raw['validation'], raw['test']
        text_field = 'text'
        label_field = 'label'

    elif task == 'news':
        raw = load_dataset('SetFit/ag_news')
        splits = raw['train'].train_test_split(
            test_size=0.1,
            stratify_by_column='label',
            seed=42
        )
        train_ds, val_ds, test_ds = splits['train'], splits['test'], raw['test']
        text_field = 'text'
        label_field = 'label'

    else:
        raise ValueError(f"Unknown task {task}")

    # unify column names across splits
    train_ds = unify_columns(train_ds, text_field, label_field)
    val_ds = unify_columns(val_ds, text_field, label_field)
    test_ds = unify_columns(test_ds, text_field, label_field)

    # --- Common tokenization ---
    def tokenize_fn(examples):
        return tokenizer(
            examples['sentence'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    # --- Format train & val for PyTorch ---
    for ds in (train_ds, val_ds):
        ds = ds.remove_columns(['sentence'])
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # leave test_ds unformatted so it retains 'sentence' and 'labels'
    return train_ds, val_ds, test_ds