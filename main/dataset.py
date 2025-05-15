import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split


def prepare_datasets(task, tokenizer, max_length=128):
    """
    Prepare train/validation/test datasets for financial, emotion, or news tasks.
    """
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

    elif task == 'emotion':
        raw = load_dataset('dair-ai/emotion', cache_dir=None)
        train_ds = raw['train']
        val_ds   = raw['validation']
        test_ds  = raw['test']
        text_field = 'text'

        # Align label column
        train_ds = train_ds.rename_column('label', 'labels')
        val_ds   = val_ds.rename_column('label', 'labels')
        test_ds  = test_ds.rename_column('label', 'labels')

    elif task == 'news':
        raw = load_dataset('SetFit/ag_news')
        splits = raw['train'].train_test_split(
            test_size=0.1,
            stratify_by_column='label',
            seed=42
        )
        train_ds = splits['train']
        val_ds   = splits['test']
        test_ds  = raw['test']
        text_field = 'text'

        # Align label column
        train_ds = train_ds.rename_column('label', 'labels')
        val_ds   = val_ds.rename_column('label', 'labels')
        test_ds  = test_ds.rename_column('label', 'labels')

    else:
        raise ValueError(f"Unknown task {task}")

    # --- Common tokenization and formatting ---
    def tokenize_fn(examples):
        return tokenizer(
            examples[text_field],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )

    # Apply tokenization
    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds   = val_ds.map(tokenize_fn, batched=True)
    test_ds  = test_ds.map(tokenize_fn, batched=True)

    # Remove raw text column and set format for PyTorch
    train_ds = train_ds.remove_columns([text_field])
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    val_ds = val_ds.remove_columns([text_field])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    test_ds = test_ds.remove_columns([text_field])
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    return train_ds, val_ds, test_ds
