# TESTING PURPOSES USE THE JUPYTER NOTEBOOK VERSION

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from tqdm import tqdm

TRAIN = False # Toggle

df = pd.read_csv('./dataset/Reviews.csv')
df = df.head(10000)


def map_score_to_sentiment(score):
    if score < 3:
        return 0  # Negative
    elif score == 3:
        return 1  # Neutral
    else:
        return 2  # Positive


df['Sentiment'] = df['Score'].apply(map_score_to_sentiment)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Text'], df['Sentiment'], test_size=0.9, random_state=42)

train_df = pd.DataFrame({'text': train_texts, 'label': train_labels})
test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

MODEL_TYPE = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_TYPE, num_labels=3)


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

if(TRAIN):
    trainer.train()

try:
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
except:
    print("fuck")

test_df['predicted'] = preds

predicted_negative = []
predicted_neutral = []
predicted_positive = []

false_negative = []
false_neutral = []
false_positive = []

true_negative = []
true_neutral = []
true_positive = []

for idx in range(len(test_df)):
    score = test_df['label'].iloc[idx]
    result = {
        'roberta_neg': predictions.predictions[idx][0],
        'roberta_neu': predictions.predictions[idx][1],
        'roberta_pos': predictions.predictions[idx][2]
    }

    if score == 0:  # fr negative
        true_negative.append(result)
    elif score == 1:  # fr neutral
        true_neutral.append(result)
    elif score == 2:  # fr positive
        true_positive.append(result)

    if result['roberta_neg'] > result['roberta_pos'] and result['roberta_neg'] > result['roberta_neu']:  # predicted negative
        predicted_negative.append(result)
        if score > 0:  # false negatives
            false_negative.append(result)
    elif result['roberta_neu'] > result['roberta_neg'] and result['roberta_neu'] > result['roberta_pos']:  # predicted neutral
        predicted_neutral.append(result)
        if score != 1:
            false_neutral.append(result)
    elif result['roberta_pos'] > result['roberta_neg'] and result['roberta_pos'] > result['roberta_neu']:  # predicted positive
        predicted_positive.append(result)
        if score < 2:
            false_positive.append(result)


def plot_all_comparisons(true_negatives, predicted_negatives, false_negatives, true_positives, predicted_positives, false_positives, true_neutrals, predicted_neutrals, false_neutrals):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    data = [
        (true_negatives, predicted_negatives, false_negatives, 'Negatives'),
        (true_positives, predicted_positives, false_positives, 'Positives'),
        (true_neutrals, predicted_neutrals, false_neutrals, 'Neutrals')
    ]

    colors = ['blue', 'green', 'red']

    for ax, (true_data, predicted_data, false_prediction, title) in zip(axes, data):
        categories = ['True ' + title, 'Predicted ' + title, 'False ' + title]
        counts = [len(true_data), len(predicted_data), len(false_prediction)]
        ax.bar(categories, counts, color=colors)
        ax.set_title('True vs Predicted vs False ' + title)
        ax.set_xlabel('Category')
        ax.set_ylabel('Count')
        ax.set_ylim(0, max(counts) + 10)

    plt.tight_layout()
    plt.show()


plot_all_comparisons(true_negative, predicted_negative, false_negative, true_positive, predicted_positive, false_positive, true_neutral, predicted_neutral, false_neutral)
