import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, BertConfig
from sklearn.model_selection import train_test_split
from datasets import Dataset
from BERTCNN import BERTCNN
from RoBERTa import RoBERTa
import torch

def load_data():
    df = pd.read_csv('Reviews.csv', on_bad_lines='warn', nrows=10000)
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

    return train_df, test_df


def tokenize_data(model_type, train_df, test_df):
    tokenizer = AutoTokenizer.from_pretrained(model_type)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

    return train_dataset, test_dataset, tokenizer

def train_model(model, train_dataset, test_dataset, tokenizer, output_dir):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=2,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator
    )
    
    trainer.train()

if __name__ == "__main__":

    train_df, test_df = load_data()
    
    # BERT-CNN
    bert_model_type = 'bert-base-uncased'
    config = BertConfig.from_pretrained(bert_model_type, num_labels=3)
    train_dataset, test_dataset, tokenizer = tokenize_data(bert_model_type, train_df, test_df)
    bert_cnn_model = BERTCNN(config=config)
    train_model(bert_cnn_model, train_dataset, test_dataset, tokenizer, './bert_cnn_results')
    
    # Custom RoBERTa
    roberta_model_type = 'cardiffnlp/twitter-roberta-base-sentiment'
    train_dataset, test_dataset, tokenizer = tokenize_data(roberta_model_type, train_df, test_df)
    roberta = RoBERTa(model_type=roberta_model_type)
    roberta_model = roberta.get_model()
    train_model(roberta_model, train_dataset, test_dataset, tokenizer, './roberta_results')