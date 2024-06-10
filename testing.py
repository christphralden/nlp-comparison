import pandas as pd
import numpy as np
from transformers import AutoTokenizer, Trainer
from datasets import Dataset
from RoBERTa import RoBERTa
from BERTCNN import BERTCNN
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv('./dataset/Reviews.csv')
    df = df.head(10000)
    df['Sentiment'] = df['Score'].apply(lambda x: 0 if x < 3 else 1 if x == 3 else 2)
    
    _, test_texts, _, test_labels = train_test_split(
        df['Text'], df['Sentiment'], test_size=0.2, random_state=42)
    
    test_df = pd.DataFrame({'text': test_texts, 'label': test_labels})
    
    return test_df

def tokenize_data(model_type, test_df):
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    
    test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)
    
    return test_dataset, tokenizer

def test_model(model, test_dataset, tokenizer):
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    return preds

if __name__ == "__main__":
    test_df = load_data()
    
    # BERT-CNN
    bert_model_type = 'bert-base-uncased'
    test_dataset, tokenizer = tokenize_data(bert_model_type, test_df)
    bert_cnn_model = BERTCNN(model_type=bert_model_type)
    bert_cnn_preds = test_model(bert_cnn_model, test_dataset, tokenizer)
    
    # Custom RoBERTa
    roberta_model_type = 'cardiffnlp/twitter-roberta-base-sentiment'
    test_dataset, tokenizer = tokenize_data(roberta_model_type, test_df)
    roberta_model = RoBERTa(model_type=roberta_model_type)
    roberta_preds = test_model(roberta_model, test_dataset, tokenizer)
    
    # Save predictions
    test_df['bert_cnn_predicted'] = bert_cnn_preds
    test_df['roberta_predicted'] = roberta_preds
    test_df.to_csv('test_predictions.csv', index=False)