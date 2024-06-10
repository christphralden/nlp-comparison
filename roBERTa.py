from transformers import AutoModelForSequenceClassification

class RoBERTa:
    def __init__(self, model_type='cardiffnlp/twitter-roberta-base-sentiment', num_labels=3):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=num_labels)
    
    def get_model(self):
        return self.model