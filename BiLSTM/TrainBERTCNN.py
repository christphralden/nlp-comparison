import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import h5py

# Define the dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        if self.tokenizer:
            inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            return {
                'text': text,
                'label': torch.tensor(label, dtype=torch.long)
            }

# Define the BERT + CNN model
class BERTCNN(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_classes=2, kernel_sizes=[2, 3, 4], num_filters=100):
        super(BERTCNN, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, 768)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state.unsqueeze(1)  # [batch_size, 1, seq_len, 768]
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [torch.max_pool1d(c, c.size(2)).squeeze(2) for c in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits

# Load data from CSV
df = pd.read_csv('./Reviews.csv')
df = df.head(500)
display(df)

texts = df['Text'].values
labels = df['Score'].values

# Initialize tokenizer and dataset for BERTCNN
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_dataset = SentimentDataset(texts, labels, tokenizer)
bert_dataloader = DataLoader(bert_dataset, batch_size=16, shuffle=True)

# Initialize model, loss function, and optimizer
bert_cnn_model = BERTCNN(num_classes=2)
criterion = nn.CrossEntropyLoss()
bert_optimizer = torch.optim.Adam(bert_cnn_model.parameters(), lr=2e-5)

bert_cnn_model.train()
for epoch in range(3):  # Number of epochs
    for batch in bert_dataloader:
        bert_optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        outputs = bert_cnn_model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        bert_optimizer.step()
    print(f'BERTCNN - Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the trained model to .h5 format
def save_model_to_h5(model, filepath):
    model_params = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    with h5py.File(filepath, 'w') as f:
        for k, v in model_params.items():
            f.create_dataset(k, data=v)


save_model_to_h5(bert_cnn_model, 'bert_cnn_model.h5')