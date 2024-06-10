import torch
from transformers import BertModel, BertPreTrainedModel
from torch import nn

class BERTCNN(BertPreTrainedModel):
    def __init__(self, config):
        super(BERTCNN, self).__init__(config)
        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config)
        self.conv = nn.Conv1d(in_channels=config.hidden_size, out_channels=128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        sequence_output = sequence_output.permute(0, 2, 1)
        x = self.conv(sequence_output)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else logits