from transformers import BertModel, RobertaModel, XLNetModel
import torch
from torch import nn


class SentimentAnalyzer(nn.Module):
    def __init__(self,
                 model_name,
                 num_class=5,
                 num_other_features=3,
                 hidden_size=10,
                 dropout_rate=0.3,
                 use_pooled=True):
        super().__init__()
        self.use_pooled = use_pooled
        if "roberta" in model_name:
            transformer_base = RobertaModel
        elif "bert" in model_name:
            transformer_base = BertModel
        elif "xlnet" in model_name:
            transformer_base = XLNetModel
            self.use_pooled = False  # no pooler for xlnet
        self.transformer = transformer_base.from_pretrained(model_name)
        if not self.use_pooled:
            self.hidden = nn.Linear(self.transformer.config.hidden_size,
                                    self.transformer.config.hidden_size)
        self.fc1 = nn.Linear(num_other_features, hidden_size)
        self.classifier = nn.Linear(self.transformer.config.hidden_size + hidden_size,
                                    num_class)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, input_ids, attention_mask, other_features):
        transformer_out = self.transformer(input_ids=input_ids,
                                           attention_mask=attention_mask)
        if self.use_pooled:
            output = transformer_out["pooler_output"]
        else:
            cls_token = transformer_out["last_hidden_state"][:, 0]  # get the [CLS] token
            output = self.hidden(cls_token)
        dropped = self.dropout(output)  # [batch_size, 768]

        feat = self.fc1(other_features)  # [batch_size, num_other_features]

        final = torch.cat([dropped, feat], axis=1)
        return self.classifier(final)