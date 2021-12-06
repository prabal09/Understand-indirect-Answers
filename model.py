import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers import AutoConfig, AutoTokenizer
class Bert(BertPreTrainedModel):
    def __init__(self, config,n_classes):
        super().__init__(config)
        self.bert = BertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = nn.Linear(config.hidden_size,n_classes)

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        # pdb.set_trace()
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        # print(output)
        output,index = torch.max(output, dim=1)
        return output,index

  # def __init__(self, n_classes: int):
  #   super().__init__()
  #   self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
  #   self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
  #   self.n_training_steps = n_training_steps
  #   self.n_warmup_steps = n_warmup_steps
  #   self.criterion = nn.BCELoss()
  # def forward(self, input_ids, attention_mask, labels=None):
  #   output = self.bert(input_ids, attention_mask=attention_mask)
  #   output = self.classifier(output.pooler_output)
  #   output = torch.sigmoid(output)
