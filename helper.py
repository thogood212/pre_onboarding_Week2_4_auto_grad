import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, random_split
import os
import sys
import pandas as pd
import numpy as np 
import random
from torch.utils.data.dataset import random_split
from collections import Counter
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

def set_device():
  if torch.cuda.is_available():
    return torch.device("cuda")
  else:
    return torch.device("cpu")

def load_clean_data():
  !wget https://raw.githubusercontent.com/ChristinaROK/PreOnboarding_AI_assets/e56006adfac42f8a2975db0ebbe60eacbe1c6b11/data/sample_df.csv
  dataset = pd.read_csv('sample_df.csv')
  dataset = CustomDataset(list(dataset['document']), list(dataset['label']))
  return dataset


def label_evenly_balanced_dataset_sampler(dataset,train_ratio,valid_ratio):
  n_train = int(len(dataset)*train_ratio)
  n_valid = int(len(dataset)*valid_ratio)
  train_dataset, valid_dataset = random_split(dataset, [n_train,n_valid])
  return train_dataset, valid_dataset

train_dataset, valid_dataset = label_evenly_balanced_dataset_sampler(dataset,0.9,0.1)

def custom_collate_fn(batch):
  tokenizer_bert = BertTokenizer.from_pretrained("klue/bert-base")
  
  batch=list(batch)

  input_list=[]
  target_list=[]
  
  for k,v in batch:
    input_list.append(k)
    target_list.append(v)
  
  tensorized_input = tokenizer_bert(
        text = input_list,
        padding='longest',
        truncation = True,
        return_tensors = 'pt'
        )
  
  tensorized_label = torch.tensor(target_list)
  
  return (tensorized_input, tensorized_label)



class CustomDataset(Dataset):
  """
  - input_data: list of string
  - target_data: list of int
  """

  def __init__(self, input_data:list, target_data:list):
      self.X = input_data
      self.Y = target_data

  def __len__(self):
      return len(self.Y)

  def __getitem__(self, index):
      return self.X[index], self.Y[index]


class CustomClassifier(nn.Module):

  def __init__(self, hidden_size: int, n_label: int):
    super(CustomClassifier, self).__init__()

    self.bert = BertModel.from_pretrained("klue/bert-base")

    dropout_rate = 0.1
    linear_layer_hidden_size = 32

    self.classifier = nn.Sequential(
          nn.Linear(hidden_size, linear_layer_hidden_size),
          nn.ReLU(),
          nn.Dropout(dropout_rate),
          nn.Linear(linear_layer_hidden_size,n_label)
        ) # torch.nn에서 제공되는 Sequential, Linear, ReLU, Dropout 함수 활용
  
  def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
    )

    # BERT 모델의 마지막 레이어의 첫번재 토큰을 인덱싱
    cls_token_last_hidden_states = outputs['pooler_output'] # 마지막 layer의 첫 번째 토큰 ("[CLS]") 벡터를 가져오기, shape = (1, hidden_size)

    logits = self.classifier(cls_token_last_hidden_states)

    return logits