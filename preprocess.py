import re
import torch
from transformers import BertTokenizer

MAX_LEN = 300

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def text_preprocessing(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'[0-9]+' , '' ,text)
    text = re.sub(r'\s([@][\w_-]+)', '', text).strip()
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("#" , " ")
    encoded_string = text.encode("ascii", "ignore")
    decode_string = encoded_string.decode()
    
    return decode_string

def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text = text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens = True,        # Add `[CLS]` and `[SEP]`
            max_length = MAX_LEN,             # Max length to truncate/pad
            pad_to_max_length = True,         # Pad sentence to max length
            #return_tensors='pt',             # Return PyTorch tensor
            truncation = True,
            return_attention_mask = True      # Return attention mask
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks
