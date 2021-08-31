import torch
import torch.nn as nn
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup


# Create the BertClassfier class
class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertClassifier, self).__init__()

        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        # D_in, H, D_out = 768, 50, 3
        D_in, H, D_out = 2304, 50, 3 # 768*3=2304

        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert.resize_token_embeddings(21130) # 21130 is length of tokenizer_chi, after adding special tokens

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out),
            # nn.ReLU(),
            nn.Sigmoid(),
        )

        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):

        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        # last_hidden_state_cls = outputs[0][:, 0, :]
        concat_hidden_state_cls = torch.cat((outputs[0][:, 0, :], outputs[0][:, 1, :], outputs[0][:, 2, :]), 1)
        # print(concat_hidden_state_cls.shape)  # [16, 2304]

        # Feed input to classifier to compute logits
        # logits = self.classifier(last_hidden_state_cls)
        logits = self.classifier(concat_hidden_state_cls)

        return logits


def initialize_model(train_dataloader, device, epochs=4):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = BertClassifier(freeze_bert=False)

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)

    return bert_classifier, optimizer, scheduler
