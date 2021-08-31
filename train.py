import torch
import torch.nn as nn
import time
import numpy as np

# Specify loss function
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.MultiLabelMarginLoss()
loss_fn = nn.BCELoss()


# def loss_function(input_tensor, output_tensor):

#     assert input_tensor.size() == output_tensor.size()

#     row_size = input_tensor.size()[0]
#     column_size = output_tensor.size()[1]

#     loss = 0
#     for i in range(row_size):
#         for j in range(column_size):
#             if output_tensor[i][j] == 0:
#                 loss += input_tensor[i][j] - 0
#             else:
#                 loss += 1 - input_tensor[i][j]

#     return torch.tensor(loss)


def evaluate(model, val_dataloader, device):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode.
    # The dropout layers are disabled during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        b_labels = b_labels.float()
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        # preds = torch.argmax(logits, dim=1).flatten()
        preds = []
        for idx, logit in enumerate(logits):
            pred = [1 if l>=0.5 else 0 for l in logit]
            preds.append(pred)

        # Calculate the accuracy rate
        # accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        b_labels_numpy = b_labels.cpu().detach().numpy()
        correct = 0
        for idx in range(len(preds)):
            if (preds[idx] == b_labels_numpy[idx]).all() == True:
                correct += 1
        accuracy = correct / len(preds) * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def train(model, train_dataloader, device, optimizer, scheduler, val_dataloader=None, epochs=4, evaluation=False):
    """Train the BertClassifier model.
    """
    
    cur_highest_val_acc = 0
    
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1

            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model.forward(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            b_labels = b_labels.float()

            loss = loss_fn(logits, b_labels)
            # loss = loss_function(logits, b_labels)

            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 1 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation is True:
            # After the completion of each training epoch,
            # measure the model's performance on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader, device)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)

            if val_accuracy > cur_highest_val_acc:
                print('Save model')
                torch.save(model, './saved_model/chi_model.pt')

        print("\n")

    print("Training complete!")

    return model, val_accuracy
