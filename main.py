import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader,random_split
from transformers import BertTokenizer,BertForTokenClassification, AdamW, get_scheduler,BertForMaskedLM
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

# Load original data twitter data
data_path = r'F:\code\NLP\labeled_data.csv'
data = pd.read_csv(data_path)
# Load dirty word dataset
profanity_path =  r'F:\code\NLP\en.txt'
with open(profanity_path, 'r') as file:
    profanity_list = set([line.strip().lower() for line in file if line.strip()])
# check some of the dirty word
print(list(profanity_list)[:10])


# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def preprocess_text(text):
    # remove spave from begining and ending
    return text.lower().strip()
# label dirty word funtion
def label_text(text, tokenizer, profanity_list):
   # lable dirty words as 1, others are 0
    cleaned_text = preprocess_text(text)
    tokens = tokenizer.tokenize(cleaned_text)
    labels = [1 if token in profanity_list else 0 for token in tokens]
    return tokens, labels

class ProfanityDataset(Dataset):
    #using previous rules labels the data
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels = [label_text(text, tokenizer, profanity_list)[1] for text in texts]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens, labels = label_text(text, tokenizer, profanity_list)
        encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_len)
        labels += [0] * (self.max_len - len(labels))
        labels = torch.tensor(labels[:self.max_len])

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels
        }
# test lablingling 
example_text = "What the fuck did you bitch say, stupid?" # change this sentance if other sentance needs to be tested
tokens, labels = label_text(example_text, tokenizer, profanity_list)
print(list(zip(tokens, labels)))    

#tesitng the exaample text  with encoding and tokenizer
texts = [example_text]  
dataset = ProfanityDataset(texts, tokenizer, max_len=128)

# encoding result
sample_encoding = dataset[0]
print("Input IDs:", sample_encoding['input_ids'])

def compute_accuracy(outputs, labels):
    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(outputs, dim=-1)
    # Get the most likely label (class 1 if binary classification)
    predictions = torch.argmax(probabilities, dim=-1)
    # Calculate how many predictions match the labels
    correct = (predictions == labels).float()
    # Calculate the accuracy across all predictions in the batch
    accuracy = correct.sum() / correct.numel()
    return accuracy

# GPU avaliable check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model chosing
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# prepare the data and pre process
dataset = ProfanityDataset(data['tweet'], tokenizer)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# optimizer and learning rate setting 
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 10
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

train_losses = []
val_losses = []
best_val_loss = float('inf')

train_accuracies = []
val_accuracies = []
# training 
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        loss = outputs.loss
        accuracy = compute_accuracy(outputs.logits, batch['labels'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        #loss calculation
        total_loss += loss.item()
        total_accuracy += accuracy.item()

    avg_train_loss = total_loss / len(train_loader)
    avg_train_accuracy = total_accuracy / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)
    print(f"Epoch {epoch+1}, Average Training loss: {avg_train_loss}, Average Training Accuracy: {avg_train_accuracy}")

    # validation
    model.eval()
    total_eval_loss = 0
    total_eval_accuracy = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            accuracy = compute_accuracy(outputs.logits, batch['labels'])

            total_eval_loss += loss.item()
            total_eval_accuracy += accuracy.item()

    avg_val_loss = total_eval_loss / len(val_loader)
    avg_val_accuracy = total_eval_accuracy / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)
    print(f"Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")

    # save the best model for Masked model input
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

# plot of loss 
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# plot of accuracy 
plt.figure(figsize=(10, 5))
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracies')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

print("Training complete!")

# tesing
sentence = example_text

# preprocessing the test sentance
inputs = tokenizer(sentence, return_tensors="pt")
input_ids = inputs['input_ids'].to(device)

# Masked LM prediction
model.eval() 
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

# get the highest preditoin mark
predictions = torch.argmax(logits, dim=-1)

# get the predicted word
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
predicted_labels = predictions.squeeze().tolist()

# print the word and labels
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {'Dirty' if label == 1 else 'not dirty'}")


from transformers import BertTokenizer, BertForTokenClassification, BertForMaskedLM

# load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#load BERT for masekd LM
model_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased')
# check if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load classification to GPU
model.to(device)
# load masked to GPU
model_mlm.to(device)

# example sentance
sentence = example_text

# using classification modle detect dirty words
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
input_ids = inputs['input_ids'].to(device)

model.eval()
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# removel special indecate CLS and SEP
sensitive_indices = (predictions.squeeze()[1:-1] == 1).nonzero(as_tuple=True)[0].tolist()

# Using mased LM predicting 
tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
for idx in sensitive_indices:
    mask_index = idx + 1  
    original_token = tokens[mask_index]
    tokens[mask_index] = predicted_token
    masked_input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([masked_input_ids]).to(device)


    model_mlm.eval()
    with torch.no_grad():
        outputs = model_mlm(input_tensor)
        predictions = outputs.logits

    # predtion of masked words
    predicted_index = torch.argmax(predictions[0, mask_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(f"Original Token: {original_token} -> New Token: {predicted_token}")
    
    # Replace the original token with new token
    tokens[mask_index] = predicted_token

# print the result
new_sentence = tokenizer.convert_tokens_to_string(tokens)
print("Original:", sentence)
print("Modified:", new_sentence)