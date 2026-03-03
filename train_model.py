import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("data/remote_jobs.csv")

X = df["text"]
y = df["fraudulent"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training size:", len(X_train))
print("Testing size:", len(X_test))

# ==============================
# CLASS WEIGHTS
# ==============================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = torch.tensor(class_weights, dtype=torch.float)
print("Class Weights:", class_weights)

# ==============================
# TOKENIZER
# ==============================

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def encode_text(text_list):
    return tokenizer(
        text_list.tolist(),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

train_encodings = encode_text(X_train)
train_labels = torch.tensor(y_train.values)

# ==============================
# MODEL
# ==============================

class BERT_LSTM(nn.Module):
    def __init__(self):
        super(BERT_LSTM, self).__init__()

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        lstm_out, _ = self.lstm(sequence_output)

        final_output = lstm_out[:, -1, :]
        output = self.dropout(final_output)

        logits = self.fc(output)

        return logits

# ==============================
# TRAINING
# ==============================

from torch.utils.data import TensorDataset, DataLoader

# Create Dataset
dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_labels
)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = BERT_LSTM()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss(weight=class_weights)

print("Starting Training...")

model.train()

for epoch in range(3):

    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = batch

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")

print("Training Completed!")

torch.save(model.state_dict(), "model.pth")
tokenizer.save_pretrained("tokenizer")

print("Model saved successfully!")