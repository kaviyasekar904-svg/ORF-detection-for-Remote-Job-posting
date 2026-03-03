import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel

app = Flask(__name__)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("tokenizer")

# Model class
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

# Load trained model
model = BERT_LSTM()
model.load_state_dict(torch.load("model.pth"))
model.eval()
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator=" ")
        return text

    except:
        return None

def predict(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        prediction = torch.argmax(outputs, dim=1).item()

    if prediction == 1:
        return "🚨 FAKE JOB"
    else:
        return "✅ REAL JOB"

@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/predict", methods=["POST"])
def predict_route():
    url = request.form["url"]

    if url.startswith("http"):
        extracted_text = extract_text_from_url(url)

        if extracted_text:
            result = predict(extracted_text)
        else:
            result = "⚠ Unable to fetch website content"
    else:
        result = predict(url)

    return render_template("index.html", prediction=result)
    


if __name__ == "__main__":
    print("Starting Flask Server...")
    app.run(debug=True)