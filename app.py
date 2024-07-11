import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
import fastapi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI
app = FastAPI()

# Define the RNN model class
class SMSRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SMSRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = torch.sigmoid(self.fc(out))
        return out

# Define the data schema
class Message(BaseModel):
    text: str

# Training function
def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = pd.read_csv('scam7.csv', encoding='utf-8')
    data['v2'].fillna('', inplace=True)
    data['v2'] = data['v2'].apply(lambda x: f"System: This message needs to be classified as scam or ham. Message: {x}")
    X = data['v2']
    y = data['v1'].apply(lambda x: 1 if x == 'scam' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    X_train_tensor = torch.tensor(X_train_vectorized.toarray(), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_vectorized.toarray(), dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    input_size = X_train_vectorized.shape[1]
    hidden_size = 128
    num_layers = 2
    model = SMSRNN(input_size, hidden_size, num_layers).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    epochs = 50

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_inputs = X_test_tensor.unsqueeze(1)
        outputs = model(val_inputs).squeeze()
        accuracy = (outputs > 0.5).float().eq(y_test_tensor).sum().item() / y_test_tensor.size(0)
        print(f'Validation Accuracy: {accuracy:.4f}')

    return model, vectorizer

# Train the model and vectorizer at startup
model, vectorizer = train_model()

# Prediction functions
def preprocess_message(message, vectorizer):
    system_message = f"System: This message needs to be classified as scam or ham. Message: {message}"
    message_vectorized = vectorizer.transform([system_message])
    message_tensor = torch.tensor(message_vectorized.toarray(), dtype=torch.float32).unsqueeze(1).to(device)
    return message_tensor

def predict_message(model, message_tensor):
    model.eval()
    with torch.no_grad():
        output = model(message_tensor)
        prediction = (output > 0.5).float().item()
    return prediction

def interpret_prediction(prediction):
    return 'scam' if prediction == 1 else 'ham'

@app.post("/predict/")
def predict(msg: Message):
    try:
        message_tensor = preprocess_message(msg.text, vectorizer)
        prediction = predict_message(model, message_tensor)
        result = interpret_prediction(prediction)
        return {"message": msg.text, "prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
