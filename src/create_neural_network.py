import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

data_file = 'data/modified_data/BTC-2017min_modified.csv'
output_model_file = 'models/test_model.pth'

features = ['pct_chg_5p', 'pct_chg_2p', 'pct_chg_1p']
target = 'pct_chg_1f'

data = pd.read_csv(data_file)

# drop target features and redundant features and date column since we already have unix timestamps
# X = data.drop(columns=['price_1_hours_future', 'pct_chg_1f', 'symbol', 'date']).values
X = data[features]
y = data[target].values


# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Conver to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Check if there are NaNs in features and targets
print(torch.isnan(X_tensor).any())  
print(torch.isnan(y_tensor).any()) 


# Split data
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, shuffle=False)

# Adding sequence dimension
X_train = X_train.unsqueeze(1)  
X_test = X_test.unsqueeze(1)  


print(X_train.shape)
print(X_test.shape) 




# Define neural network
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    




# Initialize the model, loss function, and optimizer
input_size = X_train.shape[2]
print(input_size)
hidden_size = 50
output_size = 1



model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1} / {num_epochs}], Loss: {loss.item():.4f}')


# Evaluate model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), output_model_file)