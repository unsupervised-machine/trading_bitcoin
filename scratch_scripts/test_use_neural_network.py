import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

input_model_file = 'models/test_model.pth'
input_size = 3
hidden_size = 50
output_size = 1


input_data_file = 'data/modified_data/BTC-2018min_modified.csv'
output_data_file = 'data/with_preds/BTC-2018min_with_preds.csv'
features = ['pct_chg_5p', 'pct_chg_2p', 'pct_chg_1p']
target = 'pct_chg_1f'



data = pd.read_csv(input_data_file)

X_new = data[features]
y_true = data[target].values

scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)

X_new_tensor = torch.tensor(X_new, dtype=torch.float32).unsqueeze(1)
y_true_tensor = torch.tensor(y_true, dtype=torch.float32).view(-1, 1)


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
    

# Initialize the model
model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load(input_model_file))
model.eval()


# Make predictions
with torch.no_grad():
    y_pred_tensor = model(X_new_tensor)

print(y_pred_tensor)

# Convert predictions to numpy array (if needed)
y_pred = y_pred_tensor.numpy().flatten()

# Add predictions to a new column and save df
data['preds'] = y_pred
data['signal'] = data['preds'].apply(lambda x: 'hold' if -0.01 <= x <= 0.01 else ('buy' if x > 0.01 else 'sell'))
data['pred_true_abs_diff'] = abs(data['preds'] - data[target])
data.to_csv(output_data_file, index=False)



# Calculate metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'Mean Absolute Error (MAE): {mae:.4f}')
print(f'R-squared: {r2:.4f}')