import csv
from datetime import datetime, timedelta
import pandas as pd

hist_input_data_file = 'data/original_data/BTC-2018min.csv'
hist_output_data_file = 'data/modified_data/BTC-2018min_modified.csv'




df = pd.read_csv(hist_input_data_file)

# Convert 'date' to datetime if not already
df['date'] = pd.to_datetime(df['date'])

# Set date_time as the index so that time-based operations are easier.
df.set_index('date', inplace=True)


# Columns containing previous prices
# negative shift is down
# positive shift is up
df['price_5_hours_past'] = df['close'].shift(60*-5) # 5 hours past
df['price_2_hours_past'] = df['close'].shift(60*-2)
df['price_1_hours_past'] = df['close'].shift(60*-1)

# Column for future price
df['price_1_hours_future'] = df['close'].shift(60*1) # 1 hours future


# Drop the first few hours of data where there is not a old price
df = df.dropna(subset=['price_5_hours_past', 'price_1_hours_future'])

# Calculate the percentage change for each duration
df['pct_chg_5p'] = (df['close'] - df['price_5_hours_past']) / df['price_5_hours_past']
df['pct_chg_2p'] = (df['close'] - df['price_2_hours_past']) / df['price_2_hours_past']
df['pct_chg_1p'] = (df['close'] - df['price_1_hours_past']) / df['price_1_hours_past']

df['pct_chg_1f'] = (df['price_1_hours_future'] - df['close']) / df['close']

df.reset_index(inplace=True)
df.to_csv(hist_output_data_file, index=False)