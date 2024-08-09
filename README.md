# Instructions
## Preparing Data
- Unzip the file `data/bitcoin_historical_dataset.zip`
- Move the unzipped files into `data/original_data/BTC-2017min.csv` and `data/original_data/BTC-2018min.csv`
## Processing Data
- Run the following script `src/feature_engineering.py` this will generate files into the `data/modified_data/` folder.
## Training Model
- Run the following script `src/create_neural_network.py` this will train a LSTM neural network model using PyTorch on bitcoin data from 2017.
- This will create a saved model file `models/model.pth`
## Deploy Model
- To test the model load the saved model file and turn it on evaluation mode use a new data set. The script `src/use_neural_network.py` use the 2018 bitcoin data to evaluate the model.
- This displays the Mean Squared Error, Mean Absolute Error, R-squared values for the models performance on new data.