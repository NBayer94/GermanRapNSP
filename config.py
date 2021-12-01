#data path
data_path = 'lyrics_raw.pkl'

# Sequence length
seq_length = 150

# Model hyperparameters
hyperparameters = {
    'batch_size': 128,
    'embed_dim': 64,
    'rnn_neurons': 512,
    'rnn_layers': 2,
    'dropout': True,
    'epochs': 64
}

# Model path
model_path = 'models/model_incl_uppercase.h5'