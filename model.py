import tensorflow
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, GRU

loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def create_model(vocab_size, embed_dim, rnn_neurons, batch_size, rnn_layers, dropout):

    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))
    
    # RNN layers
    for i in range(rnn_layers):
      model.add(LSTM(
        rnn_neurons, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
      if dropout:
        model.add(Dropout(0.2))

    # Dense prediction layer    
    model.add(Dense(vocab_size))

    model.compile(optimizer='adam', loss=loss)

    return model