import tensorflow
from tensorflow.keras import models
import config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, GRU

def sparse_cat_loss(y_true,y_pred):
  return tensorflow.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):

    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))
    model.add(GRU(
        rnn_neurons, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=sparse_cat_loss)

    return model