from six import b
from tensorflow.python.data.util import options
import data
import train
import model
import config
import numpy as np
from tensorflow.keras.models import load_model

embed_dim = config.hyperparameters['embed_dim']
rnn_neurons = config.hyperparameters['rnn_neurons']
batch_size = 1

text_generator = model.create_model(60, embed_dim, rnn_neurons, batch_size)
text_generator.load_weights('model.h5')

print(text_generator.predict(np.array([4]), batch_size=1).reshape((0, -1)))