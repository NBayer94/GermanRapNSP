import tensorflow
import config
import model
import data
import numpy as np
import streamlit as st


#@st.cache
def get_prediction_model(path):
    """Generates prediction model from specified path.

    Args:
        path (str): Path of model.

    Returns:
        tuple: Tuple of text_generator model, char_to_ind, and ind_to_char dicts.
    """
    lyrics = data.load_data(config.data_path, raw=False)

    char_to_ind, ind_to_char = data.get_dicts(lyrics)

    vocab_size = data.get_vocab_size(lyrics)
    embed_dim = config.hyperparameters['embed_dim']
    rnn_neurons = config.hyperparameters['rnn_neurons']
    batch_size = 1
    rnn_layers = config.hyperparameters['rnn_layers']
    dropout = config.hyperparameters['dropout']

    text_generator = model.create_model(vocab_size, embed_dim, rnn_neurons, batch_size, rnn_layers, dropout)
    text_generator.load_weights(path)

    return (text_generator, char_to_ind, ind_to_char)

#@st.cache
def generate_text(model, start_seed, gen_size=100, temp=1.0):

    """Generates text based on start seed. Requires model tuple from get_prediction_model method.

    Returns:
        str: Generated Text
    """

    text_generator, char_to_ind, ind_to_char = model

    # Transform seed to lowercase
    #start_seed = start_seed.lower()

    # Replace chars not included in dict with empty string
    for char in set(start_seed) - set(ind_to_char):
         start_seed.replace(char, '')

    # Transform char input to index values
    input_ids = np.array([char_to_ind[char] for char in start_seed])

    # Reset state of model
    text_generator.reset_states()

    # List for genereted text
    generatet_text = []

    for i in range(gen_size):
        
        # Log-odds prediction
        predictions = text_generator.predict(input_ids, batch_size=1)[0, -1, :].reshape((1, -1))
        
        # Use a cateogircal disitribution to select the next character
        predictions = predictions / temp
        predicted_id = tensorflow.random.categorical(predictions, num_samples=1).numpy().flatten()[0]

        # Update input id's
        input_ids = np.array([predicted_id])

        # Append predicted char to list
        generatet_text.append(ind_to_char[predicted_id])
    
    return start_seed + ''.join(generatet_text)