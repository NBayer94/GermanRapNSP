import tensorflow
import config
import model
import numpy as np


def generate_text(model, start_seed, char_to_ind, ind_to_char, gen_size=100, temp=1.0):
    
    # Transform char input to index values
    input_ids = np.array([char_to_ind[char] for char in start_seed])

    # Reset state of model
    model.reset_states()

    # List for genereted text
    generatet_text = []

    for i in range(gen_size):
        
        # Log-odds prediction
        predictions = model.predict(input_ids, batch_size=1)[0, -1, :].reshape((1, -1))
        
        # Use a cateogircal disitribution to select the next character
        predictions = predictions / temp
        predicted_id = tensorflow.random.categorical(predictions, num_samples=1).numpy().flatten()[0]

        # Update input id's
        input_ids = np.array([predicted_id])

        # Append predicted char to list
        generatet_text.append(ind_to_char[predicted_id])
    
    return start_seed + ''.join(generatet_text)