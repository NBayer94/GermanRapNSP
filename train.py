import numpy as np
import config
import tensorflow as tf

def create_tf_dataset(lyrics):
    """Creates tf dataset from lyrics string

    Args:
        lyrics (str): lyrics string

    Returns:
        tf.Dataset: tf Dataset
    """
    return tf.data.Dataset.from_tensor_slices(lyrics)

def create_sequences(data):
    """Creates batches from tf dataset

    Args:
        data (tf.Dataset): tf Dataset

    Returns:
        batches
    """
    sequences = data.batch(config.seq_length+1, drop_remainder=True)
    return sequences

def create_seq_targets(sequence):
    """Create input and target sequences

    Args:
        seq (batch): [description]

    Returns:
        [type]: [description]
    """
    input_txt = sequence[:-1]
    target_txt = sequence[1:]
    return input_txt, target_txt


def create_dataset(sequences):
    """Creates final training dataset

    Args:
        batches (tf.Dataset): [description]

    Returns:
        [type]: [description]
    """
    dataset = sequences.map(create_seq_targets)
    return dataset

def create_train_batches(dataset):
    dataset = dataset.cache().shuffle(buffer_size=10000).batch(
        config.hyperparameters['batch_size'], drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def train_model(model, dataset):
    epochs = config.hyperparameters['epochs']
    model.fit(dataset, epochs=epochs)

def save_model(model, path):
    model.save(path)

