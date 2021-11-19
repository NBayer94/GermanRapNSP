import pickle
import numpy as np
import tensorflow as tf

def preprocess(lyrics):
    '''
    Converts list of lyrics to one lowercase string and replaces unnecessary chars
    '''
    # Combine lyrics in one big string
    lyrics_complete = ''

    for lyric in lyrics:
        lyrics_complete = lyrics_complete + lyric

    # transform to lower case
    lyrics_complete = lyrics_complete.lower()

    # Replace unneccessary chars
    str_to_replace = ['\u2005', '\u200b', '–', '—', '‘', '’', '‚', '“', '”', '„', '•', '…', '′', '″', '−', '\ufeff', '«',
    '\xad', '´', '»', '%', '&', "'", '(', ')', '*', '+', '\n', ':', ';', '[', '\u205f', '/', '-', '_', '"', '#', '€', '$',
    '¡', 'ê', 'ð', 'ó', 'ú', 'ī', 'ş', 'а', 'б', 'г', 'д', 'и', 'й', 'к', 'л', 'м', 'о', 'р', 'с', 'у', 'ь', 'я']

    for repl in str_to_replace:
        lyrics_complete = lyrics_complete.replace(repl, ' ')

    return lyrics_complete

def load_data(path, raw=True):
    '''
    Loads lyric data and preprocesses if selected
    '''
    # Load data
    with open(path, 'rb') as f:
            lyrics = pickle.load(f)
    if raw:
        return lyrics
    else:
        return preprocess(lyrics)

def get_dicts(lyrics):
    '''
    Computes conversion dictionaries for index <--> char
    '''
    vocab = sorted(set(lyrics))

    char_to_ind = {char: i for i, char in enumerate(vocab)}
    ind_to_char = np.array(vocab)
    return char_to_ind, ind_to_char

def get_vocab_size(lyrics):
    '''
    Computes vocab size of lyrics
    '''
    vocab_size = len(set(lyrics))
    return vocab_size

def convert_lyrics(lyrics, conversion_dict, char_to_ind=True):
    '''
    Converts lyrics from char to int or other way around
    '''
    if char_to_ind:
        return np.array([conversion_dict[char] for char in lyrics])
    else:
        return np.array([conversion_dict[ind] for ind in lyrics])