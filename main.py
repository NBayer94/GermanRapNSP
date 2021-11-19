import data
import train
import model
import config

######
#Data#
######
lyrics = data.load_data('lyrics_raw.pkl', raw=False)

char_to_ind, ind_to_char = data.get_dicts(lyrics)

vocab_size = data.get_vocab_size(lyrics)

encoded_lyrics = data.convert_lyrics(lyrics, char_to_ind, char_to_ind=True)


#######
#Train#
#######
tf_dataset = train.create_tf_dataset(encoded_lyrics)

sequences = train.create_sequences(tf_dataset)

dataset = train.create_dataset(sequences)

dataset = train.create_train_batches(dataset)

#######
#Model#
#######
embed_dim = config.hyperparameters['embed_dim']
rnn_neurons = config.hyperparameters['rnn_neurons']
batch_size = config.hyperparameters['batch_size']

model = model.create_model(vocab_size, embed_dim, rnn_neurons, batch_size)

train.train_model(model, dataset)

train.save_model(model, 'model_gru.h5')

