
import string
import re

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint


import numpy as np

import sys
print sys.argv

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model








print("Please enter some seed text to feed to the model: ")
# my name is what you
input_text = raw_input()
print("Input text: ",input_text)

input_tokens = str(input_text).split()
# print(input_tokens)



filename = 'data/shakespeare2.txt'
content = open(filename, 'r')
data = content.read()
content.close()

print("Total characters in the text file: ",len(data))
print("First 20 characters: ",data[:20])

tokens = data.split()
print("Total tokens in the text file",len(tokens))
print("First 20 tokens from the text file: ",tokens[:20])





sentence_len = 10


## Removing punctuations from the text file ==========================
exclude = set(string.punctuation)
# print(exclude)
text = ''.join(ch for ch in data if ch not in exclude)

# print(text[:20])
tokens = text.lower().split()
# print(tokens[:20])
# print(len(tokens))




## Converting text to appropriate input shape for the first model==========================
lines = []
for i in range(len(tokens)-sentence_len):
	line = tokens[i:i+sentence_len]
	lines.append(line)
# print(">>"*10)
# print(lines[:5])	



tokenizer = Tokenizer(oov_token='UNK')
tokenizer.fit_on_texts(lines)
text = tokenizer.texts_to_sequences(lines)
# print(text[:5])
unique_words = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index


#===============================
# print(type(word_index), ".....")
index_word = {v: k for k, v in word_index.iteritems()}
#===============================





# print(len(text[-1]))
text = np.array(text)
# print(text.shape)

X = text[:,:-1]
Y = text[:,-1]
Y = to_categorical(Y, num_classes= unique_words)
seq_length = X.shape[1]



# print(text[:5])
# print(X[:5])
# print(Y[:5])

# print(X.shape)
# print(Y.shape)





#############################################################
# MODEL 1
#############################################################

## Model-1	(with an embedding layer, but trauned from scratch) ===========================================================
# model = Sequential()
# model.add(Embedding(unique_words, 50, input_length=seq_length))
# model.add(CuDNNGRU(256, return_sequences= True))
# model.add(CuDNNGRU(256))
# model.add(Dense(128, activation= 'relu'))
# model.add(Dense(unique_words, activation='softmax'))
# # print(model.summary())

# model.compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
# model.fit(X, Y, batch_size= 128, epochs=100)







#############################################################
# MODEL 2
#############################################################


## Expanding Dimentions for Model-2 ===========================================================
## (Because we are not using the embedding layer, so the input to the LSTM must follow the standard shape) 

X_new = np.expand_dims(X, axis=2)
# print(X_new.shape)

## Model-2 
# model = Sequential()
# model.add(CuDNNGRU(256, input_shape=(X_new.shape[1], X_new.shape[2]), return_sequences= True ))
# model.add(CuDNNGRU(256))
# model.add(Dense(128, activation= 'relu'))
# model.add(Dense(unique_words, activation='softmax'))
# model.compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
# print(model.summary())

# model.fit(X_new, Y, batch_size= 128, epochs=200)













#############################################################
# MODEL 3
#############################################################


## Working with pre-trained embeddings ==================================================

# '''
embeddings_dict = {}
EMB_path = '../glove.840B.300d/glove.840B.300d.txt'
content = open(EMB_path)
for line in content:
	pair = line.split()
	emb = np.asarray(pair[1:], dtype= 'float32')
	word = pair[0]
	embeddings_dict[word] = emb
content.close()	

print(len(embeddings_dict))


MAX_WORDS = 3000
emb_dim = 300
number_words = min(MAX_WORDS, unique_words)
emb_matrix = np.zeros((number_words, emb_dim))

for word, i in word_index.items():
	if i >= MAX_WORDS:
		continue
	emb = embeddings_dict.get(word)
	if emb is not None:
		emb_matrix[i] = emb

# emb_layer = Embedding(number_words, emb_dim, weights = [emb_matrix], input_length=sentence_len, trainable= False)



## New Model with word embeddings ===========================================================
model = Sequential()
model.add(Embedding(number_words, emb_dim, weights = [emb_matrix], input_length=sentence_len-1, trainable= False))
model.add(CuDNNGRU(256, return_sequences= True))
model.add(CuDNNGRU(256))
model.add(Dense(128, activation= 'relu'))
model.add(Dense(unique_words, activation='softmax'))
print(model.summary())


# model.load_weights('weights-improvement-64-0.0030.hdf5')
# model.compile(loss='categorical_crossentropy', optimizer='adam')


model = load_model('text_gen_model.h5')
model.compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])

# saved_weight="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(saved_weight, monitor='loss', verbose=1, save_best_only = True, mode='min')
# callbacks_list = [checkpoint]

# model.fit(X, Y, batch_size= 128, epochs=45)
# model.save('text_gen_model.h5')







## Preporcessing the raw input text using the same tokenizer ============================
tokens_list_of_list = []
tokens_list_of_list.append(input_tokens)
# print(tokens_list_of_list)

tokenized_tokens = tokenizer.texts_to_sequences(tokens_list_of_list)
padded_raw_input = pad_sequences(tokenized_tokens, maxlen=seq_length)
# print(padded_raw_input)
# print(padded_raw_input.shape)



generated_sentence = ''
seed = np.zeros((padded_raw_input.shape))
seed[:,:] = padded_raw_input 

print("Padded raw input text: ", padded_raw_input)
print("seed: ", seed)


## Printing the newly generated text from the seed text ===================================
for i in range(sentence_len):
	prediction = model.predict(seed)
	index = np.argmax(prediction)
	print(index)
	word = index_word[index]  
	generated_sentence = generated_sentence + str(word) + ' '
	
	seed[:,:-1] = seed[:,1:]
	seed[:,-1] = index
	padded_raw_input = padded_raw_input[1:]


print(generated_sentence)

# '''