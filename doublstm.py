import glob
import re
import os.path
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

raw_text = ''
print('Reading file.')
with open('countmonte.txt', encoding='utf8') as f:
    raw_text = f.read().lower()

print('Processing file.')
chars = sorted(list(set(raw_text)))
print(chars)
uchars = len(chars)
print('Number of unique characters: ' + str(uchars))
uchars = float(uchars)
char_to_int = {c : i for i, c in enumerate(chars)}
int_to_char = {i : c for i, c in enumerate(chars)}

n_chars = len(raw_text)

seq_length = 100
dataIn = []
dataOut = []

for i in range(n_chars - seq_length):
    seq_in = raw_text[i:i+seq_length]
    seq_out = raw_text[i+seq_length]
    dataIn.append([char_to_int[v] for v in seq_in])
    dataOut.append(char_to_int[seq_out])

n_patterns = len(dataIn)

print('Number of patterns: {:}'.format(n_patterns))

print('Formatting data...')
procIn = np.reshape(dataIn, (n_patterns, seq_length, 1))
procIn = procIn / uchars
procOut = np_utils.to_categorical(dataOut)
print('Data formatted.')

print('Preparing model.')
model = Sequential()
model.add(LSTM(256, input_shape=(procIn.shape[1], procIn.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(procOut.shape[1], activation='softmax'))

# This is our prediction model, which forgoes the dropout layers and instead has
# a lambda layer to implement a temperature parameter
temp = 0.3
pred_model = Sequential()
pred_model.add(LSTM(256, input_shape=(procIn.shape[1], procIn.shape[2]), return_sequences=True))
pred_model.add(LSTM(256))
pred_model.add(Lambda(lambda x : x / temp))
pred_model.add(Dense(procOut.shape[1], activation='softmax'))

print('Model prepared.')

wfiles = glob.glob('weights-relu-*-*-*.hdf5')
rx = re.compile('weights-relu-(\\d+)-\\d+-([\\d.]+).hdf5')
max_num = 0
if wfiles:
    best = (50., wfiles[0])
    for mch, name in [(rx.fullmatch(fname), fname) for fname in wfiles]:
        max_num = max(int(mch.group(1)), max_num)
        val = float(mch.group(2))
        if val < best[0]:
            best = (val, name)
    print('Loading weights from {:}'.format(best[1]))
    model.load_weights(best[1])

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath='weights-relu-' + '{:02}'.format(max_num+1) + '-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list=[checkpoint]

def pick_ind(arr, k=3):
    arr = np.array(list(map(lambda x: max(0,x), arr)))
    arr /= np.sum(arr)
    return np.random.choice(range(len(arr)), p=arr)

# Let's have 20 epochs, and create a sample after each
for ep in range(20):
    model.fit(procIn, procOut, epochs=ep, batch_size=128, callbacks=callbacks_list, initial_epoch=(ep-1))

    # Copy the weights we got into our prediction model
    pred_model.set_weights(model.get_weights())
    start = np.random.randint(0, len(dataIn)-1)
    pattern = dataIn[start]
    print('Seed: {:}'.format(''.join([int_to_char[v] for v in pattern])))
    print('Generated with temp of {:}:'.format(temp))
    for i in range(500):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / uchars
        prediction = pred_model.predict(x, verbose=0)
        #ind = np.argmax(prediction)
        ind = pick_ind(prediction[0])
        result = int_to_char[ind]
        seq_in = [int_to_char[v] for v in pattern]
        print(result, end='', flush=True)
        pattern.append(ind)
        pattern = pattern[1:len(pattern)]

    print()
    print('Done')
