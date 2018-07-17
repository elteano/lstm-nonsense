import glob
import re
import os.path
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint

raw_text = ''
print('Reading file.')
with open('countmonte.txt') as f:
    raw_text = f.read().lower()

print('Processing file.')
chars = sorted(list(set(raw_text)))
print(chars)
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

procIn = np.reshape(dataIn, (n_patterns, seq_length, 1))
procIn = procIn / float(len(char_to_int))
procOut = np_utils.to_categorical(dataOut)

print('Preparing model.')
model = Sequential()
model.add(LSTM(256, input_shape=(procIn.shape[1], procIn.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(procOut.shape[1], activation='softmax'))
print('Model prepared.')

wfiles = glob.glob('weightsv2-*-*-*.hdf5')
rx = re.compile('weightsv2-(\\d+)-\\d+-([\\d.]+).hdf5')
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

filepath='weightsv2-' + '{:02}'.format(max_num+1) + '-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list=[checkpoint]

def pick_ind(arr, k=3):
    inds = np.argsort(arr)[::-1]
    vals = []
    for i in inds[:k]:
        vals.append(arr[i])
    if min(vals) < 0:
        m = abs(min(vals) * 2)
        vals = list(map(lambda a: a + m))
    tot = sum(vals)
    r = np.random.random() * tot
    total = 0
    for i, v in zip(inds, vals):
        total += v
        if r <= total:
            return i

# Let's have 20 epochs, and create a sample after each
for ep in range(20):
    #model.fit(procIn, procOut, epochs=ep, batch_size=128, callbacks=callbacks_list, initial_epoch=(ep-1))

    start = np.random.randint(0, len(dataIn)-1)
    pattern = dataIn[start]
    print('Seed: {:}'.format(''.join([int_to_char[v] for v in pattern])))
    print('Generated:')
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(len(char_to_int))
        prediction = model.predict(x, verbose=0)
        #ind = np.argmax(prediction)
        ind = pick_ind(prediction[0])
        result = int_to_char[ind]
        seq_in = [int_to_char[v] for v in pattern]
        print(result, end='', flush=True)
        pattern.append(ind)
        pattern = pattern[1:len(pattern)]

    print()
    print('Done')
