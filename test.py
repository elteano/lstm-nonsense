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
char_to_int = {c : i for i, c in enumerate(set(raw_text))}

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
model.add(LSTM(256, input_shape=(procIn.shape[1], procIn.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(procOut.shape[1], activation='softmax'))
print('Model prepared.')

wfiles = glob.glob('weights-*-*-*.hdf5')
rx = re.compile('weights-(\\d+)-\\d+-([\\d.]+).hdf5')
max_num = 0
if wfiles:
    min = (50., wfiles[0])
    for mch, name in [(rx.fullmatch(fname), fname) for fname in wfiles]:
        max_num = max(int(mch.group(1)), max_num)
        val = float(mch.group(2))
        if val < min[0]:
            min = (val, name)
    print('Loading weights from {:}'.format(min[1]))
    model.load_weights(min[1])

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath='weights-' + '{:02}'.format(max_num+1) + '-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list=[checkpoint]

model.fit(procIn, procOut, epochs=20, batch_size=128, callbacks=callbacks_list)
