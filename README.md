A few LSTM models in Keras to refamiliarize myself with the processes. The goal
is a character-level generator based on the *The Count of Monte Christo* by
Alexandre Dumas (p&egrave;re), accessible from
[Project Gutenberg](https://www.gutenberg.org/ebooks/1184). The [Plain Text
UTF-8](https://www.gutenberg.org/files/1184/1184-0.txt) version is used.

The original design was based on a [tutorial by Jason Brownlee](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/).
This tutorial worked with *Alice's Adventures in Wonderland* by Lewis Carroll, a
much shorter work - the filesize for *Alice's Adventures in Wonderland* is about
170kB compared to the 2.7MB for *The Count of Monte Cristo*. As such, I've so
far added another LSTM layer to take advantage of the additional data, and will
be making other modifications until I am satisfied with the result.
