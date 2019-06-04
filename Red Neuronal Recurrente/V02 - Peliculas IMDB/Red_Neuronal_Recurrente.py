# LSTM Secuencia de Clasificación de Críticas de Películas con IMDB dataset
import numpy
import os
import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

if __name__ == "__main__":
    # Eliminacion de mensajes de advertencia de TensorFlow y Keras.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Semilla aleatoria de reproducibilidad.
    numpy.random.seed(7)

    # Cargar el conjunto de datos pero sólo mantener 
    # las n palabras principales, cero el resto.
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # Truncar y rellenar secuencias de entrada.
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # Creacion del modelo.
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Entrenamiento.
    model.fit(X_train, y_train, epochs=3, batch_size=64, verbose=0)

    # Final evaluacion del modelo.
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Precision: %.2f%%" % (scores[1]*100))