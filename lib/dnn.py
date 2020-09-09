from tensorflow import keras
from tensorflow.keras import layers


def dnn_model():
    num_out = 6
    model = keras.Sequential()

    model.add(layers.Conv1D(128, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.Conv1D(128, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.Conv1D(128, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.Conv1D(128, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(96, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.Conv1D(96, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.Conv1D(96, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.MaxPool1D(2))
    # model.add(layers.Flatten())
    model.add(layers.Conv1D(64, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.Conv1D(64, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(32, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.Conv1D(32, 1, input_shape=(26, 2), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(60, activation='relu'))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(num_out, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
