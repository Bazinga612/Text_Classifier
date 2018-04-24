from keras.models import Sequential, load_model
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, GRU, Dense
from keras.layers.wrappers import Bidirectional

y_idx2word=8

def run_classifier(x_train_seq,x_test_seq,y_train_one_hot,y_test_one_hot):
    model = Sequential()
    print("-------------------------------------------building model----------------------------------------")
    # embedding layer
    model.add(Embedding(7000,
                        300,
                        input_length=20))
    # RNN layers
    for d in range(1):
        model.add(Bidirectional(GRU(units=100,
                                    dropout=0.2,
                                    recurrent_dropout=0.2,
                                    return_sequences=True)))

    model.add(Bidirectional(GRU(units=100,
                                dropout=0.2,
                                recurrent_dropout=0.2,
                                return_sequences=False)))

    # fully-connected output layer
    model.add(Dense(y_idx2word, activation='softmax'))

    # ---- COMPILE THE MODEL ----

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    print("-------------------------------------------training model----------------------------------------")
    history = model.fit(x_train_seq,
                        y_train_one_hot,
                        batch_size=64,
                        epochs=15, validation_data=(x_test_seq, y_test_one_hot))
    print("-------------------------------------------model trained------------------------------------------")
    acc = model.evaluate(x_test_seq,y_test_one_hot)
    print("accuracy is:",acc[1]*100)
    return acc