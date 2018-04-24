from keras.models import Sequential, load_model
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, GRU, Dense
from keras.layers.wrappers import Bidirectional
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy

def run_classifier(x_train_seq,x_test_seq,y_train,y_test):
    model = Sequential()

    # embedding layer
    model.add(Embedding(10000,
                        300,
                        input_length=20))
    # RNN layers
    for d in range(1):
        model.add(Bidirectional(GRU(units=300,
                                    dropout=0.2,
                                    recurrent_dropout=0.2,
                                    return_sequences=True)))

    model.add(Bidirectional(GRU(units=300,
                                dropout=0.2,
                                recurrent_dropout=0.2,
                                return_sequences=False)))

    # fully-connected output layer
    model.add(Dense(8, activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # ---- COMPILE THE MODEL ----

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['top_k_categorical_accuracy', categorical_accuracy])

    model.summary()
    print("-------------------------------------------training model----------------------------------------")
    history = model.fit(x_train_seq,
                        y_train,
                        batch_size=64,
                        epochs=24, validation_data=(x_test_seq, y_test))
    #predicting the values
    print("-------------------------------------------model trained------------------------------------------")
    acc = model.evaluate(x_test_seq,y_test)

    print("Accuracy is:",acc[2]*100)
