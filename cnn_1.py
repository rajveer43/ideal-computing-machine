from keras.models import Sequential, Model
from keras.layers import (
    Dense,
    Embedding,
    Activation,
    merge,
    Input,
    Lambda,
    Reshape,
    LSTM,
    RNN,
    SimpleRNNCell,
    SpatialDropout1D,
    Add,
    Maximum,
)
from keras.layers import (
    Conv1D,
    Flatten,
    Dropout,
    MaxPool1D,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    concatenate,
    AveragePooling1D,
)
from keras import regularizers
from keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

from config import inputLen


def textcnn1(tokenizer, class_num=2):
    """
    Build a text classification model using a convolutional neural network.

    Args:
        tokenizer: A tokenizer object used to preprocess the text data.
        class_num (int): Number of classes for classification.

    Returns:
        keras.models.Model: A Keras model for text classification.
    """
    kernel_size = [1, 3, 5]
    acti = "relu"
    my_input = Input(shape=(inputLen,), dtype="int32")
    emb = Embedding(
        len(tokenizer.word_index) + 1, 20, input_length=inputLen
    )(my_input)
    emb = SpatialDropout1D(0.2)(emb)

    net = []
    for kernel in kernel_size:
        con = Conv1D(
            32,
            kernel,
            activation=acti,
            padding="same",
            kernel_regularizer=l2(0.0005),
        )(emb)
        con = MaxPool1D(2)(con)
        net.append(con)
    net = concatenate(net, axis=-1)
    # net = concatenate(net)
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(256, activation="relu")(net)
    net = Dropout(0.5)(net)
    net = Dense(
        class_num,
        activation="softmax",
        kernel_regularizer=l2(l=0.001),
    )(net)
    model = Model(inputs=my_input, outputs=net)
    return model
