import pickle
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from config import inputLen

########################################################################################


def train_tokenizer(train_datas, test_datas, tokenizer_file_path):
    """
    Train a tokenizer on training and test data and save it to a file.

    Args:
        train_datas (list): List of training data samples as strings.
        test_datas (list): List of test data samples as strings.
        tokenizer_file_path (str): Path to save the trained tokenizer as a pickle file.

    Returns:
        Tokenizer: Trained tokenizer.
        numpy.ndarray: Padded and tokenized training data.
        numpy.ndarray: Padded and tokenized test data.
    """
    tokenizer = Tokenizer(
        num_words=None,
        filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
        lower=True,
        split=" ",
        char_level=False,
    )
    tokenizer.fit_on_texts(train_datas)
    tokenizer.fit_on_texts(test_datas)
    # print(tokenizer.word_index)
    # # vocal = tokenizer.word_index
    train_datas = tokenizer.texts_to_sequences(train_datas)
    test_datas = tokenizer.texts_to_sequences(test_datas)
    train_datas = pad_sequences(
        train_datas, inputLen, padding="post", truncating="post"
    )
    test_datas = pad_sequences(
        test_datas, inputLen, padding="post", truncating="post"
    )

    with open(tokenizer_file_path, "wb") as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    return tokenizer, train_datas, test_datas


########################################################################################


def train_tokenizer_with_val(
    train_datas, val_datas, test_datas, tokenizer_file_path
):
    """
    Train a tokenizer on training, validation, and test data and save it to a file.

    Args:
        train_datas (list): List of training data samples as strings.
        val_datas (list): List of validation data samples as strings.
        test_datas (list): List of test data samples as strings.
        tokenizer_file_path (str): Path to save the trained tokenizer as a pickle file.

    Returns:
        Tokenizer: Trained tokenizer.
        numpy.ndarray: Padded and tokenized training data.
        numpy.ndarray: Padded and tokenized validation data.
        numpy.ndarray: Padded and tokenized test data.
    """
    tokenizer = Tokenizer(
        num_words=None,
        filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
        lower=True,
        split=" ",
        char_level=False,
    )
    tokenizer.fit_on_texts(train_datas)
    tokenizer.fit_on_texts(val_datas)
    tokenizer.fit_on_texts(test_datas)
    # print(tokenizer.word_index)
    # # vocal = tokenizer.word_index
    train_datas = tokenizer.texts_to_sequences(train_datas)
    val_datas = tokenizer.texts_to_sequences(val_datas)
    test_datas = tokenizer.texts_to_sequences(test_datas)
    train_datas = pad_sequences(
        train_datas, inputLen, padding="post", truncating="post"
    )
    val_datas = pad_sequences(
        val_datas, inputLen, padding="post", truncating="post"
    )
    test_datas = pad_sequences(
        test_datas, inputLen, padding="post", truncating="post"
    )

    with open(tokenizer_file_path, "wb") as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    return tokenizer, train_datas, val_datas, test_datas
