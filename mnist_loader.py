"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()

    #plot some numbers!
    #plt.show()
    #data = [np.reshape(x, (28,28)) for x in tr_d[0]]
    #for i in range(6):
    #    plt.imshow(data[i], cmap='gray')
    #    plt.show(block=True)

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]

    n_components = 784 #default input layer dimension
    use_PCA = False #change to True and adjust n_components for PCA
    if use_PCA:
        n_components = 50
        pca = PCA(n_components)

        x_train = np.hstack(training_inputs)
        pca.fit(x_train.transpose())
        x_transformed_train = pca.transform(x_train.transpose())
        training_inputs = np.hsplit(x_transformed_train.transpose(), 50000)

        x_valid = np.hstack(validation_inputs)
        x_transformed_valid = pca.transform(x_valid.transpose())
        validation_inputs = np.hsplit(x_transformed_valid.transpose(), 10000)

        x_test = np.hstack(test_inputs)
        x_transformed_test = pca.transform(x_test.transpose())
        test_inputs = np.hsplit(x_transformed_test.transpose(), 10000)

    training_data = zip(training_inputs, training_results)
    validation_data = zip(validation_inputs, va_d[1])
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data, n_components)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def save(training_data, validation_data, test_data, n_components, PCA_data):
    """Save the reduced dimensionality data to the file ``PCA_data``."""
    data = [training_data, validation_data, test_data, n_components]
    f = open(PCA_data, "wb")
    pickle.dump(data, f)
    f.close()

def load(PCA_data):
    """Load the dataset from the file ``PCA_data``.  Returns the dataset
    with reduced dimensionality.
    """
    f = open(PCA_data, "rb")
    data = pickle.load(f)
    f.close()
    return data
