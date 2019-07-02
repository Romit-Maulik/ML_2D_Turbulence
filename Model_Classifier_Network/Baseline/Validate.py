import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import metrics
from tensorflow.keras.backend import set_session

np.random.seed(10)

def validate_model(features, targets):
    sess = tf.Session()
    set_session(sess)
    sess.run(tf.global_variables_initializer())
    #Import parameters from dict
    n_inputs = np.shape(features)[1]

    #Partition to train and test
    total_data_size = np.shape(features)[0]
    train_index = int(0.67 * total_data_size)
    test_index = int(0.33 * total_data_size)

    idx = np.random.randint(total_data_size,size=train_index)
    training_features = features[idx,:]
    training_targets = targets[idx,:]

    idx = np.random.randint(total_data_size, size=test_index)
    validation_features = features[idx,:]
    validation_targets = targets[idx,:]

    model = load_model('best_model.hd5')

    # Classify - softmax
    predictions = model.predict(validation_features)
    predictions = np.argmax(predictions,axis=1)
    targets = np.argmax(validation_targets,axis=1)
    
    sumval = 0
    for i in range(np.shape(predictions)[0]):
    	if predictions[i] == targets[i]:
    		sumval = sumval + 1

    print('Accuracy: ',sumval/np.shape(predictions)[0])
    return model


def load_data():
    # Load master 2D turbulence dataset
    training_data = np.load('Feature_data.npy')

    # Select input features - Check Fortran code for quantities
    features = training_data[:, 0:np.shape(training_data)[1] - 1]
    evs = np.reshape(training_data[:, np.shape(training_data)[1] - 1], newshape=(np.shape(training_data)[0], 1))

    # Select what portion of data to be considered outliers
    sdev = 0.01 * np.std(evs)

    # Array for one-hot encoding
    targets = np.zeros(shape=(np.shape(features)[0], 3), dtype=int)

    # Prepare for logistic regression - negative eddy viscosities (will be targeted for AD)
    mask = evs[:, 0] < -sdev
    targets[mask, 0] = 1

    # Prepare for logistic regression - positive eddy viscosities (will be targeted for Smag)
    mask = evs[:, 0] > sdev
    targets[mask, 1] = 1

    # Prepare for logistic regression - not outliers (will be targeted for no modeling)
    mask = np.where((evs[:, 0] >= -sdev) & (evs[:, 0] <= sdev))
    targets[mask, 2] = 1

    # Retain all outlier data (since this number of samples is low)
    targets_a = targets[targets[:, 2] != 1, :]
    features_a = features[targets[:, 2] != 1, :]

    # Subsample no-model data (a large number of samples)
    targets_b = targets[targets[:, 2] == 1, :]
    features_b = features[targets[:, 2] == 1, :]
    idx = np.random.randint(low=0, high=np.shape(targets_b)[0], size=np.shape(targets_a)[0] // 2)
    targets_b = targets_b[idx, :]
    features_b = features_b[idx, :]

    # Make training space with equal physical representation
    targets = np.concatenate((targets_a, targets_b), axis=0)
    features = np.concatenate((features_a, features_b), axis=0)

    return features, targets


if __name__ == "__main__":
    features, targets = load_data()
    validate_model(features,targets)


