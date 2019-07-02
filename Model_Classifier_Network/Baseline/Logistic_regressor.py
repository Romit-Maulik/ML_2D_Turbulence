import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from keras import metrics

np.random.seed(10)
tf.random.set_random_seed(10)

def create_keras_model(features, targets):

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

    # Layers start
    input_layer = Input(shape=(n_inputs,))

    # ANN for logistic regression
    x = Dense(40, activation='relu', use_bias=True)(input_layer)
    x = Dense(40, activation='relu', use_bias=True)(x)
    x = Dense(40, activation='relu', use_bias=True)(x)
    x = Dense(40, activation='relu', use_bias=True)(x)
    x = Dense(40, activation='relu', use_bias=True)(x)

    op = Dense(3, activation='softmax', use_bias=True)(x)

    custom_model = Model(inputs=input_layer, outputs=op)

    filepath = "ML_Logistic.hd5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    custom_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])

    history_callback = custom_model.fit(training_features, training_targets, epochs=2000, batch_size=512, verbose=1,
                     validation_data=(validation_features, validation_targets),
                     callbacks=callbacks_list)

    loss_history = history_callback.history["loss"]
    loss_history = np.array(loss_history)
    np.savetxt("loss_history.txt", loss_history, delimiter=",")


    val_loss_history = history_callback.history["val_loss"]
    val_loss_history = np.array(val_loss_history)
    np.savetxt("val_loss_history.txt", val_loss_history, delimiter=",")

    training_accuracy = history_callback.history["acc"]
    training_accuracy = np.array(training_accuracy)
    np.savetxt("training_accuracy.txt", training_accuracy, delimiter=",")

    validation_accuracy = history_callback.history["val_acc"]
    validation_accuracy = np.array(validation_accuracy)
    np.savetxt("validation_accuracy.txt", validation_accuracy, delimiter=",")

    return custom_model


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
    
    model = create_keras_model(features,targets)


