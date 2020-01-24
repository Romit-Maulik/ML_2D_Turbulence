from keras.models import Sequential, Model
from keras.layers import Dense, dot, Input
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import numpy as np

def create_keras_model(parameter_dict, training_dict, validation_dict):

    #Import parameters from dict
    n_inputs = parameter_dict[0]

    #Import training data from dict
    training_inputs = training_dict[0]
    training_outputs = training_dict[1]

    # Import validation data from dict
    validation_inputs = training_dict[0]
    validation_outputs = training_dict[1]

    # initialization of turbulence models basis model
    model = Sequential()

    # Layers start
    input_layer = Input(shape=(n_inputs,))

    # Hidden layers
    x = Dense(50, activation='relu', use_bias=True)(input_layer)
    x = Dense(50, activation='relu', use_bias=True)(x)
    x = Dense(50, activation='relu', use_bias=True)(x)
    x = Dense(50, activation='relu', use_bias=True)(x)
    x = Dense(50, activation='relu', use_bias=True)(x)

    op_val = Dense(1, activation='linear', use_bias=True)(x)

    # # Output layer - invariant subspaces
    # x = Dense(n_invariants, activation='tanh', use_bias=False)(x)
    #
    # # Dot product with invariant basis vectors needs to be inserted as final layer
    # int_layer = Input(shape=(n_invariants,))
    #
    # op_val = dot([x, int_layer], axes=1, normalize=False)

    custom_model = Model(inputs=input_layer, outputs=op_val)

    filepath = "best_model.hd5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    custom_model.compile(optimizer='adam', loss='mean_squared_error')
    history_callback = custom_model.fit(training_inputs, training_outputs, epochs=5000, batch_size=512, verbose=1,
                     validation_data=(validation_inputs, validation_outputs),
                     callbacks=callbacks_list)

    loss_history = history_callback.history["loss"]
    val_loss_history = history_callback.history["val_loss"]

    loss_history = np.array(loss_history)
    np.savetxt("loss_history.txt", loss_history, delimiter=",")

    val_loss_history = np.array(val_loss_history)
    np.savetxt("val_loss_history.txt", val_loss_history, delimiter=",")

    #print(custom_model.summary())

def load_data():
    # Each row is a sample - each column is an input/output

    network_inputs = np.load('Training_stencil.npy')
    network_source = np.load('Training_source.npy')

    total_data_size = np.shape(network_inputs)[0]
    n_inputs = np.shape(network_inputs)[1]

    train_index = int(0.67 * total_data_size)
    test_index = int(0.33 * total_data_size)

    idx = np.random.randint(total_data_size,size=train_index)
    training_inputs = network_inputs[idx,:]
    training_source = network_source[idx,:]


    idx = np.random.randint(total_data_size, size=test_index)
    validation_inputs = network_inputs[idx, :]
    validation_source = network_source[idx, :]

    return [n_inputs], [training_inputs, training_source], [validation_inputs, validation_source]

#Main function
parameter_dict, training_dict, validation_dict = load_data()

create_keras_model(parameter_dict, training_dict, validation_dict)