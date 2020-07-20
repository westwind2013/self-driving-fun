from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2


def train(generate_training_data, generate_validation_data, 
          epoch_count, train_sample_count, valid_sample_count):
  input_shape = (64,64,3)
  model = Sequential()
  model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
  model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(80, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(40, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(16, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(10, W_regularizer = l2(0.001)))
  model.add(Dense(1, W_regularizer = l2(0.001)))
  adam = Adam(lr = 0.0001)
  model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
  model.summary()
  model.fit_generator(generate_training_data, 
                      samples_per_epoch=train_sample_count, 
                      nb_epoch=epoch_count, 
                      validation_data=generate_validation_data, 
                      nb_val_samples=valid_sample_count)
  # save model
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
  model.save_weights("model.h5")
