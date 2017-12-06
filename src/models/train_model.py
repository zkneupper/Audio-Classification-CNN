
import os
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.utils import HDF5Matrix
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras import regularizers

from string import punctuation, printable


class User_Defined_Callback(Callback):

    def __init__(self):

      # Create a variable for the project root directory
        self.proj_root = os.path.join(os.path.join(os.pardir), os.path.join(os.pardir))

      # Save the path to the models folder
        self.models_dir = os.path.join(self.proj_root, "models")

      # Save the path to the models/log folder
        self.models_log_dir = os.path.join(self.models_dir, "log")

      # log_figure file_name
        self.fig_file_name = "log_figure"

      # Save the path to the log_figure
        self.log_fig_path = os.path.join(self.models_log_dir, self.fig_file_name)

      # log_dataframe.csv file_name
        self.log_file_name = "log_dataframe.csv"

      # Save the path to the log_figure
        self.log_file_path = os.path.join(self.models_log_dir, self.log_file_name)


    def on_train_begin(self, logs={}):
        self.i = 1
        self.x = []
        self.accuracy = []
        self.val_accuracy = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []
        self.df_log = pd.DataFrame()
        self.df_log.to_csv(self.log_file_path)

    def on_epoch_end(self, epoch, logs={}):
        
        # Update lists
        self.logs.append(logs)
        self.x.append(self.i)
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        # Create log dataframe
        self.df_log = pd.DataFrame({'epoch_x': self.x,
                               'accuracy' : self.accuracy,
                               'val_accuracy' : self.val_accuracy,
                               'losses' : self.losses,
                               'val_losses' : self.val_losses,
                               'logs' : self.logs})

        # Reorder dataframe columns
        self.df_log = self.df_log[['epoch_x', 'accuracy', 'val_accuracy', 'losses', 'val_losses', 'logs']]

        # Save log dataframe to csv
        self.df_log.to_csv(self.log_file_path)

        
        # Create summary plots of Loss vs Epoch and Accuracy vs Epoch
        
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))

        ax1.plot(self.x, self.losses, label="loss")
        ax1.plot(self.x, self.val_losses, label="val_loss")
        ax1.axes.set_xlabel('Epoch')
        ax1.axes.set_ylabel('Loss Function')   
        ax1.axes.set_title('Loss vs Epoch')  
        ax1.legend()

        ax2.plot(self.x, self.accuracy, label="acc")
        ax2.plot(self.x, self.val_accuracy, label="val_acc")
        ax2.axes.set_xlabel('Epoch')
        ax2.axes.set_ylabel('Accuracy')   
        ax2.axes.set_title('Accuracy vs Epoch')  
        ax2.legend()

        plt.tight_layout()
        
        plt.savefig(self.log_fig_path, dpi=300)
#        plt.show()



class SOUNDNET:
    def __init__(self):

       # File paths
        self.proj_root = None
        self.interim_data_dir = None
        self.spectrogram_arrays_path = None
        self.test_hdf5_path = None
        self.train_hdf5_path = None
        self.models_dir = None
        self.model_path = None
        self.models_log_dir = None
        self.fig_file_name = None
        self.log_fig_path = None
        self.log_file_name = None
        self.log_file_path = None
        self.models_checkpoints_dir = None
        self.src_dir = None

       # Data objects
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.Y_train = None
        self.Y_test = None

       # Callback objects
        self.user_defined_callback = None
        self.checkpoint_file_name = None
        self.models_checkpoints_path = None
        self.checkpoint = None
        self.callbacks_list = None

       # Model objects
        self.model = None
        self.score = None
        self.history = None


    def set_paths(self):
        '''
        Set file paths
        '''
        print('Setting file paths...')

      # Create a variable for the project root directory
        self.proj_root = os.path.join(os.path.join(os.pardir), os.path.join(os.pardir))


      # Save the path to the folder that contains the interim data sets for modeling: /data/interim
        self.interim_data_dir = os.path.join(self.proj_root, "data", "interim")

      # Save path to the folder for the spectrogram arrays
        self.spectrogram_arrays_path = os.path.join(self.interim_data_dir, "spectrogram_arrays")

      # Full path for test_hdf5_path
        self.test_hdf5_path = os.path.join(self.spectrogram_arrays_path, "spectrogram_arrays_test.hdf5")

      # Full path for train_hdf5_path
        self.train_hdf5_path = os.path.join(self.spectrogram_arrays_path, "spectrogram_arrays_train.hdf5")

      # Save the path to the models folder
        self.models_dir = os.path.join(self.proj_root, "models")

      # Full path for my_model.hdf5
        self.model_path = os.path.join(self.models_dir,  "my_model.hdf5")

      # Save the path to the models/log folder
        self.models_log_dir = os.path.join(self.models_dir, "log")

      # log_figure file_name
        self.fig_file_name = "log_figure"

      # Save the path to the log_figure
        self.log_fig_path = os.path.join(self.models_log_dir, self.fig_file_name)

      # log_dataframe.csv file_name
        self.log_file_name = "log_dataframe.csv"

      # Save the path to the log_figure
        self.log_file_path = os.path.join(self.models_log_dir, self.log_file_name)

      # Save the path to the models/checkpoints folder
        self.models_checkpoints_dir = os.path.join(self.models_dir, "checkpoints")

      # add the 'src' directory as one where we can import modules
        self.src_dir = os.path.join(self.proj_root, "src")

        sys.path.append(self.src_dir)


    def create_data_objects(self):
        '''
        Create data objects
        '''
        print('Creating data objects...')
        self.X_train = HDF5Matrix(self.train_hdf5_path, 'spectrogram_arrays_X_train')
        self.y_train = HDF5Matrix(self.train_hdf5_path, 'spectrogram_arrays_y_train')
        self.X_test = HDF5Matrix(self.test_hdf5_path, 'spectrogram_arrays_X_test')
        self.y_test = HDF5Matrix(self.test_hdf5_path, 'spectrogram_arrays_y_test')

        print('Preprocessing class labels...')
        self.Y_train = np_utils.to_categorical(self.y_train)
        self.Y_test = np_utils.to_categorical(self.y_test)


    def set_callbacks(self):
        '''
        Set callbacks
        '''
        print('Setting callbacks...')
        self.user_defined_callback = User_Defined_Callback()
        self.checkpoint_file_name = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        self.models_checkpoints_path = os.path.join(self.models_checkpoints_dir, self.checkpoint_file_name)
        self.checkpoint = ModelCheckpoint(self.models_checkpoints_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        self.callbacks_list = [self.checkpoint, self.user_defined_callback]


    def train_soundnet(self):
        print('Training SOUNDNET...')
        # For reproducibility
        np.random.seed(42)

      # Define model architecture
        self.model = Sequential()

      # Input Layer
        self.model.add(Activation(None, input_shape=(96, 173, 1)))
        self.model.add(BatchNormalization())

      # Convolution Layer 1
        self.model.add(Convolution2D(24, (5, 5), activation='relu', input_shape=(96, 173, 1)))
        self.model.add(MaxPooling2D(pool_size=(4,2)))
#        self.model.add(BatchNormalization())

      # Convolution Layer 2
        self.model.add(Convolution2D(48, (5, 5), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(4,2)))
#        self.model.add(BatchNormalization())

      # Convolution Layer 3
        self.model.add(Convolution2D(48, (5, 5), padding='same', activation='relu'))
#        self.model.add(BatchNormalization())

      # Dense Layer 
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001)))

      # Softmax Layer 
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001)))


      # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

      # Fit model on training data
        self.history = self.model.fit(self.X_train, self.Y_train,  batch_size=100, epochs=50,  verbose=1,  callbacks=self.callbacks_list, validation_data=(self.X_test, self.Y_test), shuffle="batch")

      # Evaluate model on test data
        self.score = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
        print('test score:', self.score[1])

      # Create a HDF5 file 'my_model.hdf5'
        print('Saving convnet...')
        self.model.save(self.model_path)


if __name__ == '__main__':
    soundnet = SOUNDNET()
    soundnet.set_paths()
    soundnet.create_data_objects()
    soundnet.set_callbacks()
    soundnet.train_soundnet()


