import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

class DualPlotCallback(Callback):
    """Callback that saves a .csv file containing the loss and 
            accuracy values by epoch and  saves a .png file plotting
            the  loss and accuracy values by epoch. It will open an 
            updated plot (inline)  at the end of each epoch. 
            
    # Arguments
    
        log_file_path:
            A string containing a path to a filename for the .csv 
            file containing the loss and accuracy values by epoch.

        log_fig_path:
            A string containing a path to a filename for the .png 
            file plotting the loss and accuracy values by epoch.            
    """
    
    def __init__(self, log_file_path=None, log_fig_path=None, **kwargs):
        super(DualPlotCallback, self).__init__(**kwargs)
        self.log_file_path = log_file_path
        self.log_fig_path = log_fig_path
        if self.log_file_path == None:
            raise ValueError("log_file_path=None is not supported.")
        if self.log_fig_path == None:
            raise ValueError("log_fig_path=None is not supported.")   
        
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
        self.df_log = self.df_log[['epoch_x', 'accuracy', 'val_accuracy', \
                                   'losses', 'val_losses', 'logs']]

        # Save log dataframe to csv
        self.df_log.to_csv(self.log_file_path)        
        
        # Create summary plots of Loss vs Epoch and Accuracy vs Epoch
        try:
            clear_output(wait=True)
            plt.close(fig)        
        except:
            pass
        
        plt.style.use('ggplot')
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
        plt.show()