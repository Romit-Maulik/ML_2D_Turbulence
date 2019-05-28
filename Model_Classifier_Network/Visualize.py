import numpy as np
import matplotlib.pyplot as plt
cmap = plt.get_cmap('jet_r')

train_history = np.loadtxt('loss_history.txt')
val_history = np.loadtxt('val_loss_history.txt')

train_accuracy = np.loadtxt('training_accuracy.txt')
val_accuracy = np.loadtxt('validation_accuracy.txt')

plt.figure()
plt.plot(train_history[:],c='black',linewidth=2,label='Train')
plt.plot(val_history[:],c='red',linewidth=2,label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(train_accuracy[:],c='black',linewidth=2,label='Train')
plt.plot(val_accuracy[:],c='red',linewidth=2,label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

