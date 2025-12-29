import numpy as np

def load_bloodmnist(path="bloodmnist.npz"):
    data = np.load(path)
    

    x_train = data['train_images'].transpose(0, 3, 1, 2) / 255.0
    y_train = data['train_labels'].flatten()
    
    x_val = data['val_images'].transpose(0, 3, 1, 2) / 255.0
    y_val = data['val_labels'].flatten()
    
    x_test = data['test_images'].transpose(0, 3, 1, 2) / 255.0
    y_test = data['test_labels'].flatten()
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)