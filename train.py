#import csv
import numpy as np
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from cnn_model import cnn_model_fn
from sklearn.cross_validation import train_test_split
import h5py 
import matplotlib.pyplot as plt
from gol_conf import *



def train_and_evaluate_model(model, x_train, y_train, x_val, y_val):
    
    n_samples=np.size(x_train, axis=0)+np.size(x_val, axis=0)
    width=np.size(x_train, axis=1)-2
    height=np.size(x_train, axis=2)-2

    model.compile(loss='mse',
              optimizer=Adam(lr=0.0001, decay=1e-6),
              metrics=['accuracy'])
    checkpoint = ModelCheckpoint("gol_w_"+str(width)+"x"+str(height)+"x"+str(n_samples)+".h5", monitor="accuracy", verbose=1, save_best_only=True,
                                 save_weights_only=True, mode="max", period=1)
    stop = EarlyStopping(monitor="val_acc", patience=15, mode="auto")
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,              
              validation_data=(x_val, y_val),
              callbacks=[checkpoint, stop],
              shuffle=True)
    

def load_data(width, height, n_samples):
    dataset_name="dataset_"+str(width)+"x"+str(height)+"x"+str(n_samples)+".h5"
    try:
        data_file = h5py.File(dataset_name, 'r')
    except OSError:
        print(dataset_name," - not found")

    x_train = data_file["x_train"][:]
    width=np.size(x_train,axis=1)
    height=np.size(x_train,axis=2)
    X_tmp1=np.concatenate([x_train, x_train, x_train], axis=1)
    x_tmp2=np.concatenate([X_tmp1, X_tmp1, X_tmp1], axis=2)
    X=x_tmp2[:,width-1:2*width+1,height-1:2*height+1]
    X=X[:,:,:,np.newaxis]
    y_train = data_file["y_train"][:]
    y_train=y_train[:,:,:,np.newaxis]

    data_file.close() 
    return X, y_train

def main():
    
    x_train, y_train=load_data(width, height, n_samples)
    X_train, X_val, Y_train, Y_val   = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
 
    model=cnn_model_fn(width, height, n_samples)
    train_and_evaluate_model(model, X_train, Y_train, X_val, Y_val)
              
    model.save_weights("gol_w_"+str(width)+"x"+str(height)+"x"+str(n_samples)+".h5")
  #  model.save("gol_model_"+str(width)+"x"+str(height)+"x"+str(n_samples))   
    print(model.summary())

if __name__ == "__main__":
    main()