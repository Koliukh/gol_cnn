import h5py 
import numpy as np
import tqdm
from cnn_model import cnn_model_fn
from gol_conf import *
import matplotlib.pyplot as plt


def load_data(width, height, n_samples):
    dataset_name="dataset_test_"+str(width)+"x"+str(height)+"x"+str(n_samples)+".h5"
    try:
        data_file = h5py.File(dataset_name, 'r')
    except OSError:
        print(dataset_name," - not found")   
    
    x_test = data_file["x_test"][:]  
    width=np.size(x_test,axis=1)
    height=np.size(x_test,axis=2)
    X_tmp1=np.concatenate([x_test, x_test, x_test], axis=1)
    x_tmp2=np.concatenate([X_tmp1, X_tmp1, X_tmp1], axis=2)
    X=x_tmp2[:,width-1:2*width+1,height-1:2*height+1]

    X=X[:,:,:,np.newaxis]
    y_test = data_file["y_test"][:]
    y_test=y_test[:,:,:,np.newaxis]
    data_file.close()
    return X, y_test

def plot_data(x_train , y_train):
       
    dpi=12
    x = np.asarray(x_train)
    y = np.asarray(y_train)
    X=np.abs(x-y)
    assert X.ndim == 2
    X = X.astype(bool)
    figsize = (X.shape[1] , X.shape[0] )
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
    im = ax.imshow(X, cmap=plt.cm.binary, interpolation='nearest')
    im.set_clim(-0.05, 1)  # Make background gray  
    plt.show()
    input("Press Enter to continue...")

    return


def main():
    
    x_test, y_test=load_data(width, height, n_samples_test)
 
    model=cnn_model_fn(width, height, n_samples_test)
    print(model.summary())

    model.load_weights("gol_w_"+str(width)+"x"+str(height)+"x"+str(n_samples)+".h5")
           
    errors=0

    y_pred=model.predict(x_test, batch_size=None, verbose=0, steps=1)
    y_pred=np.round(y_pred)
    for idx in range(np.size(y_test,axis=0)):
        if ~(y_test[idx,:,:,0]==y_pred[idx,:,:,0]).all():
            errors+=1
          #  plot_data(y_pred[idx,:,:,0], y_test[idx,:,:,0])
    print("Total error frames :",errors," in ",np.size(y_test,axis=0)," frames")
    
if __name__ == "__main__":
    main()
    
    