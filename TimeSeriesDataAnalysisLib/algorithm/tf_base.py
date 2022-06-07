import numpy as np
import tensorflow
tf=tensorflow.compat.v1
M=tf.keras
L=M.layers
K=M.backend

def one_hot_encoder(labels:np.ndarray,class_num:int)->np.ndarray:
    """int label to one hot label
    """
    enc = M.utils.to_categorical( labels,class_num )
    return enc

def r2(y_true, y_pred):
    """metrics R-squared v2
    args
    ---------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    """
    ss_res =  K.sum(K.square( y_true-y_pred ))
    ss_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - ss_res/(ss_tot + K.epsilon()) )