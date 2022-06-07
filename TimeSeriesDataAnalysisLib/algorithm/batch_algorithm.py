import numpy as np
from pandas import DataFrame
import enum,math
from typing import List,Dict,Tuple,Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from functools import reduce
from sklearn import cluster, datasets, metrics,decomposition

def reshape_b2d_to_b1d(x_train:np.ndarray)->np.ndarray:
    """ reshape 2,3 Dimension input to 1 Dimension
    """
    if len(x_train.shape)==2:
        #shape=(batch,x)
        x=x_train
    elif len(x_train.shape)==3:
        #shape=(batch,x,channel)
        x=x_train.copy().reshape((x_train.shape[0],reduce(lambda x,y:x * y,x_train.shape[1:])))
        print("reshape {0} as {1}".format(x_train.shape,x.shape))
    else:
        raise ValueError("input shape not supported")
    return x

def rmse_metrics(y_true, y_pred):
    """metrics root-mean-square deviation
    args
    ---------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
def mae_metrics(y_true, y_pred):
    """metrics Mean Square Error
    args
    ---------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    """
    return mean_absolute_error(y_true, y_pred)
def r2_metrics(y_true, y_pred):
    """metrics R-squared v1
    args
    ---------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.
    """
    return r2_score(y_true, y_pred)

def top_k_acc(y_true, y_pred,k=5): 
    """Top-k Accuracy
    args
    ---------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    kwargs
    ---------
    k : int, top num
    """
    max_k_preds = y_pred.argsort(axis=1)[:, -k:][:, ::-1] #得到top-k label
    match_array = np.logical_or.reduce(max_k_preds==y_true, axis=1) #得到匹配结果
    topk_acc_score = match_array.sum() / match_array.shape[0]
    return topk_acc_score

def eval_metrics(y_true, y_pred):
    """combine metrics rmse,mae,r2
    args
    ---------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) labels.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

    """
    rmse = rmse_metrics(y_true, y_pred)
    mae = mae_metrics(y_true, y_pred)
    r2 = r2_metrics(y_true, y_pred)
    return rmse, mae, r2
def kmeans_k_iteration(x_train:np.ndarray,method:str,max_k:int=15)->(list,list):
    """scan kmeans k value

    this api is not supported for big data

    args
    ---------
    x_train: np.ndarray
        input x

    kwargs
    ---------
    method: 'silhouette'|'elbow'
        kmeans method

    max_k: int
        max value of k to scan

    """
    x=reshape_b2d_to_b1d(x_train)
    score = []
    if max_k>x.shape[0]:
        max_k=x.shape[0]
    if method=='silhouette':
        ks = range(2, max_k)
    elif method=='elbow':
        ks = range(1, max_k)
    for k in ks:
        kmeans_fit = cluster.KMeans(init='k-means++',n_clusters = k,max_iter =1000,random_state=10101,precompute_distances=True).fit(x)
        cluster_labels = kmeans_fit.labels_
        if method=='silhouette':
            silhouette_avg = metrics.silhouette_score(x, cluster_labels)
            score.append(silhouette_avg)
        elif method=='elbow':
            score.append(kmeans_fit.inertia_)

    return ks,score
def kmeans_auto_label(x_train:np.ndarray,k:int):
    """label by kmeans

    args
    ---------
    x_train: np.ndarray
        input x

    k: int
        value of k

    """
    x=reshape_b2d_to_b1d(x_train)  
    kmeans_fit = cluster.KMeans(init='k-means++',n_clusters = k,max_iter =1000,random_state=10101,precompute_distances=True).fit(x)
    distance_to_cluster=kmeans_fit.transform(x)
    cluster_centers=kmeans_fit.cluster_centers_
    labels=kmeans_fit.labels_
    return distance_to_cluster, cluster_centers, labels 


def pca_distance(x_train:np.ndarray):
    """get 2d Principal Component Analysis feature"""
    x=reshape_b2d_to_b1d(x_train)  
    pca_fit=decomposition.PCA(n_components=2)
    pca_fit.fit(x)
    distance_to_0=pca_fit.transform(x)
    return distance_to_0
def ica_distance(x_train:np.ndarray):
    """get 2d Independent Component Analysis feature"""
    x=reshape_b2d_to_b1d(x_train)  
    ica_fit=decomposition.FastICA(n_components=2)
    ica_fit.fit(x)
    distance_to_0=ica_fit.transform(x)
    return distance_to_0

def score_to_onehot(predicted:np.ndarray)->np.ndarray:
    """classify result score to one hot label
    """
    result=np.zeros(predicted.shape)
    ind=np.argmax(predicted, axis=1)
    for i in range(predicted.shape[0]):
        result[i,ind[i]]=1
    return result
def indexes_split(indexes:Union[np.ndarray,List[int]],train_ratio:float ,valid_ratio:float ,test_ratio:float,shuffle:bool=True,random_seed:int=10101):
    """split index list with ratio and shuffle

    args
    ---------
    indexes: Union[np.ndarray,List[int]]
        input indexes to be split

    train_ratio: float 
        max is 1, train_ratio + valid_ratio + test_ratio must be 1

    valid_ratio: float 
        max is 1, 

    test_ratio: float 
        max is 1, 


    kwargs
    ---------
    shuffle: bool
        shuffle list

    random_seed: int
        available when shuffle=True


    """
    if round(train_ratio + valid_ratio + test_ratio,10) !=1.:
        raise ValueError("train_ratio + valid_ratio + test_ratio !=1")
    if type(indexes)==np.ndarray:
        pass
    else:
        x=np.asarray(indexes)
    size=x.shape[0]
    train_size=math.floor(size*train_ratio)
    valid_size=math.floor(size*valid_ratio)
    test_size=size-train_size-valid_size
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(x)
        np.random.seed(None)
    x_train=x[:train_size]
    x_valid=x[train_size:valid_size+train_size]
    x_test=x[valid_size+train_size:]
    msg='size: {0}, train_size: {1}, valid_size: {2}, test_size: {3}, random_seed={4}'.format(size,train_size,valid_size,test_size ,None if shuffle else random_seed)
    print(msg) 
    return  x_train,x_valid,x_test

def one_hot_encoder(labels:np.ndarray,class_num:int)->np.ndarray:
    """int label to one hot label with numpy
    """
    res = np.eye(class_num)[labels.reshape(-1)]
    enc=res.reshape(list(labels.shape)+[class_num])
    return enc