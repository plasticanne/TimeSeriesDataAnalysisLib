import TimeSeriesDataAnalysisLib.algorithm.batch_algorithm as ba
from TimeSeriesDataAnalysisLib.interface.trainable_data_provider import TrainableDataProvider

from TimeSeriesDataAnalysisLib.algorithm.net.base import AbcWorkflow,AbcModelBuilder

#from sklearn.externals import joblib 
import joblib
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import abc,os,shutil
import numpy as np
from typing import List,Dict,Union,Tuple,Generator


class BaseSKModelBuilder(AbcModelBuilder):
    """sklearn model builder for sklearn training task

    *for create new model, you should override create_model() and compile()

    *for loading exist pkl model, you can just use this class

    args
    ----------
    inputs_shape : List[Tuple[int]]
        the shapes of inputs list

    class_num : int
        the classify number
    """

    def __init__(self,inputs_shape,class_num):
        self.inputs_shape=inputs_shape
        self.class_num=class_num
        self.result_dir='result'
        self.model_dir='sk_model'
        self.model_filename='model.pkl'
        self.report_filename='report.txt'
        self.params_log={}
        self.metrics_names=('acc','RMSE','MAE')
                
    def create_model(self,*args, **kwargs)->bool:
        ## build your model here ##
        ## self.model = ...
        ## 
        return self.isModelExist()
    def evaluate_metrics(self, x:np.ndarray, y_true:np.ndarray, y_pred:np.ndarray,*args, **kwargs):
        evaluate=accuracy_score(y_true,y_pred)
        rmse = ba.rmse_metrics(y_true, y_pred)
        mae = ba.mae_metrics(y_true, y_pred)
        metrics_values=(evaluate, rmse, mae)
        return metrics_values
    def fit(self, x, y,*args, **kwargs):
        """  *args, **kwargs is skmodel input *args, **kwargs
        args
        ----------
        x : 1d array-like, or indicator array / sparse matrix

        y : 1d array-like, or label indicator array / sparse matrix
        
        """
        return self.model.fit(x, y,*args, **kwargs)
    def evaluate(self, x:np.ndarray, y_true:np.ndarray,*args, **kwargs)->Tuple:
        """ 
        args
        ----------
        x : 1d array-like, indicator array / sparse matrix

        y_pred : 1d array-like, or label indicator array / sparse matrix
        Predicted labels, as returned by a classifier.

        *args, **kwargs is skmodel input *args, **kwargs

        return
        ----------
        (metrics_names,metrics_values)
        """
        y_pred = self.model.predict(x)
        evaluate=accuracy_score(y_true,y_pred)
        rmse, mae, r2=ba.eval_metrics(y_true,y_pred)
        #top_5_acc=ba.top_k_acc(ba.one_hot_encoder(y,model.class_num ),   ba.one_hot_encoder( y_pred ,model.class_num  ))
        metrics_values=(evaluate, rmse, mae)
        self.metrics_names=('acc','RMSE','MAE')
        return metrics_values
    def predict(self, x:np.ndarray,*args, **kwargs)->np.ndarray:
        """ 
        args
        ----------
        x : 1d array-like, indicator array / sparse matrix

        *args, **kwargs is skmodel input *args, **kwargs

        return
        ----------
        ndarray
        """
        y_pred = self.model.predict(x)
        return y_pred

    def fit_generator(self,*args, **kwargs):
        """not available
        """
        pass
    def evaluate_generator(self,*args, **kwargs):
        """not available
        """
        pass
    def load_model(self,model_dir:str)->bool:
        """load joblib pkl format
        """
        self.model=joblib.load(os.path.join(model_dir,self.model_filename))
        return self.isModelExist()
    def save_model(self,model_dir:str):
        """save as joblib pkl format
        """
        joblib.dump(self.model, os.path.join(model_dir,self.model_filename) )  
        """Model.log(artifact_path=model_path,
                    flavor=mlflow.sklearn,
                    sk_model=g_model,
                    conda_env=None,
                    serialization_format="cloudpickle",
                    registered_model_name=None)"""
    
    

class SKWorkflow(AbcWorkflow):
    """workflow for training task detail 

    args
    ----------
    model_builder : BaseSKModelBuilder
        the sk ModelBuilder

    """
    def __init__(self,model_builder:BaseSKModelBuilder,model_name=None,output_dir=None):
        self.model_builder=model_builder
        self.class_num=self.model_builder.class_num
        self.inputs_shape=self.model_builder.inputs_shape
        self.model_name=model_name
        self.output_dir=output_dir

    def set_params_log(self):
        train_size=self.data_provider.x_train_indexes.shape[0]
        valid_size=self.data_provider.x_valid_indexes.shape[0]
        test_size=self.data_provider.x_test_indexes.shape[0]
        size=train_size+valid_size+valid_size
        params_log={
            'all':size,
            'train':train_size,
            'valid':valid_size,
            'test':test_size,
            'class_num':self.model_builder.class_num,
            }
        params_log.update(self.model_builder.params_log)
        params_log.update(self.model_builder.model.get_params(deep=False))
        return params_log
    

    def set_flow(self,data_provider:TrainableDataProvider,batch_size:int,*args,**kwargs):
        """set for training task detail incloud data input, evaluate, logging
        args
        ---------
        data_provider: TrainableDataProvider

            data

        batch_size: int

            batch_size of each fit epoch

        evaluate_flow_fn: Callable

            detail of how to feed data, training, and evaluate, see KerasWorkflow

        *args,**kwargs:

            direct input evaluate_flow_fn(*args,**kwargs)
        """
        self.data_provider=data_provider
        self.batch_size=batch_size
        self.args=args
        self.kwargs=kwargs
        
    def train_evaluate_flow(self):
        """workflow for training task detail incloud data input, evaluate, logging
        
        """
        history=None
        def process(mode,do_fit):
            nonlocal history
            if self.get_x_size(mode)>0:
                x_=self.load_x_value_ndarray(mode)
                y_=self.load_y_value_ndarray(mode)
                x_=ba.reshape_b2d_to_b1d(x_) 
                if do_fit:
                    history=self.model_builder.fit(x_, y_)  
                metrics_values=self.model_builder.evaluate(x_, y_) 
                return {"names":self.model_builder.metrics_names,"values":metrics_values} 
            else:
                return None

        metrics_log={
            "train":process("train",True),
            "valid": None,
            "test":process("test",False),
            
        }
        return history,metrics_log

    def predict_evaluate_flow(self):
        """workflow for training task detail incloud data input, evaluate, logging
        
        """
        metrics_names=["y_pred", "y_true"]
        def process(mode):
            if self.get_x_size(mode)>0:
                x_=self.load_x_value_ndarray(mode) #load x values after shuffled
                x_=ba.reshape_b2d_to_b1d(x_) 
                y_true=self.load_y_value_ndarray(mode) #load y values after shuffled
                y_pred=self.model_builder.predict(x_)
                metrics_values=[
                    self.convert_label_int_to_name_ndarray(y_pred),
                    self.convert_label_int_to_name_ndarray(y_true)
                ]
                return {"names":metrics_names,"values":metrics_values}
            else:
                return None
            
        metrics_log={
            "train":process('train'),
            "valid": None,  #no valid for predict
            "test": process('test'),
            
        }
        return metrics_log
    



