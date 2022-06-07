import os,re,math,shutil
from contextlib import redirect_stdout
import numpy as np
from pandas import DataFrame
import TimeSeriesDataAnalysisLib.algorithm.batch_algorithm as ba

import TimeSeriesDataAnalysisLib.util.plt_tool as plt_tool
import TimeSeriesDataAnalysisLib.util.time_tool as time_tool


from typing import List,Dict,Callable,Any,Union,Tuple,Callable
from TimeSeriesDataAnalysisLib.interface.trainable_data_provider import TrainableDataProvider
import logging
import pathlib





class BaseExperiment:
    """define train experiment for usage mlflow ui and logging

    args
    ---------
    experiment_name: str

        give name as the experiment, this will list on mlflow ui navigate menu

    output_dir: str

        the output folder

    """
    import mlflow
    def __init__ (self,experiment_name:str,output_dir:str):
        
        reg='[^A-Za-z0-9_\.]+'
        name_check=re.match(reg, experiment_name)
        if name_check is not None:
            raise ValueError("experiment_name must be format as [A-Za-z0-9_\.]+")
        self.name=experiment_name
        self.output_dir=output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._load_model_from=None
        self._model_builder=None

    def _init_mlflow(self):
        
        run_name="{0:.3f}".format(time_tool.timestamp_utc_now())  #.replace('.','_')
        filename="{0}_{1}".format(self.name,run_name)
        tracking_output=os.path.join(self.output_dir,'mlruns')
        tracking_uri=pathlib.Path(tracking_output).as_uri()
        self.mlflow.set_tracking_uri( tracking_uri )
        self.mlflow.set_experiment(self.name)
        return run_name,filename,tracking_output
    def _log_mlflow(self,model_name:str,run_name:str,filename:str,store_dir:str,tracking_output:str):
        print("--------training: {0}".format(model_name))
        info=self.mlflow.active_run().info
        self.mlflow.set_tag('model',model_name)
        self.mlflow.set_tag('experiment_class',self.__class__.__name__)
        #mlflow.log_artifacts(self.store.store_dir)
        if store_dir is not None:
            self.mlflow.set_tag('store_uri',pathlib.Path(store_dir).as_uri())  
        self.mlflow.set_tag('img_uri',pathlib.Path(os.path.join(self.output_dir,'img')).as_uri()) 
        print("--------run_id: {0}".format(info.run_id))
        print("--------tracking output: {0}".format(tracking_output))

    def _check_model_loaded(self):
        if self._model_builder.isModelExist()==False:
            raise RuntimeError("you should create or load model before")
        # if isinstance(self._g_model,BaseEstimator):
        #     print("loaded sk model")
        #     return 'sk'
        # if isinstance(self._g_model,M.Model):
        #     print("loaded keras model")
        #     return 'keras'
        return self._load_model_from
        
    def _log_metric(self,metric_keys:list,metric_values:list):
        for i,key in enumerate(metric_keys):
            self.mlflow.log_metric( key, metric_values[i])

    


class SKExperiment(BaseExperiment):
    
    def __init__ (self,experiment_name:str,output_dir:str):
        from TimeSeriesDataAnalysisLib.algorithm.net.sklearn import BaseSKModelBuilder,SKWorkflow
        super().__init__(experiment_name,output_dir)
    def create_model(self,model_builder:'BaseSKModelBuilder',**kwargs):
        """create a new model, create detail is in BaseSKModelBuilder.create_model()
        """
        self._model_builder=model_builder
        self._model_builder.create_model(**kwargs)
        self._load_model_from=None
    def load_model(self,model_builder:'BaseSKModelBuilder',model_dir:str):
        """load a exist model, load detail is in BaseSKModelBuilder.load_model()
        """
        self._model_builder=model_builder
        self._model_builder.load_model(model_dir)
        self._load_model_from=pathlib.Path(os.path.join( model_dir, self._model_builder.model_filename)).as_uri()

    
    def predict_evaluate(self,workflow:'SKWorkflow'):
        check=self._check_model_loaded()
        run_name,filename,tracking_output=self._init_mlflow()
        result_path=os.path.join(self.output_dir,self._model_builder.result_dir,filename)
        model_path=os.path.join(result_path,self._model_builder.model_dir)
        model_name=self._model_builder.get_model_name()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        
        with self.mlflow.start_run(run_name=run_name) :
            # logs meta
            self._log_mlflow(model_name,run_name,filename,workflow.data_provider.store.store_dir,tracking_output)
            self.mlflow.set_tag('result_uri',pathlib.Path(result_path).as_uri()) 
            self.mlflow.set_tag('model_uri',pathlib.Path(model_path).as_uri())  
            # do  evaluate_flow_fn
            params_log=workflow.set_params_log()
            predict_log=workflow.predict_evaluate_flow()
            # logs result
            if params_log is not None:
                self.mlflow.log_params(params_log)
            if check is not None:
                self.mlflow.log_param('load_model',check)
            if predict_log is not None:
                for cate_key,cate_v in predict_log.items():
                    if cate_v is not None:
                        if cate_v["values"] is not None:
                            metric_keys=[ cate_key+'_evaluate_'+name for name in cate_v["names"]]
                            self._log_metric(metric_keys,cate_v["values"])
            print("--------model output: {0}".format(model_path))
            print("--------finished--------")
    def train_evaluate(self,workflow:'SKWorkflow'):
        """train detail is in BaseSKModelBuilder.fit() and work flow is in evaluate_flow_fn

        args
        ---------
        workflow: SKWorkflow


        """
        check=self._check_model_loaded()
        run_name,filename,tracking_output=self._init_mlflow()
        result_path=os.path.join(self.output_dir,self._model_builder.result_dir,filename)
        model_path=os.path.join(result_path,self._model_builder.model_dir)
        model_name=self._model_builder.get_model_name()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        
        with self.mlflow.start_run(run_name=run_name) :
            # logs meta
            self._log_mlflow(model_name,run_name,filename,workflow.data_provider.store.store_dir,tracking_output)
            self.mlflow.set_tag('result_uri',pathlib.Path(result_path).as_uri()) 
            self.mlflow.set_tag('model_uri',pathlib.Path(model_path).as_uri())  
            # do  evaluate_flow_fn
            params_log=workflow.set_params_log()
            history,evaluate_log=workflow.train_evaluate_flow()
            # logs result
            if params_log is not None:
                self.mlflow.log_params(params_log)
            if check is not None:
                self.mlflow.log_param('load_model',check)
            if evaluate_log is not None:
                for cate_key,cate_v in evaluate_log.items():
                    if cate_v is not None:
                        if cate_v["values"] is not None:
                            metric_keys=[ cate_key+'_evaluate_'+name for name in cate_v["names"]]
                            self._log_metric(metric_keys,cate_v["values"])

            self._model_builder.save_model(model_path)
            print("--------model output: {0}".format(model_path))
            print("--------finished--------")

            
    


class KerasExperiment(BaseExperiment):
    
    def __init__ (self,experiment_name:str,output_dir:str):
        from TimeSeriesDataAnalysisLib.algorithm.net.keras import BaseKerasModelBuilder,KerasWorkflow
        
        super().__init__(experiment_name,output_dir)
        self._sess=None
    def create_model(self,model_builder:'BaseKerasModelBuilder',**kwargs):
        """create a new model, create detail is in BaseKerasModelBuilder.create_model(), 
        compile detail is in model_builder
        """
        self._model_builder=model_builder
        self._model_builder.create_model(**kwargs)
        self._model_builder.compile() 
        self._model_builder.summary()
        self._load_model_from=None
 
    def load_model(self,model_builder:'BaseKerasModelBuilder',model_dir:str):
        """load a exist model, load detail is in BaseKerasModelBuilder.load_model(), 
        compile detail is in model_builder
        """
        self._model_builder=model_builder
        self._model_builder.load_model(model_dir)
        self._model_builder.compile() 
        self._model_builder.summary()
        self._load_model_from=pathlib.Path(os.path.join( model_dir, self._model_builder.model_filename)).as_uri()

    def predict_evaluate(self,workflow:'KerasWorkflow'):
        check=self._check_model_loaded()
        run_name,filename,tracking_output=self._init_mlflow()
        result_path=os.path.join(self.output_dir,self._model_builder.result_dir,filename)
        model_path=os.path.join(result_path,self._model_builder.model_dir)
        model_name=self._model_builder.get_model_name()
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        
        with self.mlflow.start_run(run_name=run_name) :
            # logs meta
            self._log_mlflow(model_name,run_name,filename,workflow.data_provider.store.store_dir,tracking_output)
            self.mlflow.set_tag('result_uri',pathlib.Path(result_path).as_uri()) 
            self.mlflow.set_tag('model_uri',pathlib.Path(model_path).as_uri())  
            # do  evaluate_flow_fn
            params_log=workflow.set_params_log()
            predict_log=workflow.predict_evaluate_flow()
            # logs result
            if params_log is not None:
                self.mlflow.log_params(params_log)
            if check is not None:
                self.mlflow.log_param('load_model',check)
            if predict_log is not None:
                for cate_key,cate_v in predict_log.items():
                    if cate_v is not None:
                        if cate_v["values"] is not None:
                            metric_keys=[ cate_key+'_evaluate_'+name for name in cate_v["names"]]
                            self._log_metric(metric_keys,cate_v["values"])
            print("--------model output: {0}".format(model_path))
            print("--------finished--------")

    def train_evaluate(self,workflow:'KerasWorkflow'):
        """train detail is in BaseKerasModelBuilder.fit() or fit_generator(). And workflow is in evaluate_flow_fn

        args
        ---------
        workflow: KerasWorkflow
        """
        check=self._check_model_loaded()
        run_name,filename,tracking_output=self._init_mlflow()
        result_path=os.path.join(self.output_dir,self._model_builder.result_dir,filename)
        model_path=os.path.join(result_path,self._model_builder.model_dir)
        model_name=self._model_builder.get_model_name()
        tb_path=os.path.join(result_path,self._model_builder.tb_name)
        workflow.set_tb_path(tb_path)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        with self.mlflow.start_run(run_name=run_name) :
            # logs meta
            self._log_mlflow(model_name,run_name,filename,workflow.data_provider.store.store_dir,tracking_output)
            self.mlflow.set_tag('tensorboard_uri',pathlib.Path(tb_path).as_uri())   
            self.mlflow.set_tag('result_uri',pathlib.Path(result_path).as_uri()) 
            self.mlflow.set_tag('model_uri',pathlib.Path(model_path).as_uri())  
            # do  evaluate_flow_fn
            params_log=workflow.set_params_log()
            history,evaluate_log=workflow.train_evaluate_flow()
            
            # logs result
            if params_log is not None:
                self.mlflow.log_params(params_log)
            if check is not None:
                self.mlflow.log_param('load_model',check)
            if evaluate_log is not None:
                for cate_key,cate_v in evaluate_log.items():
                    if cate_v is not None:
                        if cate_v["values"] is not None:
                            metric_keys=[ cate_key+'_evaluate_'+name for name in cate_v["names"]]
                            self._log_metric(metric_keys,cate_v["values"])
            if 'acc' in history.history.keys():
                history_acc=history.history['acc']
            else:
                history_acc=history.history['accuracy']
            history_loss=history.history['loss']
            if params_log['valid'] > 0:
                self._draw_history( history_acc,history.history['val_acc'],'history_acc',result_path )
                self._draw_history( history_loss,history.history['val_loss'],'history_loss',result_path )
            else:
                self._draw_history( history_acc,None,'history_acc',result_path )
                self._draw_history( history_loss,None,'history_loss',result_path )


            self._dump_summary(self._model_builder,result_path)
            self._model_builder.save_model(model_path)
            print("--------model output: {0}".format(model_path))
            print("--------finished--------")
        
    def _dump_summary(self,model_builder:'BaseKerasModelBuilder',output_dir:str):
        target=os.path.join(output_dir , 'model_summary.txt')
        with open( target , 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                model_builder.summary()
        self.mlflow.log_artifact(target)
    def _draw_history(self,train,valid,title:str,output_dir:str):
        target=os.path.join(output_dir,"{0}.png".format(title)  ) 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt_tool.sns.set()
        fig = plt_tool.plt.figure(figsize=(10,5),dpi=300)
        # summarize history for accuracy
        plt_tool.plt.plot(train)
        legend=['train']
        if valid is not None:
            plt_tool.plt.plot(valid)
            legend=['train','valid']
        plt_tool.plt.title(title)
        plt_tool.plt.legend(legend, loc='upper left')
        plt_tool.plt.tight_layout()
        fig.savefig(target  )
        plt_tool.plt.clf()
        self.mlflow.log_artifact(target)