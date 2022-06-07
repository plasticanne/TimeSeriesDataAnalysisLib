
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import example.working_for_training.interface2 as interface
import numpy as np
from pandas import DataFrame
import json
import TimeSeriesDataAnalysisLib.interface.untrainable_stored_v1_2 as un_v1_2
import TimeSeriesDataAnalysisLib.interface.trainable_stored_v1_0 as tr_v1_0
from TimeSeriesDataAnalysisLib.interface.trainable import SKExperiment,KerasExperiment,TrainableDataProvider
from TimeSeriesDataAnalysisLib.algorithm.ts_analysis import SimpleAnalysis,TimeSeriesStepAnnotatedFeature,PlotTerm,TimeSeriesStep
import TimeSeriesDataAnalysisLib.algorithm.single_ts_algorithm as ts
from TimeSeriesDataAnalysisLib.algorithm.net.keras import KerasWorkflow,BaseKerasModelBuilder
from TimeSeriesDataAnalysisLib.algorithm.net.sklearn  import SKWorkflow,BaseSKModelBuilder
import tensorflow
tf=tensorflow.compat.v1
M=tf.keras
L=M.layers
K=M.backend


class ProjResearchDataProvider(un_v1_2.ResearchArrayDataProvider):
    # for a normal Time Series project, extends TimeSeriesInjection.

    def feature_process(self)->dict:
        # override this method for define feature process on each feature value
        def cate1_v(value:np.ndarray,cfg:np.ndarray):
            vs=value/cfg[:,:,0]
            vx=cfg[:,:,1]-vs
            rs=cfg[:,:,3]*vx/5
            return rs
        def cate2_v(value:np.ndarray,cfg:np.ndarray):
            vs=value/cfg[:,:,0]
            vx=cfg[:,:,1]-vs
            rs=cfg[:,:,3]*vx*10
            return rs
        def other(value:np.ndarray,cfg:np.ndarray):
            return value
        alg_set={
            "cate1_v":{
                "set":[],
                "alg":cate1_v
                },
            "cate2_v":{
                "set":[],
                "alg":cate2_v
                },
            "other":{
                "set":[],
                "alg":other
                },
        }

        return alg_set
    
    def label_process(self):
        """override this method for how to get label
        if retrun None means this Injection whih no label"""
        return None

class ProjTimeSeriesStepAnalysis(TimeSeriesStepAnnotatedFeature):
    def ts_step_process(self):
        """override this method for define how to get TimeSeriesStep
        use default in this case"""
        ts_point=ts.get_ts_point(self.trend,enable_features=self.ts_step_process_enable_features,window_size=self.ts_step_process_window_size)
        shift=self.ts_step_process_window_size
        return ts_point,shift
class DumpToStore:
    # the detail fo Injection process
    def __init__(self,research_data_provider:ProjResearchDataProvider):
        
        ts_step_process_enable_features=[int(e) for e in range(14)]
        ts_step_process_window_size=10
        self.process_flow=ProjTimeSeriesStepAnalysis(
            research_data_provider,
            ts_step_process_enable_features=ts_step_process_enable_features,ts_step_process_window_size=ts_step_process_window_size)
        
    def plot_1d_features_img(self,a_file,img_dir):
        # features of 0~13
        enable_features1=[int(e) for e in range(14)]
        window_size=10
    
        name="_".join(a_file.split(os.path.sep)[-3:-1])
        # define TimeSeriesStep list which we will plot
        steps=[TimeSeriesStep.State0,TimeSeriesStep.State1,TimeSeriesStep.State2,TimeSeriesStep.State3,TimeSeriesStep.State4]
        # define Terms list which we will plot
        terms=[PlotTerm.all, PlotTerm.trend, PlotTerm.seasonal ,PlotTerm.resid ,PlotTerm.slope]
        self.process_flow.plot_package(name,      img_dir,steps,terms,enable_features1,window_size=window_size,nested_dir=True)
    
        
    def get_feature_0d(self)->np.ndarray:
        """0 dimension time series feature preprocess.
        """
        enable_features1=[int(e) for e in range(14)]
        window_size=10
        enable_featuresO=[int(e) for e in range(14,15)]
        

        result =self.process_flow.get_my_feature_0d(window_size,enable_features1,enable_featuresO)
        return result

    def dump_data_to_store_by_gen(self,gen,a_file:str)->None:
        # we want save store as feature_0d data
        result=self.get_feature_0d()
        next(gen)
        # define send object
        sendMap=tr_v1_0.StoreSendMap()
        sendMap.name=a_file
        sendMap.data=result
        # no label in this case
        sendMap.label=None
        # send to gen
        gen.send(sendMap)

class ProjDataProvider(TrainableDataProvider):
    # a DataProvider define how data feed to analysis instance
    # goal to specify indexes and train ratio for simplify lazy load data 
    def pre_process_x_value(self,x_value:np.ndarray)->np.ndarray:
        """override this method to defining preprocess of "each x value" .
        This acting when call load_batch_x_values.
        """
        return x_value
    def pre_process_y_value(self,y_value:np.ndarray)->np.ndarray:
        """override this method to defining preprocess of "each y value". 
        This acting when call load_batch_y_values.
        """
        return y_value

def store_analysis(store,output_dir):
    # commonly used analysis for a time series dataset
    def buildDataProvider(store):
        # ProjDataProvider is a operating data helper and preprocess with index list and ratio
        data_provider=ProjDataProvider(store)
        train_ratio=1
        valid_ratio=0
        test_ratio=0
        data_provider.x_setter_by_store(train_ratio,valid_ratio,test_ratio,shuffle=False)
        return data_provider
    data_provider=buildDataProvider(store)
    # SimpleAnalysis 
    analysis=SimpleAnalysis(data_provider,output_dir)
    analysis.kmeans_k_iteration()
    analysis.pca_distance()
    analysis.ica_distance()

class Task:
    # A task of base on generate a kmeans label data_provider
    def __init__ (self,name,output_dir):
        self.output_dir=output_dir
        self.name=name
    def gen_kmeans_label(self,store):
        data_provider=ProjDataProvider(store)
        train_ratio=1
        valid_ratio=0
        test_ratio=0
        data_provider.x_setter_by_store(train_ratio,valid_ratio,test_ratio,shuffle=False)
        analysis=SimpleAnalysis(data_provider,self.output_dir)
        k=3 #human determine k=3 is best with this dataset
        return analysis.kmeans_auto_label(k=k) 
    def kmeans_labeled_data(self,store,train_ratio,valid_ratio,test_ratio):
        labels=self.gen_kmeans_label(store)
        data_provider=ProjDataProvider(store)
        x_train_indexes,x_valid_indexes,x_test_indexes=data_provider.x_setter_by_store(train_ratio,valid_ratio,test_ratio,shuffle=False)
        data_provider.y_setter_by_input(labels[x_train_indexes],labels[x_valid_indexes],labels[x_test_indexes])
        return data_provider

class SKTaskA(Task):
    def run_sk_train(self,store):
        from TimeSeriesDataAnalysisLib.algorithm.net.sklearn.classifier import KNN,LDA,LogisticRegression,DecisionTree,RandomForest,SVC,NuSVC,LinearSVC
        builders_list=[KNN,LDA,LogisticRegression,DecisionTree,RandomForest,SVC,NuSVC,LinearSVC]
        batch_size=1000
        shapes=[(1,44)] # your feature shape
        class_num=3
        train_ratio=1
        valid_ratio=0
        test_ratio=0
        data_provider=self.kmeans_labeled_data(store,train_ratio,valid_ratio,test_ratio)
        for model_builder in builders_list:
            builder=model_builder(shapes,class_num)
            exp=SKExperiment(self.name,self.output_dir)
            exp.create_model(builder)
            workflow=SKWorkflow(builder)
            workflow.set_flow(data_provider,batch_size)
            # workflow.evaluate_flow is a function to decide how is your data input, training and evaluate steps.
            exp.train_model(workflow)
            
class SKTaskB(Task):
    def retrain_sk(self,store):
        batch_size=1000
        shapes=[(1,44)] # your feature shape
        class_num=3
        train_ratio=1
        valid_ratio=0
        test_ratio=0
        data_provider=self.kmeans_labeled_data(store,train_ratio,valid_ratio,test_ratio)
        dir='E:\\download\\44_output\\result\\sk_a_1578392529.892\\sk_model'
        class LoadedModel(BaseSKModelBuilder):
            # we define a Builder with only basic functions we need
            pass
        builder=LoadedModel(shapes,class_num)
        exp=SKExperiment(self.name,self.output_dir)
        exp.load_model(builder,dir)
        workflow=SKWorkflow(builder,output_dir=self.output_dir,model_name=self.name)
        workflow.set_flow(data_provider,batch_size)
        exp.train_model(workflow)

class KerasTaskA(Task):
    def run_keras_train(self,store):
        batch_size=1000
        shapes=[(1,44)]
        class_num=3
        initial_epoch=0
        epochs=10
        shuffle=True
        train_ratio=0.7
        valid_ratio=0.2
        test_ratio=0.1
        data_provider=self.kmeans_labeled_data(store,train_ratio,valid_ratio,test_ratio)

        from TimeSeriesDataAnalysisLib.algorithm.net.keras.classifier import OneLayerMLP as MLP
        builders_listA=[MLP]

        for model_builder in builders_listA:
            builder=model_builder(shapes,class_num)
            workflow=KerasWorkflow(builder,output_dir=self.output_dir,model_name=self.name)
            
            builder.set_gpu(0)
            # start_new_kares_task is a Simplification of sess and default_graph control, 
            # the usage just like sess 
            with builder.start_new_kares_task():
                exp=KerasExperiment(self.name,self.output_dir)
                exp.create_model(builder)
                workflow.set_flow(
                    data_provider,
                    batch_size,
                    initial_epoch=initial_epoch,
                    epochs=epochs,
                    shuffle=shuffle)
                exp.train_model(workflow)
            
   
class KerasTaskB(Task):
    def retrain_keras(self,store):
        batch_size=1000
        shapes=[(1,44)]
        class_num=3
        initial_epoch=0
        epochs=10
        shuffle=True
        dir='E:\\download\\44_output\\result\\keras_a_1581927058.823\\keras_model'
        train_ratio=0.7
        valid_ratio=0.2
        test_ratio=0.1
        data_provider=self.kmeans_labeled_data(store,train_ratio,valid_ratio,test_ratio)        
        class LoadedModel(BaseKerasModelBuilder):
            pass
        builder=LoadedModel(shapes,class_num)
        workflow=KerasWorkflow(builder)
        builder.set_gpu(0)
        with builder.start_new_kares_task():
            exp=KerasExperiment(self.name,self.output_dir)
            exp.load_model(builder,dir)
            exp.train_model(
                data_provider,
                batch_size,
                workflow.evaluate_flow,
                initial_epoch=initial_epoch,
                epochs=epochs,
                shuffle=shuffle)
        
        
    def convert(self):
        shapes=[(1,44)]
        class_num=3
        model_output='E:\\download\\44_output\\result\\keras_a_1578420074.587\\keras_model'
        class LoadedModel(BaseKerasModelBuilder):
            pass
        builder=LoadedModel(shapes,class_num)
        builder.convert_model(model_output, frozen_graph=True, lite_graph=True,quantized_graph=True,override=True)

    def evaluate_converted(self,store):
        shapes=[(1,44),] # 1 input of model
        class_num=3
        model_output='E:\\download\\44_output\\result\\keras_a_1578420074.587\\keras_model'
        train_ratio=0.7
        valid_ratio=0.2
        test_ratio=0.1
        outputs_num=1 # 1 output of model
        data_provider=self.kmeans_labeled_data(store,train_ratio,valid_ratio,test_ratio)   
        class LoadedModel(BaseKerasModelBuilder):
            pass
        builder=LoadedModel(shapes,class_num)
        workflow=KerasWorkflow(builder)
        workflow.evaluate_converted_model(data_provider,model_output,outputs_num)
def main(act):
    def get_file(local_file):
        with open(local_file,"r", encoding='utf-8') as f:
            return f.read()
    # get file list
    def file_list(folder,key):
        sets=[]
        for path, _, files in os.walk(folder):
            for name in files:
                if name.split('key_')[-1]==key:
                    sets.append(os.path.join(path, name))
        return sets

    sets=file_list("./fakedata2",'000.json')
    # name the store
    store_name='1x44'
    store_dir='E:\\download3\\44'
    output_dir="E:\\download3\\44_output"
    # create an empty store instance which we will inject data here.
    store=tr_v1_0.TrainableStore(store_dir,store_name)
    add_data_size=len(sets)
    
    # choice features and window_size for a better result of TimeSeriesStep
    
    # lets do dumping data to store
    
    if act=='dump':
        # create a generator by store with refresh mode, force override files exists in output_dir
        gen=store.dump_generator(tr_v1_0.TrainableStoreMode.refresh,add_data_size,batch_size=1000,labeled=False)
        for a_file in sets:
            print(a_file)
            dictO=json.loads(get_file(a_file))
           
            # dict object -> class object
            researchData=interface.ResearchDataObject().loads(dictO,force=False)
            
            # class object -> Array Features Store object 
            provider=ProjResearchDataProvider(researchData,interface.VERSION)
            # collects details to DoInjection
            doing_x=DumpToStore(provider)
            # dump 
            doing_x.dump_data_to_store_by_gen(gen,a_file)
            # plot
            doing_x.plot_1d_features_img(a_file,output_dir)


    elif act=='ana':
        # commonly used analysis for a time series dataset
        store_analysis(store,output_dir)
    elif act=='sk_a':
        # a sk model train task
        sk_a=SKTaskA('sk_a',output_dir)
        sk_a.run_sk_train(store)
    elif act=='sk_b':
        # a exist sk model retrain task
        sk_b=SKTaskB('sk_b',output_dir)
        sk_b.retrain_sk(store)
    elif act=='keras_a':
        # a keras model train task
        keras_a=KerasTaskA('keras_a',output_dir)
        keras_a.run_keras_train(store)
    elif act=='keras_b':
        # a exist keras model retrain task
        keras_b=KerasTaskB('keras_b',output_dir)
        keras_b.retrain_keras(store)
        # convert model to other format
        keras_b.convert()
        # evaluate converted model 
        keras_b.evaluate_converted(store)
    #elif act=='test':
    #    import TimeSeriesDataAnalysisLib.interface.trainable_stored_v1_0.structure
    #    TimeSeriesDataAnalysisLib.interface.trainable_stored_v1_0.structure._test()

            

if __name__ == '__main__':
    import argparse
    flags = argparse.ArgumentParser()
    flags.add_argument('--act',  required=True,help='set acting')
    FLAGS = flags.parse_args()
    main(FLAGS.act)