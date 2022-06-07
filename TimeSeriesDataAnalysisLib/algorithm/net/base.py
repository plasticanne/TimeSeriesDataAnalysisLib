


import abc,os
import numpy as np
from typing import List,Dict,Union,Tuple,Generator,FrozenSet
from TimeSeriesDataAnalysisLib.interface.trainable_data_provider import TrainableDataProvider,AccessSubDataProviderExpand
def my_raise(ex): raise ex
def set_defaut(test,key,default): return default if key not in test else test[key]
def get_name(target):
    try:
        return target.__class__.__name__
    except Exception:
        try :
            return target.__name__ 
        except Exception:
            return target
def concat_generator(gens:List[Generator]):
    """concat multiple generators into 1 generator,
    every generators must to have same iteration steps
    """
    while True:
        result=[gen.next() for gen in gens]
        yield result
class AbcModelBuilder(metaclass=abc.ABCMeta):
    input_shapes:List[Tuple[int]]
    class_num:int
    model:any
    metrics_names:Tuple 
    params_log:dict

    @abc.abstractmethod
    def create_model(self,*args, **kwargs):
        pass
    
    def get_model_name(self)->str:
        return self.model.__class__.__name__
    def isModelExist(self)->bool:
        return False if self.model ==None else True



class AbcWorkflow(AccessSubDataProviderExpand,metaclass=abc.ABCMeta):  
    input_shapes:List[Tuple[int]]
    class_num:int
    model_builder:AbcModelBuilder
    data_provider:TrainableDataProvider
    batch_size:int
    model_name:str =None
    model_root_path:str =None
    
    @abc.abstractmethod
    def set_params_log(self)->dict:
        pass
    
import joblib
import TimeSeriesDataAnalysisLib.util.time_tool as time_tool
class AnyFlow(AccessSubDataProviderExpand ):
    class_num:int
    inputs_shape:List[Tuple[int]]
    model_name:str
    output_dir:str
    model_root_path:str =None
    # def __init__(self,*args, **kwargs):
    #     super(PlotScatterFlow, self).__init__(*args, **kwargs)
    def __init__ (self,model_name=None,output_dir=None,auto_timer_sub_folder=False):
        self.model_name=model_name
        self._set_dump_folder(output_dir,auto_timer_sub_folder)
        
    def _set_dump_folder(self,output_dir,auto_timer_sub_folder):
        if auto_timer_sub_folder:
            date=time_tool.local_date_to_filename(time_tool.get_local_date_now())
            output_path=os.path.join(output_dir,self.model_name+'__'+date)
        else:
            output_path=output_dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_dir=output_path
    def set_flow(self,data_provider,label_set=None,shape=None):
        self.data_provider=data_provider
        self.class_num=len(label_set)
        self.inputs_shape=shape
   
    def save_model(self,model,model_path):
        joblib.dump(model,model_path)
        return 
    def load_model(self,model_path):
        joblib.load(model_path)
        return 
         
    def unzip_keys_from_otherInfos(self)->Tuple[ FrozenSet,Dict[str,list]]:
        # only process train data
        x_train_otherInfo={}
        L=self.data_provider.store.get_otherInfos()
        allkeys = frozenset().union(*L)
        for k in allkeys:
            x_train_otherInfo[k]=[]
            for value in L: 
                if k not in value:
                    x_train_otherInfo[k].append(None)
                else:
                    x_train_otherInfo[k].append(str(value[k]))
        return allkeys,x_train_otherInfo

    def ndarray2xyDict(self,array):
        names = np.array(['x', 'y'])
        return list(map(dict, np.dstack((np.repeat(names[None, :], array.shape[0], axis=0),array ))))
       
    
   
    
   
    



