import os,re,abc
import copy 
import enum
from typing import List,Dict,Union
import numpy as np
from pandas import DataFrame
import TimeSeriesDataAnalysisLib.util.time_tool as dt
from TimeSeriesDataAnalysisLib.interface.untrainable_stored_v1_2.structure import ResearchUnitDataObject,VERSION,UnitDataObject,FeatureObject,ResearchArrayDataObject
from TimeSeriesDataAnalysisLib.interface.untrainable_base import AbcUntrainableDataProvider,TimeSeriesStepFlag
import TimeSeriesDataAnalysisLib.algorithm.channel_mapping as cm
class TimeSeriesMode(enum.Enum):
    """the mode of TimeSeries data
    --------
    index: sort by index

    time: sort by time string
    """
    index=0 
    time=1    
class ResearchArrayDataProvider(AbcUntrainableDataProvider):
    """The time series data with each point has several channels/features, and need to detect TimeSeriesStep
    , should be use this DataProvider for  Research ArrayData Object

    args
    ---------
    researchDataObject: ResearchArrayDataObject, the DataProvider must with ResearchArrayDataObject instance


    required_in_ver: str, must be match researchDataObject.info.in_ver for check in same version

    kwargs
    ---------
    mode: TimeSeriesMode, default=TimeSeriesMode.index only TimeSeriesMode.index available

    """
    all:np.ndarray
    research_data_object:ResearchArrayDataObject
    datetime_list:list
    index_array:list
    feature_array:list
    label:Union[int,None]
    feature_name:List[str]
    is_time_series_step=True
    def __init__(self,researchDataObject:ResearchArrayDataObject,
        required_in_ver:str,
        mode:TimeSeriesMode=TimeSeriesMode.index):
        if mode != TimeSeriesMode.index:
            raise ValueError( mode+" not supported now.")
        self.research_data_object=researchDataObject
        if self.research_data_object.info.ex_ver!=VERSION:
            raise Exception("untrainable interface version is not match")
        if self.research_data_object.info.in_ver!=required_in_ver:
            raise Exception("project interface version is not match")
        self.feature_name=self.research_data_object.annotation.feature_name
        
        self._do_feature_process()
        self._do_label_process()
        
    def label_process(self)->Union[int,None]:
        """override this method for how to get label
        if retrun None means this Injection whih no label,
        the process is for currect research_data_object
        """
        labelmap={
            'lable0':0,
            'lable1':1,
            'lable2':2,
            }
        if self.research_data_object.annotation.label !=None:
            return labelmap[self.research_data_object.annotation.label]
        else:
            return None
    def feature_process(self)->dict:
        """convert unit form different category value by key with cfg value
        
        override this method for define feature process, the return is a method dict with different condition by middle sensor name ,  "other" for others not aiming middle names.

        args
        ---------
        data: DataObject, the currect processing DataObject

        feature: FeatureObject, the currect processing FeatureObject of currect DataObject

        """
        return copy.deepcopy(cm.CHANNEL_CATEGORY)
 
    def _do_label_process(self)->None:
        """acting label_process"""
        self.label=self.label_process()
    
    def _do_feature_process(self)->None:
        """acting feature_process"""
        data=self.research_data_object.data
        alg_set=self.feature_process()
        for i,name in enumerate( self.feature_name):
            alg=name.split("#")[1]
            if alg in alg_set.keys():
                alg_set[alg]["set"].append(i)
            else:
                alg_set["other"]["set"].append(i)
        self.all=np.zeros(data.shape)
        for algkey,algvalue in alg_set.items():
            if algvalue["set"]!=[]:
                m=algvalue["alg"]
                cols=algvalue["set"]
                app=[]
                for ro in data.cfg:
                    app.append(  [ro[i] for i in cols])
                self.all[:,cols]=m(data.value[:,cols] ,np.asarray(  app   ))
        self.datetime_list =data.measure_time
        self.index_array =[x for x in range(data.shape[0])]
    def get_datetime_list(self)->List:
        return self.datetime_list
 
    def get_all(self)->np.ndarray:
        return self.all

    def get_data_object(self)->ResearchArrayDataObject:
        return self.research_data_object


    def get_feature_name(self,only_index=False)->List[str]:
        """get the feature name 

        kwargs
        ------------
        only_index: bool, if True, returns the index string list of features
        """
        if only_index:
            return [ str(i) for i in range(len(self.feature_name))]
        else:
            return self.feature_name
    def slice_data_size(self,slice_point:np.ndarray)->ResearchArrayDataObject:

        """create new data_object from slice deepcopy data_object of data_provider,

        args
        --------

        name: str, output filename

        slice_point: np.ndarray, the slice point

        return
        --------
        new data_object from slice deepcopy data_object of data_provider

        """
        data_object=copy.deepcopy(self.get_data_object())
        for key,value in data_object.data.items():
            if key != 'shape' and value != None:
                value=value[slice_point[0]:slice_point[1]]
        self._slice_mark(data_object,slice_point)
        self._slice_step_flag(data_object,slice_point)
        return data_object

    def _slice_mark(self,data_object:ResearchArrayDataObject,slice_point:np.ndarray)->ResearchArrayDataObject:
        if data_object.annotation.mark != None:
            for mark in data_object.annotation.mark:
                mark.mark_index=mark.mark_index-slice_point[0]
        return data_object
    def _slice_step_flag(self,data_object:ResearchArrayDataObject,slice_point:np.ndarray)->ResearchArrayDataObject:
        if data_object.annotation.step_flag != None:
            for key,value in data_object.annotation.step_flag.__dict__.items():
                data_object.annotation.step_flag.__dict__[key]=value-slice_point[0]
        return data_object
    def update_step_flag(self,step_flag:TimeSeriesStepFlag)->ResearchArrayDataObject:
        """return a deepcopy of data_object
        and update the copy of data_object.annotation.step_flag 

        args
        ------------
        step_flag: TimeSeriesStepFlag, the update value
        """
        data_object=copy.deepcopy(self.get_data_object())
        data_object.annotation.step_flag=step_flag
        return data_object
class ResearchUnitDataProvider(AbcUntrainableDataProvider):
    """The time series data with each point has several channels/features, and need to detect TimeSeriesStep
    , should be use this DataProvider for  Research UnitData Object

    args
    ---------
    researchDataObject: ResearchUnitDataObject, the DataProvider must with ResearchUnitDataObject instance

    feature_name: List[str], give the name of each channels/features

    required_in_ver: str, must be match researchDataObject.info.in_ver for check in same version

    kwargs
    ---------
    mode: TimeSeriesMode, default=TimeSeriesMode.index only TimeSeriesMode.index available

    """
    all:np.ndarray

    research_data_object:ResearchUnitDataObject
    datetime_list:list
    index_array:list
    feature_array:list
    label:Union[int,None]

    #feature name, default is str of index 
    feature_name:List[str]

    def __init__(self,researchDataObject:ResearchUnitDataObject,
        feature_name:List[str],
        required_in_ver:str,
        mode:TimeSeriesMode=TimeSeriesMode.index):
        if mode != TimeSeriesMode.index:
            raise ValueError( mode+" not supported now.")

        self.research_data_object=ResearchUnitDataObject
        self.feature_name=feature_name
        if self.research_data_object.info.ex_ver!=VERSION:
            raise Exception("untrainable interface version is not match")
        if self.research_data_object.info.in_ver!=required_in_ver:
            raise Exception("project interface version is not match")
   
        self._do_feature_process()
        self._do_label_process()
        self.all=self._get_all()


    def label_process(self)->Union[int,None]:
        """override this method for how to get label
        if retrun None means this Injection whih no label,
        the process is for currect research_data_object
        """
        labelmap={
            'lable0':0,
            'lable1':1,
            'lable2':2,
            }
        if self.research_data_object.info.label !=None:
            return labelmap[self.research_data_object.info.label]
        else:
            return None
    def feature_process(self,data:UnitDataObject,feature:FeatureObject)->(int,float):
        """override this method for define feature process, the return is for currect each feature in every dataObject of researchDataObject
        

        args
        ---------
        data: DataObject, the currect processing DataObject

        feature: FeatureObject, the currect processing FeatureObject of currect DataObject

        """
        keys=["s_1","s_2"]
        for i,key in enumerate(keys):
            if feature.name==key:
                value=feature.value
                return i,value
    

    def _do_label_process(self)->None:
        """acting label_process"""
        self.label=self.label_process()
    
    def _do_feature_process(self)->None:
        """acting feature_process"""
        sized=0
        sortedArrayTime=[]
        featureArray=[]
        for index,data in enumerate(self.research_data_object.data):
            # processing every DataObject
            size=len(data.features)
            if size==0:
                raise ValueError("this record is empty, {0}".format(data))
            elif size!=sized and sized!=0:
                raise ValueError("this record is with different features size, {0}".format(data))
            else:
                sized=size
                sortedArray = [None] * size
                for feature in data.features:
                    # processing every features
                    index,value=self.feature_process(data,feature)
                    sortedArray[index]=value
                    
                if len(list(filter(lambda x: type(x) != float, sortedArray)))>0:
                    raise ValueError("there are some invilad value in array, {0}".format(sortedArray))
            
            #data.__class__.__name__='DataObject_TimeSeriesPreprocess'
            #data.featureArray=sortedArray
            featureArray.append(sortedArray)
            #sortedArrayTime.append( int(dt.any_datetime_2_utc_timestamp(data.measure_time)))
            sortedArrayTime.append( data.measure_time)
            #self.research_data_object.data[index]=data
        #self.research_data_object.timestampArray=sortedArrayTime
        self.feature_array =featureArray
        self.datetime_list =sortedArrayTime
        self.index_array =[x for x in range(len(sortedArrayTime))]
    def get_datetime_list(self)->List:
        return self.datetime_list 
 
    def _get_all(self)->np.ndarray:
        """ get orginal full size feature processed data
        """
        result=[]
        start_index=0
        end_index=len(self.datetime_list)-1
        for index in range(len(self.datetime_list)):
            if index >= start_index and index <= end_index:
                result.append( self.feature_array[index]) 
        column_keys=self.feature_name
      
        result= np.asarray( result   )
        
        if len(result.shape)!=2:
            raise ValueError("featureArray have wrong shape, this is not an expected data format")
        if result.shape[0]!=(end_index-start_index+1):
            raise ValueError("""The time series data step must be mutually exclusive of every time index.
        Expects the tail-head between stepes are continuous, non-cross and not repeating.""")
        print(result.shape)
        return result

    def get_all(self)->np.ndarray:
        return self.all

    def get_data_object(self)->ResearchUnitDataObject:
        return self.research_data_object


    def get_feature_name(self,only_index:bool=False)->List[str]:
        """get the feature name 

        kwargs
        ------------
        only_index: bool, if True, returns the index string list of features
        """
        if only_index:
            return [ str(i) for i in range(len(self.feature_name))]
        else:
            return self.feature_name
    def slice_data_size(self,slice_point:np.ndarray)->ResearchUnitDataObject:

        """create new data_object from slice deepcopy data_object of data_provider,

        args
        --------

        name: str, output filename

        slice_point: np.ndarray, the slice point

        return
        --------
        new data_object from slice deepcopy data_object of data_provider

        """
        data_object=copy.deepcopy(self.get_data_object())
        data_object.data=data_object.data[slice_point[0]:slice_point[1]]
        self._slice_mark(data_object,slice_point)
        self._slice_step_flag(data_object,slice_point)
        return data_object

    def _slice_mark(self,data_object:ResearchUnitDataObject,slice_point:np.ndarray)->ResearchUnitDataObject:
        if data_object.annotation.mark != None:
            for mark in data_object.annotation.mark:
                mark.mark_index=mark.mark_index-slice_point[0]
        return data_object
    def _slice_step_flag(self,data_object:ResearchUnitDataObject,slice_point:np.ndarray)->ResearchUnitDataObject:
        if data_object.annotation.step_flag != None:
            for key,value in data_object.annotation.step_flag.__dict__.items():
                data_object.annotation.step_flag.__dict__[key]=value-slice_point[0]
        return data_object
    def update_step_flag(self,step_flag:TimeSeriesStepFlag)->ResearchUnitDataObject:
        """return a deepcopy of data_object
        and update the copy of data_object.annotation.step_flag 

        args
        ------------
        step_flag: TimeSeriesStepFlag, the update value
        """
        data_object=copy.deepcopy(self.get_data_object())
        data_object.annotation.step_flag=step_flag
        return data_object