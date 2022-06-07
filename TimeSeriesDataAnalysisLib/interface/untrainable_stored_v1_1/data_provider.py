import os,re,abc
import enum
from typing import List,Dict,Union
import numpy as np
from pandas import DataFrame
import TimeSeriesDataAnalysisLib.util.time_tool as dt
from TimeSeriesDataAnalysisLib.interface.untrainable_stored_v1_1.structure import ResearchDataObject,VERSION,DataObject,FeatureObject
from TimeSeriesDataAnalysisLib.interface.untrainable_base import AbcUntrainableDataProvider

class TimeSeriesMode(enum.Enum):
    """the mode of TimeSeries data
    --------
    index: sort by index

    time: sort by time string
    """
    index=0 
    time=1    

class ResearchDataProvider(AbcUntrainableDataProvider):
    """The time series data with each point has several channels/features, and need to detect TimeSeriesStep
    , should be use this Injection

    args
    ---------
    researchDataObject: ResearchDataObject, the TimeSeriesInjection must with ResearchDataObject instance

    feature_name: List[str], give the name of each channels/features

    required_in_ver: str, must be match ResearchDataObject.info.in_ver for check in same version

    kwargs
    ---------
    mode: TimeSeriesMode, default=TimeSeriesMode.index only TimeSeriesMode.index available

    """
    all:np.ndarray

    research_data_object:ResearchDataObject
    datetime_list:list
    index_array:list
    feature_array:list
    label:Union[int,None]

    #feature name, default is str of index 
    feature_name:List[str]
    is_time_series_step:True
    def __init__(self,researchDataObject:ResearchDataObject,
        feature_name:List[str],
        required_in_ver:str,
        mode:TimeSeriesMode=TimeSeriesMode.index):
        if mode != TimeSeriesMode.index:
            raise ValueError( mode+" not supported now.")

        self.research_data_object=researchDataObject
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
    def feature_process(self,data:DataObject,feature:FeatureObject)->(int,float):
        """override this method for define feature process, the return is for currect each feature in every dataObject of researchDataObject

        args
        ---------
        data: DataObject, the currect processing DataObject

        feature: FeatureObject, the currect processing FeatureObject of currect DataObject

        """
        keys=["s1","s2"]
        for i,key in enumerate(keys):
            if feature.name==key:
                value=feature.value
                return i,value
    

    def _do_label_process(self):
        """acting label_process"""
        self.label=self.label_process()
    
    def _do_feature_process(self)->'TimeSeriesPreprocess':
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
        return self
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

    def get_data_object(self)->ResearchDataObject:
        return self.research_data_object


    def get_feature_name(self)->List[str]:
        return self.feature_name

    def slice_data_size(self,slice_point:np.ndarray)->ResearchDataObject:

        """create new data_object from slice deepcopy data_object of data_provider,

        args
        --------

        slice_point: np.ndarray, the slice point

        return
        --------
        new data_object from slice deepcopy data_object of data_provider

        """
        data_object=copy.deepcopy(self.get_data_object())
        data_object.data=data_object.data[slice_point[0]:slice_point[1]]
        return data_object

    def update_step_flag(self,step_flag)->None:
        raise RuntimeError("not supported of v1_1 version")
        