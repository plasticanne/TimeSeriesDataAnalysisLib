import os,re,abc
import enum
from typing import List,Dict,Union
import numpy as np
import pandas as pd
from pandas import DataFrame
import TimeSeriesDataAnalysisLib.util.time_tool as dt
from TimeSeriesDataAnalysisLib.interface.untrainable_stored_v2_0.structure import VERSION,ResearchArrayDataObject
from TimeSeriesDataAnalysisLib.interface.untrainable_base import AbcUntrainableDataProvider,TimeSeriesStepFlag
import TimeSeriesDataAnalysisLib.algorithm.channel_mapping as cm
from datetime import datetime
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
    is_time_series_step:False
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
        self.is_time_series_step=False if self.research_data_object.annotation.step_flag is None else True
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
        """override this method for define feature process, the return is a method dict with different condition by middle sensor name ,  "other" for others not aiming middle names.

        args
        ---------
        data: DataObject, the currect processing DataObject

        feature: FeatureObject, the currect processing FeatureObject of currect DataObject

        """
        #Cfg0,Cfg1,Cfg2,Cfg3,Cfg4,Cfg5
        

        return cm.CHANNEL_CATEGORY
 
    def _do_label_process(self)->None:
        """acting label_process"""
        self.label=self.label_process()
    
    def _do_feature_process(self)->None:
        """acting feature_process"""
        data=self.research_data_object.data
        alg_set=self.feature_process()
        self.all=cm.feature_process_all_rows(data.value,alg_set,self.feature_name,data.cfg)
        self.datetime_list =data.measure_time
        self.index_array =[x for x in range(data.shape[0])]
    def get_datetime_list(self)->List:
        return self.datetime_list
 
    def get_all(self)->np.ndarray:
        return self.all

    def get_data_object(self)->ResearchArrayDataObject:
        return self.research_data_object
    def get_ts_point(self):
        step_flag=self.research_data_object.annotation.step_flag
        if step_flag==None:
            raise ValueError("no step_flag")
        result=[
        [step_flag.state0_start,step_flag.state0_end],
        [step_flag.state1_start,step_flag.state1_end],
        [step_flag.state2_start,step_flag.state2_end],
        [step_flag.state3_start,step_flag.state3_end],
        [step_flag.state4_start,step_flag.state4_end],
        [step_flag.state5_start,step_flag.state5_end]]  
        result=pd.DataFrame( 
        zip(*result),
        columns=[value.value for value in TimeSeriesStep.__members__.values()]).astype('Int64')
        # https://pandas.pydata.org/pandas-docs/version/0.24/whatsnew/v0.24.0.html#optional-integer-na-support
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isna.html#pandas.isna
        
        return result

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
            if type(value).__module__ == np.__name__:
                value=value[slice_point[0]:slice_point[1]]
                continue
            if type(value)!=list:
                continue
            if len(value)==data_object.data.shape[0]:
                value=value[slice_point[0]:slice_point[1]]
        data_object.data.shape=data_object.data.value.shape
        
        self._slice_mark(data_object,slice_point[0])
        self._slice_step_flag(data_object,slice_point[0])
        self._slice_control_status(data_object,slice_point[0])
        data_object.data.sub_array_data.value=data_object.data.sub_array_data.value[slice_point[0]:slice_point[1]]
        data_object.data.sub_array_data.shape=data_object.data.sub_array_data.value.shape
        data_object.data.sub_object_data.value=data_object.data.sub_object_data.value[slice_point[0]:slice_point[1]]
        data_object.data.sub_object_data.shape=data_object.data.sub_object_datavalue.shape
        return data_object

    def slice_data_size_by_time(self,start_time:datetime,end_time:datetime)->ResearchArrayDataObject:

        """create new data_object from slice deepcopy data_object of data_provider,

        args
        --------

        name: str, output filename

        start_time: datetime, start time of slice

        end_time: datetime, start time of slice

        return
        --------
        new data_object from slice deepcopy data_object of data_provider

        """
        data_object=copy.deepcopy(self.get_data_object())
        
        enable_points=self._sliced_index_by_time(start_time,end_time)
        for key,value in data_object.data.items():
            if type(value).__module__ == np.__name__:
                value=value[enable_points[0]:enable_points[-1]]
                continue
            if type(value)!=list:
                continue
            if len(value)==data_object.data.shape[0]:
                value=value[enable_points[0]:enable_points[-1]]
            
        data_object.data.shape=data_object.data.value.shape
        self._slice_mark(data_object,enable_points[0])
        self._slice_step_flag(data_object,enable_points[0])
        self._slice_control_status(data_object,enable_points[0])
        data_object.data.sub_array_data.value=data_object.data.sub_array_data.value[enable_points[0]:enable_points[-1]]
        data_object.data.sub_array_data.shape=data_object.data.sub_array_data.value.shape
        data_object.data.sub_object_data.value=data_object.data.sub_object_data.value[enable_points[0]:enable_points[-1]]
        data_object.data.sub_object_data.shape=data_object.data.sub_object_datavalue.shape
        return data_object
    def _sliced_index_by_time(self,start_time:datetime,end_time:datetime):
        df=pd.DataFrame()
        df['date']=pd.to_datetime(self.get_datetime_list())
        mask = (df['date'] >= start_time) & (df['date'] <= end_time)
        return df.loc[mask].index.values

    def _slice_mark(self,data_object:ResearchArrayDataObject,start_index:int)->ResearchArrayDataObject:
        if data_object.annotation.mark != None:
            for mark in data_object.annotation.mark:
                # set new index of sliced
                mark.data_index=mark.data_index-start_index
        return data_object
    def _slice_control_status(self,data_object:ResearchArrayDataObject,start_index:int)->ResearchArrayDataObject:
        if data_object.annotation.mark != None:
            for control_status in data_object.annotation.control_status:
                # set new index of sliced
                control_status.data_index=control_status.data_index-start_index
        return data_object
    

    def _slice_step_flag(self,data_object:ResearchArrayDataObject,start_index:int)->ResearchArrayDataObject:
        if data_object.annotation.step_flag != None:
            for key,value in data_object.annotation.step_flag.__dict__.items():
                # set new index of sliced
                data_object.annotation.step_flag.__dict__[key]=value-start_index
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
