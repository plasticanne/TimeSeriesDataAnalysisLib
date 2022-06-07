
import TimeSeriesDataAnalysisLib.interface.untrainable_base as ib
import TimeSeriesDataAnalysisLib.util.time_tool as dt
from datetime import datetime
from typing import List, TypeVar,Callable,Union
T = TypeVar('T')
import attr
import abc,os
#VERSION=os.path.basename(__file__).split(".")[0]
VERSION='untrainable_stored_v1_1'

@attr.s(auto_attribs=True)
class FeatureObject(ib.BasicObject):
    name:str=None
    value:float=None
    cfg:Union[List[float],None]=None
    def __attrs_post_init__(self):
        pass
@attr.s(auto_attribs=True)
class DataObject(ib.BasicObject,ib.AbcUnitDataObject):
    measure_time:datetime=None
    put_time:datetime=None
    tag:Union[str,None]=None
    features:List[FeatureObject]=[]
    lat:Union[float,None]=None
    lon:Union[float,None]=None
    def __attrs_post_init__(self):
        pass

@attr.s(auto_attribs=True)
class DataInfoObject(ib.BasicObject,ib.AbcInfoObject):
    _ex_ver:str=VERSION
    _in_ver:str=None
    fw_ver:Union[str,None]=None
    sn:str=None
    analysis_term:str=None
    provider:Union[str,None]=None
    term:str=None
    label:Union[str,None]=None
    def __attrs_post_init__(self):
        pass
    @property
    def ex_ver(self):
        return self._ex_ver
    @ex_ver.setter
    def ex_ver(self,value:str):
        if self._ex_ver!=value:
            if not self._forceLoad:
                raise ValueError("ex_ver not match, check interface version or add kwargs force=True")
        
    @property
    def in_ver(self):
        return self._in_ver
    @in_ver.setter
    def in_ver(self,value:str):
        if self._in_ver!=value:
            if not self._forceLoad:
                raise ValueError("in_ver not match, check interface version or add kwargs force=True")
    
@attr.s(auto_attribs=True)    
class ResearchDataObject(ib.BasicObject,ib.AbcMajorObject):
    key:str=None
    index_id:str=None
    info:DataInfoObject=None
    data:List[DataObject]=[]
    def __attrs_post_init__(self):
        pass

    # s3 target path
    def generate_new_key(self,**kwargs):
        """generate new s3 target file path if need

        key="{info.term}/{info.analysis_term}/{info.sn}/{yyyy_mm}/{index_id=measure timestamp}"
        """
        if "index_id" in kwargs:
            if type(kwargs["index_id"])!=str: raise TypeError(kwargs["index_id"])
            self.index_id=kwargs["index_id"]
        else:
            if self.data[0].measure_time==None: raise ValueError("data[0].measure_time==None")
            self.index_id="{0:.3f}".format(dt.any_datetime_2_utc_timestamp(self.data[0].measure_time))

        if "key" in kwargs:
            if type(kwargs["key"])!=str: raise TypeError(kwargs["key"])
            self.key=kwargs["key"]
        else:
            if self.info.term==None: raise ValueError("info.term==None")
            if self.info.analysis_term==None: raise ValueError("info.analysis_term==None")
            if self.info.sn==None: raise ValueError("info.sn==None")
            if self.index_id==None: raise ValueError("index_id==None")
            self.key="{0}/{1}/{2}/{3}/{4}".format(self.info.term,
            self.info.analysis_term,
            self.info.sn,
            dt.yyyy_mm( dt.utc_timestamp_2_utc_datetime( float(self.index_id) ) ),
            self.index_id)

  
        
