import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..','..'))
import TimeSeriesDataAnalysisLib.interface.untrainable_base as ib
import TimeSeriesDataAnalysisLib.util.time_tool as dt
import numpy as np
from datetime import datetime
from typing import List, TypeVar,Callable,Union,Dict,Any
T = TypeVar('T')
import attr
import abc,os
#VERSION=os.path.basename(__file__).split(".")[0]
VERSION='untrainable_stored_v1_2'
VERSION2='untrainable_unitystored_v1_2'
@attr.s(auto_attribs=True)
class MarkObject(ib.Marshaler):
    mark_name:str=None
    mark_index:int=None
    mark_description:Union[str,None]=None




@attr.s(auto_attribs=True)
class ArrayDataObject(ib.Marshaler,ib.AbcArrayDataObject):
    """
    if not None , shape(0) means rows,every attr must be the same.
    """
    tag:ib.union_list_n_none(1,str)=None
    measure_time:List[datetime]=None
    put_time:List[datetime]=None
    lat:ib.union_list_n_none(1,float)=None
    lon:ib.union_list_n_none(1,float)=None
    cfg:ib.union_list_n_none(3,float)=None
    shape:List[int]=None
    value:np.ndarray=None
    def __attrs_post_init__(self):
        pass
    def _custom_deep_validate(self):
        size=self.shape[0]
        for key in attr.fields_dict(self.__class__).keys():
            v=self.__dict__[key]
            if key=='value':
                if type(v)!=type(None) and list(v.shape)!=self.shape:
                    raise ValueError("{0} must equel to shape".format(key))
            elif key=='cfg' or key=='measure_time':
                size=self.shape[0]
                if type(v)!=type(None) and len(v)!=size:
                    raise ValueError("{0} rows must equel to shape[0]".format(key))
            else:
                if type(v)!=type(None) and len(v)!=size:
                    raise ValueError("{0} rows must equel to shape[0]".format(key))
                validType=attr.fields_dict(self.__class__)[key].type
                is_pass=ib._TypeValidator.validate_each(self,validType,v)
                if not is_pass:
                    raise TypeError("{4} key='{0}' input={1} is required type of {2}, not {3}".format(key,v,validType,type(v),self.__class__))
                pass

TimeSeriesStepFlag=ib.TimeSeriesStepFlag

@attr.s(auto_attribs=True)
class AnnotationObject(ib.Marshaler,ib.AbcAnnotationObject):
    label:Union[str,None]=None
    label_set:Union[List[str],None]=None
    feature_name:List[str]=None
    mark:Union[List[MarkObject],None]=None
    step_flag:Union[TimeSeriesStepFlag,None]=None
    
    def __attrs_post_init__(self):
        pass



@attr.s(auto_attribs=True)
class DataInfoObject(ib.Marshaler,ib.AbcInfoObject):
    ex_ver:str=VERSION
    in_ver:str=None
    fw_ver:Union[str,None]=None
    sn:str=None
    user_sn:Union[str,None]=None
    analysis_term:str=None
    provider:Union[str,None]=None
    term:str=None
    link_set:str=None
    link_id:str=None
    global_addfield:Union[ Dict, None]=None
    device_cfg:Union[Dict,None]=None
    critical:Union[Dict,None]=None
    comment:Union[str,None]=None
    source:Union[str,None]=None
    
    def __attrs_post_init__(self):
        pass

    # def beforeLoadcheck(self,dict_obj:dict,force:bool):
    # move to Marshaler
    #     if not force:
    #         if "ex_ver" in dict_obj.keys():
    #             if self.ex_ver!=dict_obj["ex_ver"]:
    #                 raise ValueError("ex_ver not match, check interface version or add kwargs force=True")
    #         if "in_ver" in dict_obj.keys():
    #             if self.in_ver!=dict_obj["in_ver"]:
    #                 raise ValueError("in_ver not match, check interface version or add kwargs force=True")
@attr.s(auto_attribs=True)    
class ResearchArrayDataObject(ib.Marshaler,ib.AbcMajorObject):
    key:str=None
    index_id:str=None
    info:DataInfoObject=None
    data:ArrayDataObject=None
    annotation:AnnotationObject=None
    
    
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
            if self.data.measure_time[0]==None: raise ValueError("data.measure_time[0]==None")
            self.index_id="{0:.3f}".format(dt.any_datetime_2_utc_timestamp(self.data.measure_time[0]))

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
  


@attr.s(auto_attribs=True)
class FeatureObject(ib.Marshaler):
    name:str=None
    value:float=None
    cfg:Union[List[float],None]=None
    def __attrs_post_init__(self):
        pass
@attr.s(auto_attribs=True)
class UnitDataObject(ib.Marshaler,ib.AbcUnitDataObject):
    measure_time:datetime=None
    put_time:datetime=None
    tag:Union[str,None]=None
    features:List[FeatureObject]=[]
    lat:Union[float,None]=None
    lon:Union[float,None]=None
    def __attrs_post_init__(self):
        pass


class DataUnitInfoObject(DataInfoObject):
    ex_ver:str=VERSION2

class ResearchUnitDataObject(ResearchArrayDataObject):
    info:DataUnitInfoObject=None
    data:List[UnitDataObject]=None
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

