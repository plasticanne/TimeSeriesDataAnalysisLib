
import abc
import json
import attr
from datetime import datetime
from typing import List, TypeVar,Callable,Union,Type,Tuple,Any,Dict
import numbers
import numpy as np
import decimal
import logging
import re
import TimeSeriesDataAnalysisLib.util.time_tool as dt
import cattr 
import msgpack
T = TypeVar('T')
class _TypeValidator:
    """the validate methods
    """
    @classmethod
    def validate(cls,instance):
        """validate loop for doing validate all set value on interface with right type
        """
        attr_names=attr.fields_dict(instance.__class__).keys()
        for class_attr in attr_names:
            value=instance.__dict__[class_attr]
            validType=attr.fields_dict(instance.__class__)[class_attr].type
            #print(class_attr,value)
            is_pass=cls.validate_each(instance,validType,value)
            if not is_pass:
                raise TypeError("{4} key='{0}' input={1} is required type of {2}, not {3}".format(class_attr,value,validType,type(value),instance.__class__))
        

    @classmethod
    def array_data_object_validate(cls,instance,item):
        size=item.shape[0]
        
        for key in attr.fields_dict(item.__class__).keys():
            v=item.__dict__[key]
            
            if key=='shape' or key=='tag':
                pass
            elif key=='value':
                if type(v)!=type(None) and list(v.shape)!=item.shape:
                    raise ValueError("{0} must equel to shape".format(key))
            elif key=='cfg':
                if type(v)!=type(None) and len(v)!=size:
                    raise ValueError("{0} rows must equel to shape[0]".format(key))
            else:
                if type(v)!=type(None) and len(v)!=size:
                    raise ValueError("{0} rows must equel to shape[0]".format(key))
                
                validType=attr.fields_dict(item.__class__)[key].type
                is_pass=cls.validate_each(instance,validType,v)
                if not is_pass:
                    raise TypeError("{4} key='{0}' input={1} is required type of {2}, not {3}".format(key,v,validType,type(v),item.__class__))
    @classmethod
    def validate_each(cls,instance,validType,value)->bool:
        """doing validate all set value on interface with right type
        """
        uType,allowNone=cls.allowed_attr_type_validate(instance,validType)
        #print(uType,type(value))
        if type(value)==uType:
            # if isinstance(value,instance._base_class) and not isinstance(value,AbcArrayDataObject):
            #     value.validate()  
            #     return True  
            # elif isinstance(value,instance._base_class) and isinstance(value,AbcArrayDataObject):
            #     cls.array_data_object_validate(instance,value)
            #     return True 
            has_custom_deep_validate= hasattr(value, '_custom_deep_validate') and callable(getattr(value, '_custom_deep_validate'))
    
            if has_custom_deep_validate:
                value._custom_deep_validate()
                return True 
            if isinstance(value,instance._base_class) and not has_custom_deep_validate:
                value.validate()  
                return True  
            else:
                return True         
        elif isinstance(value, list):
            for item in value:
                if isinstance(item,instance._base_class):
                    item.validate()    
                else:
                    insideType=uType.__args__[0]
                    cls.validate_each(instance,insideType,item)    
            return True    
        elif isinstance(value, dict):
            
            if uType== Dict :
                return True  
            else:
                return False
        elif value==None and allowNone:
            return True  
        else:
            return False
    @classmethod
    def allowed_attr_type_validate(cls,instance,validType)->(Any,bool):
        """validate self class attr all in allowed type
        """
        allowNone=False
        allowType=[ int,float,str,bool,datetime]
        allowSubclass=[instance._base_class]
        anyEleType=[dict,list]
        allowInstance=['List','Dict']
        try:
            # try if a typing
            validType.__origin__
           
        except AttributeError:
            # just a base typing
            
            #print("name",validType.__name__,validType in allowType)
            if validType in allowType:
                uType=validType
            elif True in list(map(lambda x: issubclass( validType,x) ,allowSubclass)):
                uType=validType
            elif validType in anyEleType:
                if instance._ele_allow_any_type:
                    logging.warning("{0} there are elements of Any type {1}, maybe cause some parse issues".format(instance.__class__,validType))
                    uType=validType
                else:
                    raise TypeError("{0} is not allowed elements of Any type in {1}, instead of List[T]".format(validType,allowType))
            else:
                raise TypeError("{0} is not an allowed type in {1}".format(validType,allowType+allowInstance))
        
        else:
            #print("origin",validType.__origin__)
            # not a base typing
            # is a Union
            if validType.__origin__ is Union:
                if len(validType.__args__)==2 and type(None) in validType.__args__:
                    uType = list(filter(lambda x: x != None, validType.__args__))[0]
                    allowNone=True
                else:
                    raise TypeError("{0} union with NoneType only ".format(validType))
            else: 
                if validType.__name__ in allowInstance:
                    if validType.__args__ is None: 
                        raise TypeError("{0} is not allowed  elements of Any type, instead of list or List[T]".format(allowInstance))
                    else:
                        for arg in validType.__args__:
                            cls.allowed_attr_type_validate(instance,arg)
                    uType=validType
                else:
                    raise TypeError("{0} is not an allowed type in {1}".format(validType,allowType+allowSubclass+allowInstance))
        return uType,allowNone

    @classmethod
    def allowed_value_setattr(cls,instance,validType,uType,allowNone,instance_attr,value,**kwargs):
        """validate the type of input dict value is allowed and setter
        """
        # uType must be not None
        if type(value)==uType: 
            if isinstance(uType(),instance._base_class):
                setattr(instance, instance_attr, value) 
            else:
                setattr(instance, instance_attr, value)  
        # if typing is number typing, but do not parse float/decimal as int
        elif type(value)!=uType \
            and (uType in [float]) \
            and (type(value) in [int,float]):
            logging.warning("{3} key='{0}' input={1} parse as {2}".format(instance_attr,value,uType,instance.__class__))
            setattr(instance, instance_attr, uType(value) ) 
        # input as None
        elif value==None and allowNone: 
            setattr(instance, instance_attr, value)  
        # input as datetime string
        elif type(value)==str and uType==datetime:
            setattr(instance, instance_attr, dt.any_isoformat_2_utc_datetime(value))     
        #parse nested dict interface
        elif isinstance(value, dict) :
            # uType is extends from _base_class
            if isinstance(uType(),instance._base_class):
                # create uType instances 
                setattr(instance, instance_attr, uType().loads(value,**kwargs))
            #not allow dict[any] interface, must be a class interface
            else:
                raise TypeError("{4} key='{0}' input={1} is required type of {2}, not {3}".format(instance_attr,value,validType,type(value),instance.__class__))
        #parse nested list interface
        elif isinstance(value, list):
            insideType=uType.__args__[0] # suppose all insideType of list is the same
            # insideType is extends from _base_class
            if isinstance(insideType(),instance._base_class):
                result:List[insideType]=[]
                for item in value:
                    # create insideType instances 
                     result.append(insideType().loads(item,**kwargs))               
                setattr(instance, instance_attr, result)
            #parse as list[T] 
            else:
                setattr(instance, instance_attr, list(map(uType.__args__[0], value )))        
        else: 
            raise TypeError("{4} key='{0}' input={1} is required type of {2}, not {3}".format(instance_attr,value,validType,type(value),instance.__class__))
class Validator:
    """The validate methods for data interface object typing checking
 
    """
    _validated=False
    _ele_allow_any_type=False
    _base_class=None

    def _get_self_name(self,name:str):
        """
        """
        return list(filter(lambda x: x.__name__ ==name , self.__class__.__bases__ ))[0]
    def builder(self,ele_allow_any_type:bool=False):
        """the init trigger of this instance

        kwargs
        ----------
        ele_allow_any_type: bool, will allow class interface has elements of Any type like: list, dict (not recommend)
        """
        self._validated=False
        if ele_allow_any_type:
            self._ele_allow_any_type=ele_allow_any_type
    def validate(self):
        """validate before dump

        *notice: after this step, all given value type should be well checked, expect datetime need to convert to iso string, and is_validated()==True
        """
        self._base_class=Validator
        _TypeValidator.validate(self)
        self._validated=True
        return self._validated
    
    def is_validated(self):
        """after  validated,  return True
        """
        return self._validated

    def get_attrs(self):
        """return attrs class fields information
        """
        return attr.fields_dict(self.__class__)

class DictEncoder_:
    """the Encoder for string serialization,  Ths default serialization
    
    """
    converter = cattr.Converter()

    converter.register_unstructure_hook(datetime, lambda v: dt.any_datetime_2_loaclzone_isoformat(v))
    converter.register_structure_hook(datetime, lambda v, ty: dt.any_isoformat_2_utc_datetime(v))
    
    converter.register_unstructure_hook(np.ndarray, lambda v: v.tolist())
    converter.register_structure_hook(np.ndarray, lambda v, ty: np.asarray(v, dtype=float))

    #converter.register_structure_hook(Validator, lambda v, ty: DictEncoder_.set_private(v,ty))
    
    
    #converter.register_structure_hook_func(lambda cls: DictEncoder_.get_private(cls), lambda d, t: print(d,t))

    @staticmethod
    def _set_private(v, ty):
        attr_names=attr.fields_dict(ty).keys()  
        #print(v)
        for key in attr_names:
            match=re.match(r'^_.+', key)
            if  match!= None:
                no_d_key=match.group(0)[1:]
                
                if no_d_key in v.keys():
                    setattr(ty, no_d_key,v[no_d_key])
                
            
        
    @staticmethod
    def structure(dict_obj:dict,class_type:Type[T],ele_allow_any_type:bool=False)->T:
        """Ths default unserialization , this will create an instance of class_type

        args
        -----------
        dict_obj: dict,  input dict for unserializing

        class_type: the output instance of class

        ele_allow_any_type: bool, will allow class interface has elements of Any type like: list, dict (not recommend)
    
        """
        _object=DictEncoder_.converter.structure(dict_obj,class_type)
        _object.builder(ele_allow_any_type=ele_allow_any_type)
        return _object
    @staticmethod
    def unstructure(class_instance:Validator)->dict:
        """Ths default serialization 

        args
        -----------
        class_instance:Validator , a extends of Validator could be serializated
    
        """
        #return DictEncoder_.converter.unstructure(class_instance)
        if class_instance.is_validated():
            return DictEncoder_.converter.unstructure(class_instance)
        else:
            raise RuntimeError("Did not do validate()")

    
#static class
DictEncoder= DictEncoder_()

class AdvancedJSONEncoder(json.JSONEncoder):
    """the Encoder for direct json serialization,
    Default is DictEncoder_, not this
    
    """

    def default(self, obj):
        if hasattr(obj, '__jsonencode__'):
            return self.jsonencode(obj)

        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)
    def jsonencode(self,obj):
        """make  json.dumps(self, cls=AdvancedJSONEncoder) as self class,

        detail
        ---------
        the dumps typing check should be pass follow steps
        - is attr value on np.ndarray type convert to list
        - is attr value on datetime type convert to any_datetime_2_utc_isoformat
        """
        def converter(value):
            if type(value)==datetime:
                return dt.any_datetime_2_loaclzone_isoformat(value)
            elif type(value)==np.ndarray:
                return value.tolist()
            else:
                return value

        attr_names=attr.fields_dict(obj.__class__).keys()        
        result={}
        
        for key in attr_names:
            class_attr=re.sub(r'^_', '', key)
            value=obj.__dict__[key]
            
            if isinstance(value,list):
                result[class_attr]=[]
                for i in range(len(value)):
                    result[class_attr].append(converter(value[i]))
            else:
                result[class_attr]= converter(value)
        
        return result
    
class Marshaler(Validator):
    def _beforeLoadcheck(self,dict_obj:dict,force:bool):
        if isinstance(self,AbcInfoObject):
            if not force:
                if dict_obj["in_ver"] != self.in_ver: raise ValueError("in_ver not match")
                #if dict_obj["ex_ver"] != self.ex_ver: raise ValueError("ex_ver not match")

            
    def loads(self,dict_obj:dict,force:bool=False,ele_allow_any_type:bool=False):
        """deserialize input dict object as this class instance,
        args
        ---------
        obj_dict: dict,  nested object in dict/list or nested dict/list in object both should be working

        kwargs
        ---------
        force: bool, if interface version not match but still force parse,

        ele_allow_any_type: bool, will allow class interface has elements of Any type like: list, dict (not recommend)
        
        detail
        ---------
        the typing check should be pass follow steps
        - not check input type with attr when loading
        

        *notice: after this step, the user may not well processing yet, so not check pre-given or had given value type 
        """
        self._beforeLoadcheck(dict_obj,force)
        self.__dict__=DictEncoder.structure(dict_obj,self.__class__,ele_allow_any_type=ele_allow_any_type).__dict__
        return self
    def dumps(self,format:str='json')->str:
        """dump as string
        kwargs
        ---------
        format: 'json' | 'msgpack'

        """
        if format=='json':
            if not self.is_validated():
                raise RuntimeError("Did not do validate()")
            return json.dumps(self, cls=AdvancedJSONEncoder, ensure_ascii=False)
        elif format=='msgpack':
            msgstr=DictEncoder.unstructure(self)
            return msgpack.packb(msgstr)
        else:
            raise ValueError("no format {0}".format(format))
    def dumpBytes(self,format:str='json')->bytes:
        """dump as bytes
        kwargs
        ---------
        format: 'json' | 'msgpack'
        """
        if format=='json':
            return self.dumps(format=format).encode()
        elif format=='msgpack':
            msgstr=DictEncoder.unstructure(self)
            
            return msgpack.packb(msgstr)
        else:
            raise ValueError("no format {0}".format(format))
        
    def __jsonencode__(self):
        pass



class BasicObject:
    _forceLoad=False
    _dumpNone=False
    _ele_allow_any_type=False
    _base_class=None
    def loads(self,obj_dict:dict,force:bool=False,ele_allow_any_type:bool=False):
        """deserialize input dict object as this class instance,
        args
        ---------
        obj_dict: dict,  nested object in dict/list or nested dict/list in object both should be working

        kwargs
        ---------
        force: bool, if interface version not match but still force parse,

        ele_allow_any_type: bool, will allow class interface has elements of Any type like: list, dict (not recommend)
        
        detail
        ---------
        the typing check should be pass follow steps
        - is attr type allowed on self class (_allowed_attr_type_validate)
        - is input value type match attr type (_allowed_value_setattr)
        - nested object check loop

        *notice: after this step, the user may not well processing yet, so not check pre-given or had given value type 
        """
        
        if force:
            self._forceLoad=force
        if ele_allow_any_type:
            self._ele_allow_any_type=ele_allow_any_type
        if isinstance(obj_dict,self.__class__):
            self=obj_dict
            return self
        attr_names=attr.fields_dict(self.__class__).keys()
        #print(attr_names)
        for class_attr in attr_names:
            key=re.sub(r'^_', '', class_attr)
            validType=attr.fields_dict(self.__class__)[class_attr].type
            #print(class_attr)
            uType,allowNone=_TypeValidator.allowed_attr_type_validate(self,validType)
            if key in obj_dict.keys():
                self._allowed_value_setattr(validType,uType,allowNone, key, obj_dict[key], force=force)
        #print(attr.fields_dict(self.__class__))
        return self
    def validate(self):
        """validate before dump

        detail
        ---------
        the typing check should be pass follow steps
        - is attr type allowed on self class (_allowed_attr_type_validate)
        - all given value's type (_validate_each)
        - nested check

        *notice: after this step, all given value type should be well checked, expect datetime need to convert to iso string
        """
        self._base_class=BasicObject
        _TypeValidator.validate(self)
        return self

    def _allowed_value_setattr(self,validType,uType,allowNone,instance_attr,value,**kwargs):
        """validate the type of input dict value is allowed and setter
        """
        self._base_class=BasicObject
        _TypeValidator.allowed_value_setattr(self,validType,uType,allowNone,instance_attr,value,**kwargs)
    def dumps(self,dumpsNone=True)->str:
        """nested dump as json string
        - validate type (validate)
        """
        self._dumpNone=dumpsNone
        self.validate()
        return json.dumps(self, cls=AdvancedJSONEncoder, ensure_ascii=False)
    def dumpBytes(self,dumpsNone=True)->bytes:
        """nested dump as json bytes"""
        return self.dumps(dumpsNone=dumpsNone).encode()
    def __jsonencode__(self):
        pass
def union_list_n_none(d:int,ty:T):
    seed=Union[ty,None]
    for i in range(d):
        seed=Union[List[seed],None]
    return seed

@attr.s(auto_attribs=True)
class AbcUnitDataObject(metaclass=abc.ABCMeta):
    pass
class AbcArrayDataObject(metaclass=abc.ABCMeta):
    value:list
    shape:List[int]
@attr.s(auto_attribs=True)
class AbcInfoObject(metaclass=abc.ABCMeta):
    pass
@attr.s(auto_attribs=True)
class AbcMajorObject(metaclass=abc.ABCMeta):
    key:str
    index_id:str
    info:AbcInfoObject
    data:any

    # s3 target path
    @abc.abstractmethod
    def generate_new_key(self,**kwargs):
        pass


@attr.s(auto_attribs=True)
class AbcAnnotationObject(metaclass=abc.ABCMeta):
    pass




class AbcUntrainableDataProvider(metaclass=abc.ABCMeta):
    is_time_series_step:bool
    @abc.abstractmethod
    def get_all(self):
        pass
    def get_datetime_list(self)->List:
        pass
    @abc.abstractmethod
    def get_data_object(self):
        pass
    @abc.abstractmethod
    def get_feature_name(self):
        pass
    
@attr.s(auto_attribs=True)
class TimeSeriesStepFlag(Marshaler):
    state0_start:Union[int,None]=None
    state0_end:Union[int,None]=None
    state1_start:Union[int,None]=None
    state1_end:Union[int,None]=None
    state2_start:Union[int,None]=None
    state2_end:Union[int,None]=None
    state3_start:Union[int,None]=None
    state3_end:Union[int,None]=None
    state4_start:Union[int,None]=None
    state4_end:Union[int,None]=None
    state5_start:Union[int,None]=None
    state5_end:Union[int,None]=None