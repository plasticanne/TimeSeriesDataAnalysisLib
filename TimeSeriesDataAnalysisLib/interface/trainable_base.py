import os,re,math
import numpy as np
from pandas import DataFrame
import abc

from typing import List,Dict,Union,Tuple,Generator
import logging


class AbcTrainableStore(metaclass=abc.ABCMeta):
    data_size:int
    @abc.abstractmethod
    def get_labels(self)->Union[List[int],None]:
        pass
    @abc.abstractmethod
    def get_indexes(self)->List[int]:
        pass
    @abc.abstractmethod
    def get_names(self)->List[int]:
        pass
    @abc.abstractmethod
    def get_label_by_index(self,index:int)->np.ndarray:
        pass
    @abc.abstractmethod
    def get_value_by_index(self,index:int)->np.ndarray:
        pass



        

