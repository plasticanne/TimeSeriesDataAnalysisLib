import numpy as np
from pandas import DataFrame

from typing import List,Dict,Callable,Any,Union,Tuple
from TimeSeriesDataAnalysisLib.interface.trainable_base import AbcTrainableStore
import TimeSeriesDataAnalysisLib.algorithm.batch_algorithm as ba
import logging

class TrainableDataProvider:
    """DataProvider define how data feed to analysis instance,
    goal to specify indexes and train ratio for simplify lazy load data 

    args
    ---------
    store: AbcTrainableStore,
        a store instance

    """
    _x_train_indexes:np.ndarray
    _x_valid_indexes:np.ndarray
    _x_test_indexes:np.ndarray
    _y_train_values:np.ndarray
    _y_valid_values:np.ndarray
    _y_test_values:np.ndarray
    _is_x_set:bool
    _is_y_set:bool
    store:AbcTrainableStore
    def __init__ (self,store:AbcTrainableStore,label_set:List[str]):
        self.store=store
        self._is_x_set=False
        self._is_y_set=False
        self.label_set=label_set

    @property
    def x_train_indexes(self):
        """return x indexes list as ndarray
        """
        return self._x_train_indexes
    @property
    def x_valid_indexes(self):
        """return x indexes list as ndarray
        """
        return self._x_valid_indexes
    @property
    def x_test_indexes(self):
        """return x indexes list as ndarray
        """
        return self._x_test_indexes
    @property
    def y_train_values(self):
        """return label value list as ndarray
        """
        self._check_y()
        return self._y_train_values
    @property
    def y_valid_values(self):
        """return label value list as ndarray
        """
        self._check_y()
        return self._y_valid_values
    @property
    def y_test_values(self):
        """return label value list as ndarray
        """
        self._check_y()
        return self._y_test_values
    def x_setter_by_store(self,train_ratio:float,valid_ratio:float,test_ratio:float,shuffle:bool=True,random_seed:int=10101)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Setting ratio of train, valid, test. This action base on the deepcopy indexes list of store.
        return indexes list as ndarray (x_train,x_valid,x_test)
        """
        self._x_train_indexes,self._x_valid_indexes,self._x_test_indexes=ba.indexes_split(self.store.get_indexes(),train_ratio,valid_ratio,test_ratio,shuffle=shuffle,random_seed=random_seed)
        self._is_x_set=True
        return self._x_train_indexes.copy(),self._x_valid_indexes.copy(),self._x_test_indexes.copy()
    def _check_input_1d(self,input:np.ndarray):
        if len(input.shape)!=1:
            raise ValueError("Input must be 1d ndarray")
    def x_setter_by_input(self,indexes_train:np.ndarray,indexes_valid:np.ndarray,indexes_test:np.ndarray)->None:
        """input any indexes ndarray as a train/valid/test list
        """
        self._check_input_1d(indexes_train)
        self._check_input_1d(indexes_valid)
        self._check_input_1d(indexes_test)
        self._x_train_indexes=indexes_train
        self._x_valid_indexes=indexes_valid
        self._x_test_indexes=indexes_test
        msg='store size: {0}, train_size: {1}, valid_size: {2}, test_size: {3}'.format(self.store.data_size,len(self._x_train_indexes),len(self._x_valid_indexes),len(self._x_test_indexes))
        logging.info(msg)
        self._is_x_set=True
    def y_setter_by_store(self)->None:
        """only enable after x be set and labeled store. This action is mapping label value to match train/valid/test indexes list
        """
        self._check_x()
        y=np.asarray(self.store.get_labels())
        self._y_train_values=y[self._x_train_indexes]
        self._y_valid_values=y[self._x_valid_indexes]
        self._y_test_values=y[self._x_test_indexes]
        self._is_y_set=True
    def y_setter_by_input(self,labels_train:np.ndarray,labels_valid:np.ndarray,labels_test:np.ndarray)->None:
        """input any label value ndarray as a train/valid/test list
        """
        self._check_x()
        self._check_input_1d(labels_train)
        self._check_input_1d(labels_valid)
        self._check_input_1d(labels_test)
        if labels_train.shape[0] != self._x_train_indexes.shape[0]:
            raise ValueError("train label shape not match x_train")
        if labels_valid.shape[0] != self._x_valid_indexes.shape[0]:
            raise ValueError("valid label shape not match x_valid")
        if labels_test.shape[0] != self._x_test_indexes.shape[0]:
            raise ValueError("test label shape not match x_test")
        self._y_train_values=labels_train
        self._y_valid_values=labels_valid
        self._y_test_values=labels_test
        self._is_y_set=True
    def _check_x(self):
        if not self._is_x_set:
            raise ValueError("x should be set by x_setter_by_store() or x_setter_by_input() before run task")
    def _check_y(self):
        if not self._is_y_set:
            raise ValueError("y should be set by y_setter_by_store() or y_setter_by_input() before run task")

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
    def load_batch_x_values(self,x_indexes:np.ndarray)->np.ndarray:
        """Always use x value by this method. This will check not an empty value,
        and load x values after pre_process.    
        Override "pre_process_x_value" method if you need.
        """
        self._check_x()
        data_x=[]
        for index in x_indexes:
            x_value=self.store.get_value_by_index(index)
            data_x.append(self.pre_process_x_value(x_value))
            #print(data.shape)

        return np.concatenate(data_x, axis=0) if data_x!=[] else np.asarray(data_x)
    def load_batch_y_values(self,y_values:np.ndarray)->np.ndarray:
        """Always use y value by this method. This will check not an empty value,
        and load y values after pre_process.    
        Override "pre_process_y_value" method if you need.
        """
        self._check_y()
        result=np.apply_along_axis(self.pre_process_y_value,0,y_values)
        #print(y_values)
        #print(result)
        return result
    def data_generator(self,x_indexes:np.ndarray, y_values:np.ndarray, class_num:int, batch_size:int,  batch_shuffle=True):
        '''data generator for fit_generator.
        notice that in this case, only have 1 input node and 1 output node
        override this method if you have multiple inputs, labels.    
        
        '''
        # x hstack y for shuffle together
        self._check_x()
        self._check_y()
        x_y= np.hstack((np.expand_dims( x_indexes,axis=1) , ba.one_hot_encoder( y_values,class_num) ))
        size = x_indexes.shape[0]
        i = 0
        while True:
            x = []
            y = []
            for b in range(batch_size):
                if i==0 and batch_shuffle:
                    np.random.shuffle(x_y)
                    #np.random.seed(None)
                index=int(x_y[i,0])
                x.append(self.pre_process_x_value(self.store.get_value_by_index(index)))
                y.append(self.pre_process_y_value(x_y[i,1:]  ))
                i = (i+1) % size
            # if you have multiple input node
            # you should change data_generator as 
            # inputs=[x1,x2],labels=[y1,y2]
            inputs=[np.vstack( x ),]
            labels=[np.vstack( y ),]
            
            yield inputs, labels
    def convert_label_int_to_name(self,value:int)->str:
        return self.label_set[value]
    def convert_label_int_to_name_ndarray(self,value:np.ndarray)->np.ndarray:
        return np.array([self.label_set[xi] for xi in value])

    def load_batch_x_otherInfos(self,x_indexes:np.ndarray)->list:
        self._check_x()
        data_x=[]
        if hasattr(self.store, 'get_otherInfos'):
            all_otherInfos=self.store.get_otherInfos()
            if all_otherInfos is None: return None
        else:
            return None
        for index in x_indexes:
            x_otherInfo=all_otherInfos[index]
            data_x.append(x_otherInfo)
        return data_x

class AccessSubDataProviderExpand:
    def load_x_value_ndarray(self,category:str)->np.ndarray:
        if category=='train':
            return self.data_provider.load_batch_x_values(self.data_provider.x_train_indexes)
        elif category=='valid':
            return self.data_provider.load_batch_x_values(self.data_provider.x_valid_indexes)
        elif category=='test':
            return self.data_provider.load_batch_x_values(self.data_provider.x_test_indexes)
        else:
            raise Exception("no this category of x data")


    def load_y_value_ndarray(self,category:str)->np.ndarray:
        if category=='train':
            return self.data_provider.load_batch_y_values(self.data_provider.y_train_values)
        elif category=='valid':
            return self.data_provider.load_batch_y_values(self.data_provider.y_valid_values)
        elif category=='test':
            return self.data_provider.load_batch_y_values(self.data_provider.y_test_values)
        else:
            raise Exception("no this category of y data")
    
    def convert_label_int_to_name(self,value:int)->str:
        return self.data_provider.convert_label_int_to_name(value)
    def convert_label_int_to_name_ndarray(self,value:np.ndarray)->np.ndarray:
        return self.data_provider.convert_label_int_to_name_ndarray(value)
    def get_label_set(self)->list:
        return self.data_provider.label_set
    def get_x_otherInfos_list(self,category:str)->list:
        if category=='train':
            return self.data_provider.load_batch_x_otherInfos(self.data_provider.x_train_indexes)
        elif category=='valid':
            return self.data_provider.load_batch_x_otherInfos(self.data_provider.x_valid_indexes)
        elif category=='test':
            return self.data_provider.load_batch_x_otherInfos(self.data_provider.x_test_indexes)
        else:
            raise Exception("no this category of x data")

    def get_x_size(self,category:str)->int:
        if category=='train':
            return self.data_provider.x_train_indexes.shape[0]
        elif category=='valid':
            return self.data_provider.x_valid_indexes.shape[0]
        elif category=='test':
            return self.data_provider.x_test_indexes.shape[0]
        else:
            raise Exception("no this data category")
    def get_y_size(self,category:str)->int:
        if category=='train':
            return self.data_provider.y_train_indexes.shape[0]
        elif category=='valid':
            return self.data_provider.y_valid_indexes.shape[0]
        elif category=='test':
            return self.data_provider.y_test_indexes.shape[0]
        else:
            raise Exception("no this data category")
    def get_x_index_ndarray(self,category:str)->np.ndarray:
        """get data length with train,valid,test
        """
        if category=='train':
            return self.data_provider.x_train_indexes
        elif category=='valid':
            return self.data_provider.x_valid_indexes
        elif category=='test':
            return self.data_provider.x_test_indexes
        else:
            raise Exception("no this data category")
    def get_y_index_ndarray(self,category:str)->np.ndarray:
        """get data length with train,valid,test
        """
        if category=='train':
            return self.data_provider.y_train_indexes
        elif category=='valid':
            return self.data_provider.y_valid_indexes
        elif category=='test':
            return self.data_provider.y_test_indexes
        else:
            raise Exception("no this data category")