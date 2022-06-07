
import attr
import abc,os,glob,re
import numpy as np
from pandas import DataFrame
import enum
import yaml
import TimeSeriesDataAnalysisLib.util.time_tool as dt
from  TimeSeriesDataAnalysisLib.interface.trainable_base import AbcTrainableStore
from typing import Generator,List,Union,Tuple
VERSION='trainable_stored_v1_0'
@attr.s(auto_attribs=True)
class StoreSendMap:
    """the interface for send to store dumping generator
    properties
    --------
    name:str, the given name of this row, in usual is the file path from

    data:np.ndarray, data

    label:int, the int label
    """
    name:str=None
    data:np.ndarray=None
    label:int=None

class TrainableStoreMode(enum.Enum):
    """
    properties
    --------
    refresh: clean and new create mode

    append: append data to end mode

    """
    refresh=0
    append=1
class TrainableMemoryStore(AbcTrainableStore):
    """the interface for in memory store, the fastest, but make sure you have enough memory.
    
    """
    data_size:int=None
    store_dir=None
    _nd_list:List[np.ndarray]=None
    _label_list:List[int]=None
    _indexes:List[int]=None
    _name_list:List[str]=None
    _labeled:bool=False
    _otherInfo_list:List[dict]=None
    def __init__(self):
        pass
    def get_labels(self)->Union[List[int],None]:
        return self._label_list
    def get_otherInfos(self)->List[int]:
        return self._otherInfo_list
    def get_indexes(self)->List[int]:
        return self._indexes
    def get_names(self)->List[int]:
        return self._name_list
    def get_label_by_index(self,index:int)->np.ndarray:
        return self._label_list[index]
   
    def get_value_by_index(self,index:int)->np.ndarray:
        """get data contant of input index with expand_dims batch=1

        """
        return np.expand_dims(self._nd_list[index],axis=0)

    def dumps_to_memory(self,mode:TrainableStoreMode,name_list:List[str],nd_list:List[np.ndarray],label_list:List[int]=None,otherInfo_list:List[dict]=None):
        """read all preprocessed data row, dump all into store
        args
        -------
        mode: TrainableStoreMode, TrainableStoreMode.refresh is `clean and new create`, TrainableStoreMode.append is `append data to store tail `

        name_list: List[str], give a name list sort by index, setting file path from in usual 

        nd_list: List[np.ndarray], data content list sort by index

        kwargs
        -------
        label_list: dataset label list sort by index, the label is allowed int only. if None, the store will process as non labeled store

        otherInfo_list: store custom information

        """
        add_data_size=len(nd_list)
        if mode==TrainableStoreMode.refresh:
            if label_list!=None:
                if add_data_size != len(label_list): raise ValueError("len(label_list) not equal to len(nd_list)")
                self._labeled=True
                self._label_list=label_list
            else: 
                self._labeled=False
                self._label_list=None
            if add_data_size != len(name_list): raise ValueError("len(name_list) not equal to len(nd_list)")
            self.data_size=add_data_size
            self._nd_list=nd_list
            self._name_list=name_list
            self._indexes=range(0, len(nd_list))
            self._otherInfo_list=otherInfo_list
        # append data to end 
        elif mode==TrainableStoreMode.append:
            if label_list!=None and self._label_list!=None:
                if add_data_size != len(label_list): raise ValueError("len(label_list) not equal to len(nd_list)")
                self._labeled=True
                self._label_list+=label_list
            elif label_list!=None and self._label_list==None:
                raise ValueError("cant add labled dataset to no labeled dataset")
            else: 
                self._labeled=False
                self._label_list=None
            if add_data_size != len(name_list): raise ValueError("len(name_list) not equal to len(nd_list)")
            self.data_size+=add_data_size
            self._nd_list+=nd_list
            self._name_list+=name_list
            self._indexes=range(0, len(nd_list))
            self._otherInfo_list+=otherInfo_list

class TrainableUnconvertStore(AbcTrainableStore):
    data_size:int
    
    def get_labels(self)->Union[List[int],None]:
        pass
   
    def get_indexes(self)->List[int]:
        pass
    
    def get_label_by_index(self,index:int)->np.ndarray:
        pass
   
    def get_value_by_index(self,index:int)->np.ndarray:
        """get data contant of input index with expand_dims batch=1

        """
        pass

class TrainableStore(AbcTrainableStore):
    """the interface for access, dump trainable data ,
    save as npz format for consider both IO effectiveness and big data lazy loading.
    
    args
    --------
    store_dir: str,
        the accees target store folder

    store_name: str,
        the target store name

    store file structure
    -------- 
    meta.yaml: meta data of store, if status is not at "completed" means the store files may not complete

    map.csv: meta data of each row index, cols are [index, name, npz, label(if exist)]

    .npy: data contant


    """
    _batch_size:int
    _meta_path:str
    _map_path:str
    _maps:DataFrame
    store_dir:str
    info=None
    _npz_store={}
    def __init__(self,store_dir:str,store_name:str,skip_name=False):
        reg='[^A-Za-z0-9_\.]+'
        name_check=re.match(reg, store_name)
        if name_check!=None:
            raise ValueError("store_name must be format as [A-Za-z0-9_\.]+")
        self.store_dir=store_dir
        self._meta_path=os.path.join(self.store_dir,'meta.yaml')
        self._map_path=os.path.join(self.store_dir,'map.csv')
        self._maps=None
        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir)
        try:
            with open(self._meta_path, 'r', encoding='utf-8') as f:
                self.info=yaml.load(f, Loader=yaml.FullLoader)
                if self.info["ver"]!=VERSION:
                    raise ValueError("api version not match with meta.yaml")
                if self.info["store_name"]!=store_name and not skip_name :
                    raise ValueError("'store_name' not match with meta.yaml")
        except FileNotFoundError:
            self.info=self._init_info()
            self.info["ver"]=VERSION
            if skip_name:
                store_name=None
            self.info["store_name"]=store_name
    @property
    def data_size(self)->int:
        return self.info["data_size"]
    @property
    def batch_size(self):
        return self._batch_size
    

    def get_map(self,force:bool=False)->DataFrame:
        """get meta data of each row index, [index, name, npz, label(if exist)] from map.csv

        args
        --------
        force:bool, ignore store completed status, store files may not complete

        """
        self._check_status_instance()
        if (type(self._maps)==type(None)) or force==True:
            self._maps=pd.read_csv(self._map_path)
        return self._maps
    
    def save_labels(self,labels:List[int]):
        """ insert/override lable in map.csv

        args
        ---------
        labels: List[int], label list of indexes, label must be int
        """
        maps=self.get_map()
        size=len(labels)
        if len(list(filter(lambda x: type(x)!=int , labels )))!=0:
            ValueError("Allowed label type is int")
        if self.info["data_size"]!=size:
            raise ValueError("{0} rows not match store size{1}".format(size,self.info["data_size"]))
        if "label" in maps.columns:
            columns=maps.columns
        else:
            columns=list(maps.columns)+["label"]
        maps["label"]= pd.Series(labels, index=maps.index)
        maps.to_csv(self._map_path, mode='w',columns=columns,index=False,header=True )
        self.get_map(force=True)
    def get_labels(self)->Union[List[int],None]:
        """get labels of indexes from map.csv, a non labeled store will return None

        """
        maps=self.get_map()
        if "label" in maps.columns:
            return maps["label"].values.tolist()
        else:
            return None
    def get_indexes(self)->List[int]:
        """get indexes from map.csv

        """
        maps=self.get_map()
        return maps['index'].values.tolist()
    def get_names(self)->List[int]:
        """get names from map.csv

        """
        maps=self.get_map()
        return maps['name'].values.tolist()
    def get_label_by_index(self,index:int)->int:
        """get label of input index

        """
        maps=self.get_map()
        dfd=maps.loc[ maps['index'] == index]
        if "label" in maps.columns:
            return dfd['label'][0]
        else:
            return None
    def get_value_by_index(self,index:int)->np.ndarray:
        """get data contant of input index with expand_dims batch=1

        """
        self._check_status_instance()
        npz_file=self.get_npz_filename( self.info["batch_size"],index )
        
        if npz_file in self._npz_store.keys():
            pass
        else:
            path=os.path.join(self.store_dir,npz_file)
            self._npz_store[npz_file]=np.load(path, 'r')
        return np.expand_dims(self._npz_store[npz_file][str(index)],axis=0)

    def search_name(self,name:str)->List[np.ndarray]:
        """search name in map.csv, retrun data contant list of all found result with same name

        """
        df=self.get_map()
        dfd=df.loc[df['name'] == name]
        result=[]
        for index in dfd['index']:
            result.append(self.get_value_by_index( int(index) ))
        return result
    def _check_status_instance(self):
        if self.info["status"]!="completed":
            raise RuntimeError("This store dose not process completed last time. Clear the folder for new processing.")
    def _init_info(self):
        info={}
        info["ver"]=None
        info["store_name"]=None
        info["data_size"]=None
        info["batch_size"]=None
        info["update_date"]=None
        info["create_date"]=None
        info["status"]=None
        info["channel_num"]=None
        return info
    def dump_generator(self,mode:TrainableStoreMode,add_data_size:int,labeled:bool=True,batch_size:int=1,compress:bool=False)->Generator[int, Tuple[str,np.ndarray],None]:
        """read 1 preprocessed data row, dump 1 row into store, if the process be stoped by any reason, will mark this store status as 'processing'

        args
        -------
        mode: TrainableStoreMode, TrainableStoreMode.refresh is `clean and new create`, TrainableStoreMode.append is `append data to store tail `

        add_data_size: int, the rows num in this dataset


        kwargs
        -------
        labeled: bool, processing the store as a labeled dataset, if true, the label is allowed int only

        batch_size: int, max rows of each npz

        compress: bool, output compressed npz

        """
        # clean undump data cache
        self._kwds={} # data
        self._names={} # name
        self._labels={} # label
        start_index=0 # the start index of full batch
        start_batch_count=0 # the current processing batch count
        
        # clean and new create mode
        if mode==TrainableStoreMode.refresh:
            filelist = glob.glob(os.path.join(self.store_dir, "*"))
            for f in filelist:
                os.remove(f)
            if batch_size <=0 :
                raise ValueError("batch_size must be >= 0")
            self.info["batch_size"]=batch_size
            self.info["data_size"]=add_data_size

        # append data to end mode
        elif mode==TrainableStoreMode.append:
            self._check_status_instance()
            start_index= self.info["data_size"] #append to store end row
            # find out what npz should the start_index in
            current_npz_file=self.get_npz_filename( self.info["batch_size"],start_index )
            path=os.path.join(self.store_dir,current_npz_file)
            
            if os.path.exists( path ):
                # if the start_index should be in a exist npz, then redump this npz
                df=self.get_map()
                with np.load(path, 'r') as data:
                    # load data to undump current batch cache, for dumping at full batch. 
                    for key in data.keys():
                        self._kwds[key]=data[key]
                        self._names[key]=df.loc[int(key),'name']
                        self._labels[key]=df.loc[int(key),'label']
                os.remove(path) # going to redump this npz
                keys=[int(key) for key in data.keys()]
                start_batch_count=len(keys)
                df.iloc[:start_batch_count*-1,:].to_csv(self._map_path, mode='w',index=False,header=True )
                start_index= max(keys)+1
            else:
                # if the start_index should not be in a exist npz
                pass
            self.info["data_size"]=add_data_size+self.info["data_size"]
        else:
            raise ValueError("mode value error")
        self._batch_size=self.info["batch_size"]
        # set the meta.yaml and mark this store is in processing
        self.info["status"]="processing"
        self.info["create_date"]=dt.utc_timestamp_2_loaclzone_isoformat( dt.timestamp_utc_now())
        with open(self._meta_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.info,f, default_flow_style=False)
        return self._dump_generator(add_data_size,labeled=labeled,start_index=start_index,start_batch_count=start_batch_count,compress=compress)
    def get_npz_filename(self,batch_size:int,index:int):
        return "{0:0>6d}.npz".format( index//batch_size )
    def _dump_generator(self,add_data_size:int,labeled:bool=True,start_index:int=0,start_batch_count:int=0,compress:bool=False)->Generator[int, StoreSendMap,None]:
        if add_data_size <= 0 :
            raise ValueError("size must be >= 0")
        # the current processing index
        current_index=start_index 
        # the current processing batch count
        batch_count=start_batch_count
        # map.csv col name
        if labeled:
            fileds=['index','name','npz','label']
        else:
            fileds=['index','name','npz']
        while True:
            batch_count+=1 
            if self._batch_size >= batch_count:
                # if dose not reach full batch_size
                sendMap:StoreSendMap=(yield batch_count) # each yield will be send data as sendMap
                # cache this undump data
                self._kwds[str(current_index)]=sendMap.data 
                self._names[str(current_index)]=sendMap.name
                if labeled:
                    if type(sendMap.label) != int:
                        raise ValueError("Allowed sendMap.label type is int, or dump_generator set labeled=False")
                    self._labels[str(current_index)]=int(sendMap.label)
                # we appended a row into undump cache already
                keys=self._names.keys()
                is_at_end_index=start_index+add_data_size-1==current_index
                if len(keys)==self._batch_size or is_at_end_index:
                    # if (reach full size) or (at end index), time to dump
                    # get the file name should be
                    npz_file=self.get_npz_filename( self._batch_size,current_index )
                    path=os.path.join(self.store_dir,npz_file)
                    # dump npz, if compress the npz, less file size but loading slower 
                    if compress:
                        np.savez_compressed(path,**self._kwds)
                    else:
                        np.savez(path,**self._kwds)
                    # dump map.csv 
                    if labeled:
                        mp= [ 
                        keys,
                        self._names.values(),
                        [npz_file for i in range(self._batch_size)],
                        self._labels.values(),
                        ]
                    else:
                        mp= [ 
                        keys,
                        self._names.values(),
                        [npz_file for i in range(self._batch_size)],
                        ]
                    #print(mp)
                    kwdmap=DataFrame(  zip(*mp),columns=fileds )
                    if os.path.exists(self._map_path):
                        kwdmap.to_csv(self._map_path, mode='a',columns=fileds,index=False,header=False )
                    else:
                        kwdmap.to_csv(self._map_path, mode='w',columns=fileds,index=False,header=True )
                    # clean batch cache
                    self._kwds={}
                    self._names={}
                    self._labels={}
                    batch_count=0
                    if is_at_end_index:
                        # (at end index) we need to do set the meta.yaml
                        self.info["update_date"]=dt.utc_timestamp_2_loaclzone_isoformat( dt.timestamp_utc_now())
                        self.info["status"]="completed"
                        self.info["channel_num"]=sendMap.data.shape[-1]
                        
                        with open(self._meta_path, 'w', encoding='utf-8') as f:
                            yaml.dump(self.info,f, default_flow_style=False)
                
                current_index+=1 # as next index
            yield
           
        return
             
            
                
            
    def dumps(self,mode:TrainableStoreMode,name_list:List[str],nd_list:List[np.ndarray],label_list:List[int]=None,batch_size:int=1,compress:bool=False):
        """read all raw data row, dump all into store, if the process be stoped by any reason, will mark this store status as 'processing'

        args
        -------
        mode: TrainableStoreMode, TrainableStoreMode.refresh is `clean and new create`, TrainableStoreMode.append is `append data to store tail `

        name_list: List[str], give a name list sort by index, setting file path from in usual 

        nd_list: List[np.ndarray], data content list sort by index

        kwargs
        -------
        label_list: dataset label list sort by index, the label is allowed int only. if None, the store will process as non labeled store

        batch_size: int, max rows of each npz

        compress: bool, output compressed npz

        """
        add_data_size=len(nd_list)
        if label_list!=None:
            labeled=True
        else: 
            labeled=False
        gen=self.dump_generator(mode,add_data_size,labeled=labeled,batch_size=batch_size,compress=compress)
        for i,nd in enumerate(nd_list):
            d=next(gen)
            sendMap=StoreSendMap()
            sendMap.name=name_list[i]
            sendMap.data=nd
            if labeled==True:
                sendMap.label=label_list[i]
            gen.send( sendMap)
    
