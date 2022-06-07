import TimeSeriesDataAnalysisLib.algorithm.batch_algorithm as ba
import TimeSeriesDataAnalysisLib.algorithm.single_ts_algorithm as ts
from TimeSeriesDataAnalysisLib.algorithm.single_ts_algorithm import TimeSeriesStep
import TimeSeriesDataAnalysisLib.algorithm.channel_mapping as channel_mapping
import TimeSeriesDataAnalysisLib.util.plt_tool as plt_tool
import TimeSeriesDataAnalysisLib.util.time_tool as time_tool
from TimeSeriesDataAnalysisLib.interface.trainable_data_provider import TrainableDataProvider,AccessSubDataProviderExpand
from TimeSeriesDataAnalysisLib.interface.untrainable_base import AbcUntrainableDataProvider,AbcMajorObject,TimeSeriesStepFlag
import os,enum,re,copy
import numpy as np
from pandas import DataFrame
from typing import List,Dict,Tuple
import TimeSeriesDataAnalysisLib.util.time_tool as dt
import pandas as pd
class SimpleAnalysis:
    """some simple analysis with kmeans, ica, pca
    args
    --------
    data_provider: TrainableDataProvider,

        input data

    output_dir: str,

        output folder

    """
    def __init__ (self,data_provider:TrainableDataProvider,output_dir:str):
        self.data_provider=data_provider
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir=output_dir
    def _save_img(self,filename):
        if not os.path.exists(os.path.join(self.output_dir,'img')):
            os.makedirs(os.path.join(self.output_dir,'img'))
        plt_tool.plt.legend(loc='best')
        plt_tool.plt.title(filename)
        plt_tool.plt.savefig(os.path.join(self.output_dir,'img',"{0}.png".format(filename)))
        plt_tool.plt.clf()

    def kmeans_k_iteration(self, max_k=15):
        """scan kmeans k value, and polt result image
        kwargs
        --------
        max_k: int

            max k to iteration

        """
        x_train=self.data_provider.load_batch_x_values(self.data_provider.x_train_indexes)
        for method in [
                'silhouette',
                'elbow'
                ]:
            #print(method)
            filename="kmeans_{0}".format(method)
            ks,score=ba.kmeans_k_iteration( x_train, method ,max_k=max_k)
            #score1=np.asarray(score)
            #print(score1)
            #delta=np.diff(score1, axis=0)
            #print(delta)
            #delta2=np.diff(delta, axis=0)
            #print(delta2)
            plt_tool.sns.set()
            plt_tool.plt.plot(ks, score, label=method)
            self._save_img(filename)

    def pca_distance(self,**kwargs):
        """plot pca 2d feature map with mixing labels
        kwargs
        --------
        **kwargs: 
            matplotlib.pyplot.scatter kwargs

        """
        x_train=self.data_provider.load_batch_x_values(self.data_provider.x_train_indexes)
        filename="pca"
        distance=ba.pca_distance(x_train)
        cm = plt_tool.plt.cm.get_cmap('jet')
        plt_tool.sns.set()
        plt_tool.plt.figure(figsize=( 10,5),dpi=100)
        try:
            y_train=self.data_provider.load_batch_y_values(self.data_provider.y_train_values)
        except ValueError:
            plt_tool.plt.scatter(distance[:,0],distance[:,1],s=5,cmap=cm,**kwargs)
        else:
            y_set=list(set(y_train))
            for label in y_set:
                target_list=[i for i, y in enumerate(y_train) if y == label]
                plt_tool.plt.scatter(distance[target_list,0],distance[target_list,1],label=label,s=5,cmap=cm,**kwargs )
        self._save_img(filename)

    def ica_distance(self,**kwargs):
        """plot ica 2d feature map with mixing labels
        kwargs
        --------
        **kwargs: 
            matplotlib.pyplot.scatter kwargs

        """
        x_train=self.data_provider.load_batch_x_values(self.data_provider.x_train_indexes)
        filename="ica"
        distance=ba.ica_distance(x_train)
        cm = plt_tool.plt.cm.get_cmap('jet')
        plt_tool.sns.set()
        plt_tool.plt.figure(figsize=( 10,5),dpi=100)
        try:
            y_train=self.data_provider.load_batch_y_values(self.data_provider.y_train_values)
        except ValueError:
            plt_tool.plt.scatter(distance[:,0],distance[:,1],s=5,cmap=cm,**kwargs)
        else:
            y_set=list(set(y_train))
            for label in y_set:
                target_list=[i for i, y in enumerate(y_train) if y == label]
                plt_tool.plt.scatter(distance[target_list,0],distance[target_list,1],label=label,s=5,cmap=cm,**kwargs )
        self._save_img(filename)
    def kmeans_auto_label(self,k:int=2)->np.ndarray:
        """output re-labeled by kmeans with index-label pair csv and ica, pca image
        kwargs
        --------
        k: int

            kmeans k
        
        return
        --------
        [index,label]: np.ndarray
        """
        x_train=self.data_provider.load_batch_x_values(self.data_provider.x_train_indexes)
        
        distance_to_cluster, cluster_centers, labels  =ba.kmeans_auto_label(x_train,k)
        
        pca_distance=ba.pca_distance(x_train)
        for ik in range(k):
            target= [index for index in range( len(labels) ) if labels[index] == ik] 
            plt_tool.sns.set()
            plt_tool.plt.scatter(pca_distance[target,0],pca_distance[target,1],label=ik )
        filename="kmeans_k{0}_pca_label".format(k)
        self._save_img(filename)

        ica_distance=ba.ica_distance(x_train)
        for ik in range(k):
            target= [index for index in range( len(labels) ) if labels[index] == ik] 
            plt_tool.sns.set()
            plt_tool.plt.scatter(ica_distance[target,0],ica_distance[target,1],label=ik )
        filename="kmeans_k{0}_ica_label".format(k)
        self._save_img(filename)

        fields=['index','label']
        df=DataFrame([pd.Series(self.data_provider.x_train_indexes),pd.Series(labels)],columns=fields)
        filename="kmeans_k{0}_label".format(k)
        df.to_csv(os.path.join(self.output_dir,"{0}.csv".format(filename)), mode='w',columns=fields,index=False,header=True )
        return labels


class PlotTerm(enum.Enum):
    """Plot Decompose Time Series Data into 
    --------
    all: orginal data

    trend: data trend , all = trend + seasonal + resid

    seasonal: data seasonal

    resid: data resid

    slope: data slope
    """
    all=0
    trend=1
    seasonal=2
    resid=3
    slope=4


class TimeSeriesStepDetectionFeature:
    """The time series data with each point has several channels/features, and need to detect TimeSeriesStep
    , should be use this.  Make sure that each researchDataObject only have two hard turn points 
    of state2 start and recovery start. Or will detect wrong. 
    It's better try adjustment ts_step_process_enable_features and ts_step_process_window_size for better result.

    args
    ---------
    researchDataObject: ResearchDataObject, the TimeSeriesInjection must with ResearchDataObject instance

    required_in_ver: str, must be match ResearchDataObject.info.in_ver for check in same version

    kwargs
    ---------
    mode: TimeSeriesMode, only TimeSeriesMode.index available

    ts_step_process_enable_features: list, the features be used in detect TimeSeriesStep

    ts_step_process_window_size: int, the window_size be used in detect TimeSeriesStep

    resample_freq:str, default='1s', resmaple time space evenly, detail: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.asfreq.html

    resample_method:str, default='pad' , the method of how to resmaple time space 

    """
    _all:np.ndarray
    trend:np.ndarray=None
    seasonal:np.ndarray=None
    resid:np.ndarray=None
    _all_df:DataFrame=None
    _ts_steps_dict:dict
    _ts_point:DataFrame
    ts_step_process_enable_features:List[int]=None
    ts_step_process_window_size:int=None
    def __init__(self,
        provider:AbcUntrainableDataProvider,
        **kwargs):
        self.research_data_provider=provider
        
        self._all=provider.get_all()
        self.datetime_list=provider.get_datetime_list()
        self.feature_index=provider.get_feature_name(only_index=True)
        self.feature_name=provider.get_feature_name(only_index=False)
        if provider.is_time_series_step == False :
            raise Exception("This data is not a TimeSeriesStep data")
        #self._ts_steps_dict=self._get_init_ts_point_dict()
        self._columns=[value.value for value in TimeSeriesStep.__members__.values()]
        self.all_df=pd.DataFrame(self._all)
        

    def define_ts_step(self,**kwargs):
        """
        ts_step_process_enable_features: list, the features be used in detect TimeSeriesStep

        ts_step_process_window_size: int, the window_size be used in detect TimeSeriesStep

        """
        self.ts_step_process_enable_features=kwargs["ts_step_process_enable_features"]
        self.ts_step_process_window_size=kwargs["ts_step_process_window_size"]
        self.datetime_resample()
        self.do_seasonal_decompose()
        self.do_ts_step_process()

    def get_index_by_datetime(self,nt:dt.datetime):
        #target=dt.any_isoformat_2_utc_datetime( "2019-07-10T03:07:49.600Z")
        return self.all_df.index.get_loc(pd.to_datetime(nt), method='nearest')
    def get_all(self)->np.ndarray:
        return self.all_df.values
    def datetime_resample(self,freq='1s',method='pad'):
        """
        freq:str, default='1s', resmaple time space evenly, detail: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.asfreq.html

        method:str, default='pad' , the method of how to resmaple time space 

        """
        """override this method for define how to resample time space"""
        
        self.all_df['Datetime'] = pd.to_datetime(self.datetime_list)
        self.all_df = self.all_df.set_index('Datetime')
        self.all_df=self.all_df.asfreq(freq=freq, method=method)
        
    def do_ts_step_process(self)->None:
        """acting ts_step_process"""
        nd=self.trend if self.trend is not None else self.all_df.values
        ts_point=ts.get_ts_point(nd,enable_features=self.ts_step_process_enable_features,window_size=self.ts_step_process_window_size)
        shift=self.ts_step_process_window_size
        #check shifted window with orginal index
        row_num=int(ts_point.max().max()+1+shift)
        col_num=nd.shape[1]
        if nd.shape[0]+self.ts_step_process_window_size!=row_num:
            raise ValueError(' (trend rows + ts_step_process_window_size)  must equel to (ts_point rows + shift) ')
        #  shift point to orginal index
        #print(ts_point)
        #ts_point=ts.shift_ts_point(ts_point,shift)
        self._ts_point=ts_point
   
    def do_seasonal_decompose(self)->None:
        #cached seasonal_decompose
        #print(self._all[0])
        seasonal_decompose=ts.get_seasonal_decompose( self.all_df.values,window_size=self.ts_step_process_window_size)
        self.trend=seasonal_decompose["trend"]
        self.seasonal=seasonal_decompose["seasonal"]
        self.resid=seasonal_decompose["resid"]
        # self.trend=np.zeros(( row_num ,col_num))
        # self.trend[window_size:,:]=seasonal_decompose["trend"].values[:,:]
        # self.trend=DataFrame( self.trend,columns=seasonal_decompose["trend"].columns)
        # self.seasonal=np.zeros(( row_num ,col_num))
        # self.seasonal[window_size:,:]=seasonal_decompose["seasonal"].values[:,:]
        # self.seasonal=DataFrame( self.seasonal,columns=seasonal_decompose["seasonal"].columns)
        # self.resid=np.zeros(( row_num ,col_num))
        # self.resid[window_size:,:]=seasonal_decompose["resid"].values[:,:]
        # self.resid=DataFrame( self.resid,columns=seasonal_decompose["resid"].columns)
    def get_step_shift_ts_point(self,step:TimeSeriesStep)->DataFrame:
        """set the TimeSeriesStep start index as 0"""
        shift=self._ts_point[step.value][0]*-1
        return ts.shift_ts_point(self._ts_point,shift)
    def window_shift_ts_point(self,window_size:int=10)->DataFrame:
        """shift window_size of the TimeSeriesStep start index"""
        shift=window_size
        return ts.shift_ts_point(self._ts_point,shift)
    @property
    def ts_point(self)->DataFrame:

        return self._ts_point
    # def get_ts_point_as_dict(self)->dict:
    #     return self._ts_steps_dict
    # def ts_point_df_to_dict(self,ts_point:DataFrame)->dict:
    #     ts_steps_dict=self._get_init_ts_point_dict()
    #     for col in self._columns:
    #         if ts_point[col][0]!=None:
    #             #print(ts_point[col])
    #             ts_steps_dict[col]["start_index"]=ts_point[col][0]
    #             ts_steps_dict[col]["end_index"]=ts_point[col][1]
    #     return ts_steps_dict
    # def _get_init_ts_point_dict(self)->dict:
    #     ts_steps_dict={}
    #     for value in TimeSeriesStep.__members__.values():
    #         ts_steps_dict[value.value]={
    #             "start_index":None,
    #             "end_index":None
    #             }
    #     return ts_steps_dict
    def get_step_dataslice(self,steps:List[TimeSeriesStep],decompose:str,shift_size:int=0)->np.ndarray:
        """slice data into the needed TimeSeriesStep and decompose

        args
        ----------
        steps: List[TimeSeriesStep], retrun a combined TimeSeriesStep data into one np.ndarray

        decompose:str , "all": orginal data, "trend": data trend , "seasonal": data seasonal, "resid": data resid; all = trend + seasonal + resid

        kwargs
        ----------
        shift_size: int, shift the index
        
        """
        
        if decompose=="all":
            data=self.all_df.values
        elif decompose=="trend":
            data=self.trend
        elif decompose=="seasonal":
            data=self.seasonal
        elif decompose=="resid":
            data=self.resid
        else:
            raise ValueError("no this decompose")
        term_nums=list(filter(lambda x: not pd.isna(x), [self._ts_point[key][1]  for key in self._ts_point.columns]))
        terms=int(max(term_nums))+1
        if terms != data.shape[0]:
            shift_size=data.shape[0]-terms
        result=[]
        # combine TimeSeriesStep data
        last_end_index=-1
        for step in steps:
            start_index=int(self._ts_point[step.value][0])+shift_size
            end_index=int(self._ts_point[step.value][1])+shift_size
            if start_index == last_end_index:
                start_index+=1
            if end_index > start_index:
                result.append(data[start_index : end_index, :].copy())
            last_end_index=end_index
        if len(result) ==0:
            for step in steps:
                if not pd.isna(self._ts_point[step.value][0]):
                    start_index=int(self._ts_point[step.value][0])+shift_size
                    end_index=int(self._ts_point[step.value][0]+1)+shift_size
                    result.append(data[start_index : end_index, :].copy())
        return np.vstack(result)
    
    def check_isnan(self,steps: List[TimeSeriesStep],enable_features:List[int],decompose:str='all')->np.ndarray:
        nd=self.get_step_dataslice(steps,decompose)
        #check=np.argwhere(np.isnan(nd[:,enable_features]))  #get index
        check1=np.isnan(nd[:,enable_features]).any()  #has anyone
        check2=(nd[:,enable_features] == -32767 ).any()
        if check1 or check2 :
            return True
        else:
            return False
    def slope(self,steps:List[TimeSeriesStep],enable_features:List[int],window_size:int=10,decompose:str='all')->np.ndarray:
        """calculate slope by np.polyfit
        """
        if len(enable_features)==0:
            result=np.asarray([[]]).reshape(1,-1)
            return result
        array=self.get_step_dataslice(steps,decompose=decompose)[:,enable_features]
        #print(array.shape)
        return ts.slope( array ,0,window_size=window_size)
    
    def mean_0d(self,enable_features:List[int],time_series_step_list:List[ts.TimeSeriesStep],decompose:str='all')->np.ndarray:
        """calculate step_list mean on axis 0
        """
        if len(enable_features)==0:
            result=np.asarray([[]]).reshape(1,-1)
            return result
        nd=self.get_step_dataslice(time_series_step_list,decompose)
        result=np.asarray(np.mean(nd[:,enable_features],axis=0).reshape(1,-1))
        #print('state1_mean ',result)
        return result
    def state1_mean_0d(self,enable_features:List[int],decompose:str='all')->np.ndarray:
        """calculate state1 mean on axis 0
        """
        return self.mean_0d(enable_features,[ts.TimeSeriesStep.State1],decompose=decompose)
    
    def state3_mean_0d(self,enable_features:List[int],decompose:str='all')->np.ndarray:
        """calculate state3 mean on axis 0
        """
        return self.mean_0d(enable_features,[ts.TimeSeriesStep.State3],decompose=decompose)
    def state2_mean_0d(self,enable_features:List[int],decompose:str='all')->np.ndarray:
        """calculate state2 mean on axis 0
        """
        return self.mean_0d(enable_features,[ts.TimeSeriesStep.State2],decompose=decompose)
    def state3_delta_state1_over_state2_0d(self,enable_features,decompose:str='all')->np.ndarray:
        """calculate (state3-state1)/state2 on axis 0
        """
        if len(enable_features)==0:
            result=np.asarray([[]]).reshape(1,-1)
            return result
        def limit(a):
            if a<0:
                return min(a,-0.000001)
            else:
                return max(a,0.000001)
        b=self.state1_mean_0d(enable_features,decompose=decompose)
        c=np.vectorize(limit)(self.state2_mean_0d(enable_features,decompose=decompose))
        s=self.state3_mean_0d(enable_features,decompose=decompose)
        #print('delta_rate ',result)
        result=np.asarray((s-b)/c)

        return result
  
    def get_step_flag(self)->TimeSeriesStepFlag:
        """get step_flag as TimeSeriesStepFlag

        """
        flag=TimeSeriesStepFlag()
        stepflags=list(flag.__dict__.keys())
        for i,value in enumerate(self.ts_point.values.reshape(-1,order='F')):
            flag.__dict__[stepflags[i]]=value
        
        return flag
    def dumps_json(self)->str:
        """create new deepcopy data_object with update step_flag of annotation,

        args
        --------

        return
        --------
        new data_object string from deepcopy data_object 

        """
        data_object=self.data_provider.update_step_flag(self.get_step_flag())
        # with open(os.path.join(self.output_dir,"{0}.json".format(name)),'w') as f:
        #     f.write(data_object.dumps() )
        return data_object.dumps()

    def plot_package(self,name:str,output_dir:str,steps:List[TimeSeriesStep],term_list:List[PlotTerm],enable_features:List[int],window_size:int=10,nested_dir:bool=False):
        """plot this TimeSeriesStep list into a image

        args
        ----------
        name: str, image title, file name

        output_dir: str, image output folder

        steps: List[TimeSeriesStep], retrun a combined TimeSeriesStep data into one np.ndarray

        term_list: List[PlotTerm], the PlotTerm list to plot on separate

        enable_features: List[int], the features be show on image

        kwargs
        ----------
        window_size: int, only available when PlotTerm.slope

        nested_dir: bool, if you want dump each ResearchDataObject dump in their folder , set nested_dir=True
        
        """
        self.get_step_flag()
        reg='[^A-Za-z0-9_\.]+'
        name=re.sub(reg, '_', name)
        enable_features_str=[str(e) for e in enable_features]
        # get step point as DataFrame
        ts_point=self.ts_point
        array_list=[]
        title_list=[]
        ts_list=[]
        if nested_dir:
            output_dir=os.path.join(output_dir,name)
        if not os.path.exists( output_dir):
            os.makedirs(output_dir)
        if PlotTerm.all in term_list:
            # get data slice of the step
            ar=self.get_step_dataslice(steps,decompose="all")
            array_list.append(DataFrame(ar,columns=self.feature_index))
            title_list.append(name+'_all')
            ts_list.append(ts_point)
        if PlotTerm.trend in term_list:
            # get data slice of the step
            ar=self.get_step_dataslice(steps,decompose="trend")
            array_list.append(DataFrame(ar,columns=self.feature_index))
            title_list.append(name+'_trend')
            ts_list.append(ts_point)
        if PlotTerm.seasonal in term_list:
            # get data slice of the step
            ar=self.get_step_dataslice(steps,decompose="seasonal")
            array_list.append(DataFrame(ar,columns=self.feature_index))
            title_list.append(name+'_seasonal')
            ts_list.append(ts_point)
        if PlotTerm.resid in term_list:
            # get data slice of the step
            ar=self.get_step_dataslice(steps,decompose="resid")
            array_list.append(DataFrame(ar,columns=self.feature_index))
            title_list.append(name+'_resid')
            ts_list.append(ts_point)
        if PlotTerm.slope in term_list:
            ar=self.slope(steps,enable_features,window_size=window_size,decompose='trend')
            array_list.append(DataFrame(ar,columns=enable_features_str))
            title_list.append(name+'_slope')
            ts_list.append(ts_point)
        figs=plt_tool.simple_batch_plt_ts_point_imgs(
            array_list,
            enable_feature_names=enable_features_str,
            ts_point_list=ts_list,
            title_list=title_list
        )
        plt_tool.save_imgs(figs,output_dir,[x+'.png' for x in title_list ])

class TimeSeriesStepAnnotatedFeature(TimeSeriesStepDetectionFeature):
    """The time series data with each point has several channels/features, and already define TimeSeriesStep by control event, should be use this. 
    args
    ---------
    researchDataObject: ResearchDataObject, the TimeSeriesInjection must with ResearchDataObject instance

    required_in_ver: str, must be match ResearchDataObject.info.in_ver for check in same version

    kwargs
    ---------
    mode: TimeSeriesMode, only TimeSeriesMode.index available

    """
    def define_ts_step(self,**kwargs):
        self.datetime_resample()
        self.do_ts_step_process()
    def do_ts_step_process(self)->None:
        raw_ts_point=self.research_data_provider.get_ts_point()
        #單NA補值
        for col in raw_ts_point.columns:
            s=pd.isna(raw_ts_point[col][0])
            e=pd.isna(raw_ts_point[col][1])
            if s and e:
                continue
            if s:
                raw_ts_point[col][0]=raw_ts_point[col][1]
                continue
            if e:
                raw_ts_point[col][1]=raw_ts_point[col][0]
                continue

        ts_point=raw_ts_point.applymap( lambda x: pd.NA if pd.isna(x) else self.get_index_by_datetime( self.datetime_list[x]) )
        print(ts_point)
        self._ts_point=ts_point

    def get_my_feature_0d(self,window_size,enable_features,enable_featuresO)->np.ndarray:
        if len(enable_features)==0:
            result=np.asarray([[]]).reshape(1,-1)
            return result
        decompose='all'
        check=self.check_isnan([ts.TimeSeriesStep.State1,ts.TimeSeriesStep.State2,
        ts.TimeSeriesStep.State3],enable_features,decompose=decompose)
        if check:
            raise Exception("Those indexes has nan or -32767")
        #selected features
        feature_0d=self.state3_delta_state1_over_state2_0d(enable_features,window_size=window_size)
        state1_meanO=self.state1_mean_0d(enable_featuresO,decompose=decompose)
        state3_meanO=self.state3_mean_0d(enable_featuresO,decompose=decompose)
        result =np.hstack([feature_0d,state1_meanO,state3_meanO])
        return result
    
class DataSliceAnalysis:
    """Analysis the turn point to slice

    args
    ----------
    provider: AbcUntrainableDataProvider, must be the extends of AbcUntrainableDataProvider

    output_dir: str, the output folder

        
    """
    def __init__(self,provider:AbcUntrainableDataProvider,output_dir:str):
        self.output_dir=output_dir
        self.provider=provider
        self._all=provider.get_all()


    def get_change_point(self,
    nda: np.ndarray,
    detect_change_point_num: int, 
    window_size=10, 
    effective_col_indexes:List[int]=None, 
    err_thresh=1.1) -> np.ndarray:
        """get change/turn point on axis 0, this method incloud smooth py seasonal_decompose

    args
    ---------
    nda: np.ndarray

        input ndarray

    detect_change_point_num: int

        how may change point will be detected 


    kwargs
    ---------
    window_size: int

        scan window size

    effective_col_indexes:List[int]

        select key effective of col indexes, if None will calculate all cols

    err_thresh: float

        the allowed tolerance, if (error >= err_thresh): return median, else: return mode

    return 
    ---------
    the index of detected points, offset by window_size: np.asarray, 
    if input shape=(m,n)
    return shape=(1,n)
    """

        return ts.get_change_point(
                ts.get_seasonal_decompose(nda,window_size=window_size)['trend'],
                detect_change_point_num,
                window_size=window_size,
                effective_col_indexes=effective_col_indexes,
                err_thresh=err_thresh)+(window_size)

    def plot_slice_line(self,
        name:str , 
        detect_change_point_num: int, 
        window_size:int=10,
        effective_col_indexes:List[int]=None,
        err_thresh=1.1,):

        """plot a img to see detect result, and estimate how to slice
        the following parameters may need to adjustment case by case
        
        args
        --------
        name: str

        detect_change_point_num: int, the change points you need

        kwargs
        --------
        
        window_size: int, the smooth window

        effective_col_indexes: List[int], the col indexes of only being considered 

        select key effective of col indexes, if None will calculate all cols

        err_thresh: float, the allowed tolerance, if (error >= err_thresh): return median, else: return mode

        """
        effective_col_indexes_str=[str(e) for e in effective_col_indexes]

        change_point=self.get_change_point(
                self._all,
                detect_change_point_num,
                window_size=window_size,
                effective_col_indexes=effective_col_indexes,
                err_thresh=err_thresh)
        self.plot_img(name,change_point,effective_col_indexes=effective_col_indexes)
    def plot_img(self,
        name:str,
        change_point:np.ndarray,
        effective_col_indexes:List[int]=None,
        partitive:bool=False,
        ):
        """plot a img with change_point lines, 
        
        args
        --------
        name: str

        change_point: np.ndarray, the change/turn point, shape=(2)


        kwargs
        --------
        effective_col_indexes: List[int], the col indexes of only being considered 

        partitive: bool, True is output the partitive image after slice

        """
        effective_col_indexes_str=[str(e) for e in effective_col_indexes]
        if partitive:
            array=DataFrame(self._all[change_point[0]:change_point[1],effective_col_indexes],columns=effective_col_indexes_str)
            change_point=change_point-change_point[0]
        else:
            array=DataFrame(self._all[:,effective_col_indexes],columns=effective_col_indexes_str)

        figs=plt_tool.simple_batch_plt_change_point_imgs(
            [array],
            enable_feature_names=effective_col_indexes_str,
            change_point_list=[change_point],
            title_list=[name]
        )
        plt_tool.save_imgs(figs,self.output_dir,[name+'.png'])
    def get_slice_point(self,
        detect_change_point_num:int,
        start_slice_line_index:int=0,
        end_slice_line_index:int=None,
        slice_strides_line_num:int=0,
        slice_head_margin:int=200,
        slice_tail_margin:int=0,
        window_size:int=10,
        head_index:int=0,
        end_index:int=None,
        effective_col_indexes=None,
        err_thresh=1.1,
        )->np.ndarray:
        """get slice point with margin, 
        
        args
        --------

        detect_change_point_num: int, the change points you need, use plot_slice_line to determine value

        kwargs
        --------

        start_slice_line_index: int, the first slice line of change_point_num index

        end_slice_line_index: int, the last slice line of change_point_num index, set None means as end line

        slice_strides_line_num: int, the strides of slice line walk, more then 1 means skip regular lines.

        *line_pick= change_point[start_slice_line_index:end_slice_line_index:slice_strides_line_num]

        slice_head_margin: int, the  head  margin will add to each piece

        slice_tail_margin: int, the  tail margin will add to each piece

        head_index: int, the data start at

        end_index: int, the data end at, set None means as end data

        *data_pick= data[head_index:end_index]
        
        window_size: int, the smooth window

        effective_col_indexes: List[int], the col indexes of only being considered 

        select key effective of col indexes, if None will calculate all cols

        err_thresh: float, the allowed tolerance, if (error >= err_thresh): return median, else: return mode

        """
        change_point=self.get_change_point(
                self._all,
                detect_change_point_num,
                window_size=window_size,
                effective_col_indexes=effective_col_indexes,
                err_thresh=err_thresh)
        if end_index==None:
            end_index=int(self._all.shape[0]-1)
        slice_point=ts.regular_strides_slice(
            change_point,
            end_index,
            head_index=head_index,
            start_slice_line_index=start_slice_line_index,
            end_slice_line_index=end_slice_line_index,
            slice_strides_line_num=slice_strides_line_num,
            slice_head_margin=slice_head_margin,
            slice_tail_margin=slice_tail_margin)
            
        return slice_point
    def dumps_json(self,slice_point:np.ndarray)->str:
        """create new data_object from slice deepcopy data_object of data_provider,

        args
        --------

        slice_point: np.ndarray, the slice point

        return
        --------
        new data_object string from slice deepcopy data_object of data_provider

        """
        data_object=self.data_provider.slice_data_size(slice_point)
        # with open(os.path.join(self.output_dir,"{0}.json".format(name)),'w') as f:
        #     f.write(data_object.dumps() )
        return data_object.dumps()



