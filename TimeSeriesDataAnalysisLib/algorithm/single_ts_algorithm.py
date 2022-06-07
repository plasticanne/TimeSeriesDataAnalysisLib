import numpy as np
from pandas import DataFrame
import enum
from typing import List,Dict,Tuple


class TimeSeriesStep(enum.Enum):
    """Enum class, The time series data need to detect several step as follows. 
    for each round or cycle , then this is a TimeSeriesStep data
    
    properties
    --------
    State0: define phase as state 0

    State1: define phase as state 1

    State2: define phase as state 2

    State3: define phase as state 3

    State4: define phase as state 4

    State5: define phase as state 5
    """
    State0="State0"
    State1="State1"
    State2="State2"
    State3="State3"
    State4="State4"
    State5="State5"

def select_columns(nda:np.ndarray,columns:List[int])->np.ndarray:
    return nda[:,columns]
def select_rows_columns(nda:np.ndarray,rows:List[int],columns:List[int])->np.ndarray:
    return nda[[[x] for x in rows],columns]
def slope(nda:np.ndarray,mode:int=0,window_size:int=10)->np.ndarray:
    """calculator slope on axis 0 

    args
    ---------
    nda: np.ndarray
        input ndarray

    kwargs
    ---------
    mode: int

        0: np.polyfit

        1: np.convolve, (not available)

        2: np.diff, (not available)

    window_size: int

        watch window size, ndarray.shape[0] will reduce window_size
    """
    arrAll=nda
    channel=arrAll.shape[1]
    row_num=arrAll.shape[0]
    if mode==0:
        result = np.zeros((row_num,channel))
        for feature in range(channel):
            for row in range(row_num-window_size):
                result[row + window_size, feature] = np.polyfit(
                        np.arange(window_size),
                        arrAll[row:row+window_size, feature], 1)[0]
        """
    elif mode==1:  
        #index  當前-上一個
        window = [-1,0,0,0,0,0,0,0,0, 0, 1] # [-1,0,0 0, 1]
        #print(np.shape(arrAll[:,0]))
        for i in range(0,channel):
            arr=np.convolve(arrAll[:,i], window, mode='valid')[:,np.newaxis]
            if i==0:
                pass
            elif i==1:
                delta=arr
            else:     
                delta=np.hstack( (delta,arr)  )
        x_sample=delta.shape[0]
        #print(x_sample)
        #index以外 cols
        deltaY=delta[:,1:]
        
        #index 複製成8 cols
        deltaX=delta[:,0:1].repeat(channel-2,axis=1)
        #各自元素取斜率
        result= deltaY/deltaX
        result= np.concatenate( (delta[:,0:1], result), axis=1)
    elif mode==2:
        #差分
        delta=np.diff(arrAll[:,1:], axis=0)
        #rows會少一個
        x_sample=delta.shape[0]
        #index以外 cols
        deltaY=delta[:,1:]
        #index 複製成8 cols
        deltaX=delta[:,0:1].repeat(channel-2,axis=1)
        result= deltaY/deltaX
        result= np.concatenate( (delta[:,0:1], result), axis=1)
        
    elif mode==3:
        
        windowSize=1
        result = np.zeros((row_num-windowSize,channel-2))
        for feature in range(2,channel):
            result[:,feature-2] =tools.diff(arrAll[:,feature],k_diff= 1)
        print(result.shape,arrAll[0:result.shape[0],1:2].shape)
        result= np.concatenate( (arrAll[0:result.shape[0],1:2], result), axis=1)
        """
    else:
        raise ValueError("")

    return result


def get_seasonal_decompose(nda:np.ndarray,window_size:int=10)->Dict[str,np.ndarray]:
    """ Decompose Time Series Data into Trend and Seasonality on axis 0

    args
    ---------
    nda: np.ndarray
        input ndarray

    kwargs
    ---------
    window_size: int

        watch window size, ndarray.shape[0] will reduce window_size

    return 
    ---------
    Dict{
        "trend":np.asarray,
        "seasonal":np.asarray,
        "resid": np.asarray,
    }
    """
    from statsmodels.tsa.seasonal import seasonal_decompose,DecomposeResult
    from statsmodels.tsa.statespace import tools
    series:DecomposeResult = seasonal_decompose( nda , model='additive',period=window_size,two_sided=False)
    result={
        "trend":series.trend[window_size:,:],
        "seasonal":series.seasonal[window_size:,:],
        "resid": series.resid[window_size:,:],
    }
    #print( result["trend"].shape)
    return result  

# def get_scores(nda:np.ndarray)->np.ndarray:
#     #pip install git+https://github.com/shunsukeaihara/changefinder.git
#     import changefinder
#     result=[]
#     for i in range(nda.shape[1]):
#         points=nda[:,i]
#         cf = changefinder.ChangeFinder()
       
#         scores = [cf.update(p)[0] for p in points]
#         scores=np.asarray(scores)
        
#         result.append(  scores  )
#     return np.asarray(result).T

def get_median(nda:np.ndarray,err_base:int,err_thresh:float)->np.ndarray:
    """get median with tolerance on axis 1, this will campare median and mode

    error = abs(mode-median) / err_base

    if (error >= err_thresh): return median

    else: return mode

    args
    ---------
    nda: np.ndarray

        input ndarray

    err_base: int

        average the diff of mode and median, in common can use window_size as err_base`

    err_thresh: float

        the allowed tolerance, if (error >= err_thresh): return median, else: return mode

    return 
    ---------
    np.asarray, 
    if input shape=(m,n)
    return shape=(1,n)
    """
    
    median=np.median(nda,axis=1)
    #print('median',median)
    mode = np.floor( np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 1, nda))
    #print('mode',mode)
    error=abs(mode-median)/err_base
    result =[]
    for i,err in enumerate(error):
        if err>=err_thresh:
            result.append( int(median[i]) )
        else:
            result.append(  int(mode[i]) )
        #print(result)
    result = np.asarray(result)
    return result

def get_change_point(nda:np.ndarray,n_bkps:int,window_size:int=10,effective_col_indexes:List[int]=None,err_thresh:float=1.1)->np.ndarray:
    import ruptures
    """get change/turn point on axis 0, 

    args
    ---------
    nda: np.ndarray

        input ndarray

    n_bkps: int

        how may point will be detected 


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
    model = "l2"  
    #model="rbf"
    cp=[]
    for feature in effective_col_indexes:
        points=nda[:,[feature]]
        #algo = ruptures.Pelt(model=model).fit(points)
        #algo = ruptures.Binseg(model=model).fit(points)
        algo = ruptures.Window(width=window_size,model=model).fit(points)
        series=algo.predict(
            n_bkps=n_bkps,
            #pen=0.05
            )
        #result.append(list(map(lambda x: x-window_size, series)))
        cp.append(series)
    cp=np.asarray(list(zip(*cp)))[:n_bkps]
    #the index of detected points, offset by window_size
    cp=get_median(cp,window_size,err_thresh)-(window_size)//2
    return cp
def get_ts_state2_range_point(nda:np.ndarray,window_size:int=10,enable_features:List[int]=None)->np.ndarray:
    """get state2 start and recovery start point on axis 0 , 

    args
    ---------
    nda: np.ndarray

        input ndarray

    kwargs
    ---------
    window_size: int

        scan window size

    enable_features:List[int]

        select col indexes, if None will calculate all cols

    return 
    ---------
    [state2 start,recovery start]: np.asarray, 
    return shape=(1)
    """
    change_point=get_change_point(nda,2,window_size=window_size,effective_col_indexes=enable_features)
    try:
        # check if points at almost same position
        if change_point[0]<=window_size or change_point[1]>= nda.shape[0]-window_size:
            # if true, get more 1 point 
            change_point=get_change_point(nda,3,window_size=window_size,effective_col_indexes=enable_features)
            # selet reasonable point set
            if change_point[2]>= nda.shape[0]-window_size:
                change_point=change_point[0:2]
            else:
                change_point=change_point[1:3]
    except Exception:
        pass
    # check if point small than 0 after window_size shift
    if change_point[0]<0:
        change_point[0]=window_size
    
    print("change point: {0}".format(change_point))
    return change_point
def _find_left_b_r_point2(start_row,slope_array,intial_slope_thresh):

    r_point=np.zeros(start_row.shape,dtype=int)
    b_point=np.ones(start_row.shape,dtype=int)
    for j,col in enumerate(slope_array.T):
        found_r_point=False
        found_b_point=False
        for i,row in reversed(list(enumerate(col))):
            if  row<=intial_slope_thresh and found_b_point==False and found_r_point==False:
                r_point[j]=i
                #print('a',row,i)
                found_r_point=True
            if  row > intial_slope_thresh and found_b_point==False and found_r_point==True:
                b_point[j]=i-1
                #print('b',row,i)
                found_b_point=True
    print('candidate state2 point: ', r_point)
    print('candidate state1 point: ',b_point)
    r_point=np.min(r_point)#np.argmax(np.bincount(r_point))
    b_point=np.max(b_point)#np.argmax(np.bincount(b_point))
    return b_point,r_point
def _find_left_b_point(channel_shape,slope_array,intial_slope_thresh):
    """find state1 start point, this should be on the left side of state2 point
    
    """
    b_point=np.ones(channel_shape,dtype=int)
    # every col has their own scan
    for j,col in enumerate(slope_array.T):
        found_b_point=False
        # scan all left points, if value > intial_slope_thresh then stop
        for i,row in enumerate(reversed(list(col))):
            if  row > intial_slope_thresh[j] and found_b_point==False:
                b_point[j]=i-1
                #print('b',row,i)
                found_b_point=True
    # considered all cols, select the most right one
    result=np.max(b_point)#np.argmax(np.bincount(b_point))
    return b_point,result
def _find_left_r_point(channel_shape,slope_array,intial_slope_thresh):
    """find state2 start point, this should be on the left side of original state2 point,
    the original state2 point may a little too right 
    """
    # right is change_point[0]-1 
    # left is change_point[0]-2
    #r_point=slope_array.shape[0]-2
    r_point=np.zeros(channel_shape,dtype=int)
    # every col has their own scan
    for j,col in enumerate(slope_array.T):
        found_r_point=False
        # scan all left points, if value <= intial_slope_thresh then stop
        for i,row in enumerate(reversed(list(col))):
            if  row<=intial_slope_thresh[j] and found_r_point==False:
                r_point[j]=i
                found_r_point=True
    # considered all cols, select the most right one            
    result=np.argmax(np.bincount(r_point))
    return r_point,result
def _find_left_s_point(channel_shape,slope_array,intial_slope_thresh,window_size:int=10):
    """find state3 start point, this should be on the left side of recovery point
    
    """
    first_state3_point=np.zeros(channel_shape,dtype=int)
    # every col has their own scan
    for j,col in enumerate(slope_array.T):
        found_point=False
        # scan all left points, if value > intial_slope_thresh then stop
        for i,row in enumerate(reversed(list(col))):          
            if  row > intial_slope_thresh[j] and found_point==False:
                first_state3_point[j]=i-1
                found_point=True
    #print('candidate state3 point from change_point[0]: ', first_state3_point)
    # considered all cols, select the most right one   
    result=np.max(first_state3_point)#np.argmax(np.bincount( first_state3_point))
    return first_state3_point,result
def get_ts_point(
    nda:np.ndarray,
    window_size:int=10,
    enable_features=None,
    intial_slope_thresh_scale:float=0.01,
    scan_slope_thresh_iter_add_scale:float=0.001,
    max_scan_iter:int=500,
    )->DataFrame:
    """get TimeSeriesStep points on axis 0 , 

    args
    ---------
    nda: np.ndarray

        input ndarray

    kwargs
    ---------
    window_size: int, 

        scan window size

    enable_features:List[int], 

        select col indexes, if None will calculate all cols

    intial_slope_thresh_scale: float, 

        the slope value thresh to define it is on state3,   
        >>> intial_slope_thresh= (max_slope-min_slope)*intial_slope_thresh_scale

    scan_slope_thresh_iter_add_scale: float,  

        if did not find critical point of intial_slope_thresh , 
        >>> scan_slope_thresh_iter_add= (max_slope-min_slope)*scan_slope_thresh_iter_add_scale
        will acting loop scan for 
        >>> intial_slope_thresh+=scan_slope_thresh_iter_add

    max_scan_iter: int,

        the max iteration number, if no critical point, the result is initial point

    return 
    ---------
    DataFrame

    """

    # expect to get [state2 start with a little right side shift] and [recovery start]
    change_point=get_ts_state2_range_point(nda,window_size=window_size,enable_features=enable_features)
    
    # state2+state3 length
    change_size=change_point[1]-change_point[0]+1

    # find state3 start
    # targeted state3 start range between change_point[0] and change_point[1]
    array_act=nda[:,enable_features][change_point[0]:change_point[1],:]
    slope_act=abs(slope( array_act ,0,window_size=window_size))
    max_slope=np.argmax(slope_act,axis=0)
    min_slope=np.argmax(slope_act,axis=0)
    channel_shape=max_slope.shape
    intial_slope_thresh=(max_slope-min_slope)*intial_slope_thresh_scale
    scan_slope_thresh_iter_add=(max_slope-min_slope)*scan_slope_thresh_iter_add_scale
    max_slope_index=np.min(max_slope)
    state3_allowed_left=max_slope_index+change_point[0]
    scan_slope=slope_act[max_slope_index:,:]

    intial_slope_thresh_s=intial_slope_thresh
    # find point by reach slope thresh on critical point
    print("--find state3 point--")
    all_point,state3_point=_find_left_s_point(channel_shape,scan_slope,intial_slope_thresh_s,window_size=window_size)
    state3_point=state3_point+state3_allowed_left
    print("initail intial_slope_thresh is {0}".format(intial_slope_thresh_s))
    print("first result, all feature point: {0}, selected point: {1}".format(all_point,state3_point))
    
    # if no point, adjustment thresh till max_scan_iter
    if state3_point<=state3_allowed_left:
        print("optimization state3 point")
        for i in range(max_scan_iter):
            intial_slope_thresh_s+=scan_slope_thresh_iter_add
            all_point,state3_point=_find_left_s_point(channel_shape,scan_slope,intial_slope_thresh_s,window_size=window_size)
            state3_point=state3_point+state3_allowed_left
            if state3_point > change_point[1]:
                break
        print("state3_point intial_slope_thresh is {0}".format(intial_slope_thresh_s))
    if state3_point<=state3_allowed_left or state3_point>change_point[1]-2:
        state3_point=change_point[1]-2
    print("final result, all feature point: {0}, selected point: {1}".format(all_point,state3_point))

    

    # find state2 start
    # target [state2 start] and [base start] range between 0 and change_point[0]
    array_ini=nda[:,enable_features][:change_point[0],:] 
    slope_ini= abs(slope( array_ini ,0,window_size=window_size))
    scan_slope=slope_ini
    addwide=window_size
    intial_slope_thresh_r=intial_slope_thresh
    # find point by reach slope thresh on critical point
    print("--find state2 point--")
    all_point,state2_point=_find_left_r_point(channel_shape,scan_slope,intial_slope_thresh_r)
    print("initail intial_slope_thresh is {0}".format(intial_slope_thresh_r))
    print("first result, all feature point: {0}, selected point: {1}".format(all_point,state2_point))
    # if move distance too large, adjustment thresh till max_scan_iter
    if abs(change_point[0]-state2_point) >= window_size:
        print("optimization state2 point")
        for i in range(max_scan_iter):
            intial_slope_thresh_r+=scan_slope_thresh_iter_add
            all_point,state2_point=_find_left_r_point(channel_shape,scan_slope,intial_slope_thresh_r)
            if abs(change_point[0]-state2_point) < window_size:
                break
        #print("state2_point intial_slope_thresh/2")
        print("state2_point intial_slope_thresh is {0}".format(intial_slope_thresh_r))
    if abs(change_point[0]-state2_point) > window_size:
        state2_point=change_point[0]-2
    # [state2 start] left limit
    if state2_point<=4:
        state2_point=4
    print("final result, all feature point: {0}, selected point: {1}".format(all_point,state2_point))
    # target [base start] range between 0 and [state2 start]
    scan_slope= slope_ini[0:state2_point,:]
    intial_slope_thresh_b=intial_slope_thresh
    # find point by reach slope thresh on critical point
    print("--find base point--")
    all_point,base_point=_find_left_b_point(channel_shape,scan_slope,intial_slope_thresh_b)
    print("initail intial_slope_thresh is {0}".format(intial_slope_thresh_b))
    print("first result, all feature point: {0}, selected point: {1}".format(all_point,base_point))
    # if no point, adjustment thresh till max_scan_iter
    if base_point<=scan_slope.shape[0]-1:
        print("optimization base point")
        for i in range(max_scan_iter):
            intial_slope_thresh_b+=scan_slope_thresh_iter_add
            all_point,base_point=_find_left_b_point(channel_shape,scan_slope,intial_slope_thresh_b)
            if base_point > scan_slope.shape[0]-1:
                break
        print("base_point intial_slope_thresh is {0}".format(intial_slope_thresh_b))
    print("final result, all feature point: {0}, selected point: {1}".format(all_point,base_point))
    # [base start] left limit
    if state2_point-2<=base_point or base_point < 2:
        base_point=2
    result=[
        [0,base_point-1],
        [base_point,state2_point-1],
        [state2_point,state3_point-1],
        [state3_point,change_point[1]-1],
        [change_point[1],nda.shape[0]-1],
        [None,None]]  
    # NaN is float type
    result=DataFrame( 
    zip(*result),
    columns=[value.value for value in TimeSeriesStep.__members__.values()]).astype('Int64')
    # https://pandas.pydata.org/pandas-docs/version/0.24/whatsnew/v0.24.0.html#optional-integer-na-support
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.isna.html#pandas.isna
    print(result)
    return result
def shift_ts_point(ts_point_df:DataFrame,shift:int)->DataFrame:
    """shift DataFrame TimeSeriesStep points , 

    args
    ---------
    df: DataFrame

        input data

    shift: int

        shift size
    
    return 
    ---------
    DataFrame

    """
    df2=ts_point_df.copy()
    for i in range(ts_point_df.shape[0]):
            for j in range(ts_point_df.shape[1]):
                if df2.values[i,j]!=0 and ts_point_df.values[i,j]!=None:
                    df2.iloc[i,j]=ts_point_df.values[i,j]+shift
    return df2

def x_resample_1d(nda:np.ndarray,sample_num:int)->np.ndarray:
    """lerp resample on axis 0

    """
    result=np.empty((sample_num, nda.shape[1]))
    #col之內各自插值成新x_sample的cols
    for i in range(0,nda.shape[1]):
        result[:,i]=np.interp(np.arange(0, sample_num, 1), np.arange(0, nda.shape[0],1), nda[:,i])
    return result

def smoother(nda:np.ndarray,window_size:int=11,polyorder:int=1)->np.ndarray:
    """savgol_filter for smoother on axis 0

    """
    from scipy.signal import savgol_filter
    if window_size % 2==0:
        window_size=window_size+1
    result=np.empty(nda.shape)
    for i in range(0,nda.shape[1]):
        result[:,i]=savgol_filter(nda[:,i], window_size, polyorder, mode='nearest')
    return result

def f_normalize(nda:np.ndarray,disable_features)->np.ndarray:
    """normalize each max feature(col) value on axis 0
    
    """
    #各cols取abs最大
    maxrows=np.amax( np.abs(nda)  ,axis=0)
    #去除多餘features, 再取rows最大, +1防止0
    max= np.amax(np.delete(maxrows, list(map(lambda x: x-1,disable_features))))+1
    #全arr一齊/max
    return nda/max
def f_normalize_abs(no_index_arr,disable_features)->np.ndarray:
    """index以外cols的值 ((abs(x)+1)^2)-1  abs後增強訊號
    """
    arr=np.power(np.abs(no_index_arr)+1,2)-1 
    return f_normalize(arr,disable_features)

def x_resample_2d(nda:np.ndarray,x_sample:Tuple[int,int],repeat:int=1)->np.ndarray:
    """lerp resample on axis 0 and reshape to 2d

    """
    arrImg=x_resample_1d(nda,x_sample[0]*x_sample[1])
    arrImg=arrImg.reshape ( (x_sample[0],x_sample[1], nda.shape[1]), order='C') 
    #print(arrImg.shape)
    if repeat>1:
        arrImg=arrImg.repeat(repeat,axis=0)  
        #print(arrImg.shape)
        arrImg=arrImg.repeat(repeat,axis=1)
    return arrImg

def regular_strides_slice(change_point:np.ndarray,
        end_index:int,
        head_index:int=0,
        start_slice_line_index:int=0,
        end_slice_line_index:int=None,
        slice_strides_line_num:int=1,
        slice_head_margin:int=50,
        slice_tail_margin:int=0,
        )->np.ndarray:
    """get slice point by regular strides with margin, 
    
    args
    --------

    end_index: int, the data end at, set None means as end data

    *data_pick= data[head_index:end_index]

    kwargs
    --------

    head_index: int, the data start at

    start_slice_line_index: int, the first slice line of change_point_num index

    end_slice_line_index: int, the last slice line of change_point_num index, set None means as end line

    slice_strides_line_num: int, the strides of slice line walk, more then 1 means skip regular lines.

    *line_pick= change_point[start_slice_line_index:end_slice_line_index:slice_strides_line_num]

    slice_head_margin: int, the  head  margin will add to each piece

    slice_tail_margin: int, the  tail margin will add to each piece

    """
    slice_point=[]
    if end_slice_line_index==None:
        point_pick= change_point[start_slice_line_index::slice_strides_line_num]
    else:
        point_pick= change_point[start_slice_line_index:end_slice_line_index:slice_strides_line_num]

    for i,p in enumerate(point_pick):
        if i ==0:
            #first point
            slice_point.append([head_index,int(p)+slice_tail_margin])
        elif i == len(point_pick)-1:
            #last 2nd point
            slice_point.append([int(point_pick[i-1]-slice_head_margin),int(p)+slice_tail_margin])
            #last point
            slice_point.append([int(p-slice_head_margin),end_index])
        else:
            #middle point
            slice_point.append([int(point_pick[i-1]-slice_head_margin),int(p)+slice_tail_margin])
    return np.asarray( slice_point)





