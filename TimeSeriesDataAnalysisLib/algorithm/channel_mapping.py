import copy 
import numpy as np

def get_cfg_value(cfg_array):
    if len(cfg_array.shape)==3:
        # process all data all channel
        Cfg0=cfg_array[:,:,0]
        Cfg1=cfg_array[:,:,1]
        Cfg2=cfg_array[:,:,2]
        Cfg3=cfg_array[:,:,3]
        Cfg4=cfg_array[:,:,4]
        Cfg5=cfg_array[:,:,5]
    else:
        # process all data on 1 channel
        Cfg0=cfg_array[0]
        Cfg1=cfg_array[1]
        Cfg2=cfg_array[2]
        Cfg3=cfg_array[3]
        Cfg4=cfg_array[4]
        Cfg5=cfg_array[5]
    return Cfg0,Cfg1,Cfg2,Cfg3,Cfg4,Cfg5


def fn_cate1_v(value,cfg_array):
    Cfg0,Cfg1,Cfg2,Cfg3,Cfg4,Cfg5=get_cfg_value(cfg_array)

    Vs= value/Cfg0
    Vx = Cfg1 - Vs
    Rs= Cfg3 * Vx *100
    return Rs
def fn_cate2_v(value,cfg_array):
    Cfg0,Cfg1,Cfg2,Cfg3,Cfg4,Cfg5=get_cfg_value(cfg_array)

    Vs= value/Cfg0
    Vx = Cfg1 - Vs
    Rs=Cfg3 * Vx /10 
    return Rs

def fn_cate3_v(value,cfg_array):
    Cfg0,Cfg1,Cfg2,Cfg3,Cfg4,Cfg5=get_cfg_value(cfg_array)
    Is=(value-Cfg2)/Cfg0
    return Is
def fn_cate1_r(value,cfg_array):
    Rs=value
    return Rs

def fn_cate3_r(value,cfg_array):
    Cfg0,Cfg1,Cfg2,Cfg3,Cfg4,Cfg5=get_cfg_value(cfg_array)
    Is=Cfg2/value
    return Is

def fn_cate1_i(value,cfg_array):
    Cfg0,Cfg1,Cfg2,Cfg3,Cfg4,Cfg5=get_cfg_value(cfg_array)
    Rs=Cfg2/value
    return Rs

def fn_cate3_i(value,cfg_array):
    Is=value
    return Is
def other(value,cfg_array):
    return value

#convert unit form different category value by key
# category_unit
CHANNEL_CATEGORY={
    "cate1_v":{
        "set":[],
        "alg":fn_cate1_v,
        },
    "cate2_v":{
        "set":[],
        "alg":fn_cate2_v,
        },
    "cate3_v":{
        "set":[],
        "alg":fn_cate3_v,
        },
    "cate1_r":{
        "set":[],
        "alg":fn_cate1_r,
        },
    "cate2_r":{
        "set":[],
        "alg":fn_cate1_r,
        },
    "cate3_r":{
        "set":[],
        "alg":fn_cate3_r,
        },
    "cate1_i":{
        "set":[],
        "alg":fn_cate1_i,
        },
    "cate2_i":{
        "set":[],
        "alg":fn_cate1_i,
        },
    "cate3_i":{
        "set":[],
        "alg":fn_cate3_i
        },
    "other":{
        "set":[],
        "alg":other
        },
}

CHANNEL_MOX=['cate1','cate2','cate3']


def get_alg_name(channel_name):

    return channel_name.split("#")[1]


def feature_process_all_rows(data:np.ndarray,alg_set_dict,feature_name,cfg_array:list):
    
    alg_set=copy.deepcopy(alg_set_dict)
    result=np.zeros(data.shape)
    for col_i,name in enumerate( feature_name):
        alg=get_alg_name(name)
        if alg in alg_set.keys():
            alg_set[alg]["set"].append(col_i)
        else:
            alg_set["other"]["set"].append(col_i)
    
    for algkey,algvalue in alg_set.items():
        if algvalue["set"]!=[]:
            m=algvalue["alg"]
            cols=algvalue["set"]
            app=[]
            for ro in cfg_array:
                app.append(  [ro[i] for i in cols])
            result[:,cols]=m(data[:,cols] ,np.asarray(  app   ))
    return result


def feature_process_one_row(data:np.ndarray,alg_set_dict,feature_name,cfg_array:list):
    alg_set=copy.deepcopy(alg_set_dict)
    result=np.zeros(data.shape)

    for col_i,name in enumerate( feature_name):
        alg_name= get_alg_name(name)
        if alg_name in alg_set.keys():
            mathod=alg_set[alg_name]["alg"]
        else:
            mathod=alg_set["other"]["alg"]
        result[:,col_i] =mathod(data[:,col_i],np.asarray(cfg_array[col_i]) )
    return result