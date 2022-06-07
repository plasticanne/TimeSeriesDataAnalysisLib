import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import TimeSeriesDataAnalysisLib.interface.untrainable_stored_v1_2 as un_v1_2
from typing import List,Union
import attr
import enum
import json,msgpack
VERSION='project1_v1_2'


CFG=['Cfg0','Cfg1','Cfg2','Cfg3','Cfg4','Cfg5']

SENSORS={
0:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
    },
1:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
},
2:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
},
3:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
},
4:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
},
5:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
},
6:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
},
7:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
},
8:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
},
9:{
    "name":"_#cate2_v#0",
    "cfg":[1,1,1,1,1,1],
},
10:{
    "name":"_#cate2_v#0",
    "cfg":[1,1,1,1,1,1],
},
11:{
    "name":"_#cate2_v#0",
    "cfg":[1,1,1,1,1,1],
},
12:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
},
13:{
    "name":"_#cate1_v#0",
    "cfg":[1,1,1,1,1,1],
}
}

def feature_name():
    result=[]
    for i,v in SENSORS.items():
        result.append(v["name"])
    return result
# project interface has to extends BasicObject or Others
# # build interface for Project research data
ArrayDataObject=un_v1_2.ArrayDataObject


@attr.s(auto_attribs=True)
class DataInfoObject(un_v1_2.DataInfoObject):
    # set project class interface version as a private attribute
    in_ver:str=VERSION # _name as a private attribute
   
    # with @attr.s, use   __attrs_post_init__ instead of  __init__
    def __attrs_post_init__(self):
        super(DataInfoObject, self).__attrs_post_init__()
        
@attr.s(auto_attribs=True)
class ResearchDataObject(un_v1_2.ResearchArrayDataObject):
    # mapping field to extended
    info:DataInfoObject=None

def get_file(local_file):
        with open(local_file,"r", encoding='utf-8') as f:
            return f.read()
    # get file list
def file_list(folder,key):
    sets=[]
    for path, _, files in os.walk(folder):
        for name in files:
            if name.split('key_')[-1]==key:
                sets.append(os.path.join(path, name))
    return sets

def dump_local_file(content:Union[str,bytes],file_name:str,file_folder:str):
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    if type(content)==str:
        return open( os.path.join(file_folder, file_name),"w").write(  content  )
    elif type(content)==bytes:
        return open(os.path.join(file_folder, file_name),"wb").write(  content  )
    else:
        raise TypeError("Non supported type")
def _convert():
    sets=file_list("./fakedata1",'000.json')
    for a_file in sets:
        dictO=json.loads(get_file(a_file))
        # dict object -> class object
        dictO["info"]['ex_ver']='untrainable_stored_v1_2'
        dictO["info"]['in_ver']= 'project1_v1_2'
        dictO["annotation"]={}
        dictO["annotation"]["feature_name"]=feature_name()
        data2={}
        data2['measure_time']=[]
        data2['put_time']=[]
        data2['cfg']=[]
        data2['value']=[]
        for data in dictO["data"]:
            data2['measure_time'].append(data['measure_time'])
            data2['put_time'].append(data['put_time'])
            feature_v=[]
            feature_cfg=[]
            for i,feature in enumerate( data['features']):
                feature_v.append(feature['value'])
                if i <14:
                    feature_cfg.append( [SENSORS[i]["cfg"][2],SENSORS[i]["cfg"][1],None,SENSORS[i]["cfg"][0],None] )
                else:
                    feature_cfg.append( [None])
            data2['cfg'].append(feature_cfg)
            data2['value'].append(feature_v)
        data2['shape']=(len(data2['value']),16)
        dictO['data']=data2
        researchData=ResearchDataObject().loads(dictO,force=False)
        researchData.validate()
        dump_local_file(researchData.dumps(),'key_000.msgpack',os.path.join("./fakedata2",researchData.index_id) )
        dump_local_file(researchData.dumps(format='json'),'key_000.json', os.path.join("./fakedata2",researchData.index_id) )






if __name__ == '__main__':
    _convert()