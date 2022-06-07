import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import TimeSeriesDataAnalysisLib.interface.untrainable_stored_v1_1 as iv1_1
import TimeSeriesDataAnalysisLib.interface.untrainable as ib
from typing import List,Union
import attr
import enum
VERSION='test'

SUPPLY_VOLTAGE=5

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

DataObject=iv1_1.DataObject
FeatureObject=iv1_1.FeatureObject


@attr.s(auto_attribs=True)
class DataInfoObject(iv1_1.DataInfoObject):
    # set project class interface version as a private attribute
    _in_ver:str=VERSION # _name as a private attribute
    name:str=None
    # with @attr.s, use   __attrs_post_init__ instead of  __init__
    def __attrs_post_init__(self):
        super(DataInfoObject, self).__attrs_post_init__()
        
@attr.s(auto_attribs=True)
class ResearchDataObject(iv1_1.ResearchDataObject):
    # mapping field to extended
    info:DataInfoObject=None



