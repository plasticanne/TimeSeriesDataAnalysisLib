
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import TimeSeriesDataAnalysisLib.interface.untrainable as ib
import TimeSeriesDataAnalysisLib.interface.untrainable_stored_v1_2 as un_v1_2



from typing import List,Union
import attr

VERSION='project1_v1_2'
# project interface has to extends BasicObject or Others
# # build interface for Project research data


@attr.s(auto_attribs=True)
class ProjDataInfoObject(un_v1_2.DataInfoObject):
    # set project class interface version as a private attribute
    in_ver:str=VERSION # _name as a private attribute
    # add project class field
    link_set:str=None
    link_id:str=None
    # with @attr.s, use   __attrs_post_init__ instead of  __init__
    def __attrs_post_init__(self):
        super(ProjDataInfoObject, self).__attrs_post_init__()
        
@attr.s(auto_attribs=True)
class ProjResearchDataObject(un_v1_2.ResearchArrayDataObject):
    # mapping field to extended
    info:ProjDataInfoObject=None



