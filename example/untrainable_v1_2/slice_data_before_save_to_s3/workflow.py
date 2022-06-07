
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..','..'))
import example.slice_data_before_save_to_s3.interface as interface
import numpy as np
from pandas import DataFrame
import json,copy
import TimeSeriesDataAnalysisLib.interface.untrainable_stored_v1_1 as un_v1_1
import TimeSeriesDataAnalysisLib.interface.trainable_stored_v1_0 as tr_v1_0
import TimeSeriesDataAnalysisLib.util.plt_tool as plt_tool
import TimeSeriesDataAnalysisLib.algorithm.single_ts_algorithm as ts
import TimeSeriesDataAnalysisLib.algorithm.ts_analysis as ta


class ProjResearchDataProvider(un_v1_1.ResearchDataProvider):
    # for a normal Time Series project, extends TimeSeriesInjection.

    def feature_process(self,data:interface.DataObject,feature:interface.FeatureObject)->(int,float):
        # override this method for define feature process on each feature value
        for i in interface.SENSORS.keys():
            if feature.name==interface.SENSORS[i]["name"]:
                if feature.cfg==None:
                    value=feature.value
                else:
                    Vx=feature.cfg[1]-feature.value
                    if i in [0,1,2,3,4,5,6,11,12,13]:
                        # Rf*(Vdd-Vs)/Vs
                        value=feature.cfg[0]*Vx/feature.value
                    elif i in [7,8,9,10]:
                        # Rf*Vs/(Vdd-Vs)
                        value=feature.cfg[0]*feature.value/Vx
                    else:
                        raise ValueError("unknow sensor cfg")
                return i,value
    def label_process(self):
        """override this method for how to get label
        if retrun None means this Injection whih no label"""
        return None



class SliceData:
    # the detail fo Injection process
    def __init__(self,research_data_provider:ProjResearchDataProvider,output_dir):
        self.output_dir=output_dir
        self.research_data_provider=research_data_provider

    
        
    def plot_slice_line(self):
        #plot a img to see how many lines we should detect, and estimate how to slice
        # the following parameters may need to adjustment case by case
        # the change points you need
        detect_change_point_num=4
        # the smooth window
        window_size=30
        # the col indexes of only being considered 
        effective_col_indexes=[0,1,3,4,6,7,8,9,10]
        plot=ta.DataSliceAnalysis(self.research_data_provider,self.output_dir)
        plot.plot_slice_line('data.json',
            detect_change_point_num,
            window_size=window_size,
            effective_col_indexes=effective_col_indexes)
        
    def dump_slice(self):
        #slice the data by a regular strides method
        # those parameters we already decide at last step
        detect_change_point_num=4
        window_size=30
        effective_col_indexes=[0,1,3,4,6,7,8,9,10]
        # the start line of we are going to slice
        start_slice_line_index=2
        # the strides of me move on every slice, me may not slice every lines by case
        slice_strides_line_num=1
        # the head and tail margin size of every sliced piece
        slice_head_margin=60
        slice_tail_margin=-10
        ana=ta.DataSliceAnalysis(self.research_data_provider,self.output_dir)
        slice_point=ana.get_slice_point(
            detect_change_point_num,
            start_slice_line_index=start_slice_line_index,
            slice_strides_line_num=slice_strides_line_num,
            slice_head_margin=slice_head_margin,
            slice_tail_margin=slice_tail_margin,
            window_size=window_size,
            effective_col_indexes=effective_col_indexes,
            )
        name=self.research_data_provider.get_data_object().info.name
        for i,piece in enumerate(slice_point):
            #plot the partitive image
            ana.plot_img("{0}_{1}".format(name,i),piece,partitive=True,effective_col_indexes=effective_col_indexes)
            #dump the result data
            filename="{0}_{1}".format(name,i),
            with open(os.path.join(self.output_dir,"{0}.json".format(filename)),'w') as f:
                f.write(ana.dumps_json(piece) )



def main(act):
    
    
    def get_file(local_file):
        with open(local_file,"r", encoding='utf-8') as f:
            return f.read()
    # there is a data with multiple data round, you need to slice it to be a unit
    # and the data was format to interface.ResearchDataObject already, we just load it
    slice_target='D:\\py\\TimeSeriesDataAnalysisLib\\TimeSeriesDataAnalysisLib\\example\\slice_data_before_saveto_s3\\fakedata\\data.json'
    output_dir="E:\\download\\slice_output"
    dictO=json.loads(get_file(slice_target))
    researchData=interface.ResearchDataObject().loads(dictO)

    # define the features' name
    feature_name=[str(j) for j in interface.SENSORS.keys()]
    # process data with ResearchDataProvider
    research_data_provider=ProjResearchDataProvider(researchData,feature_name,interface.VERSION)

    slice_data=SliceData(research_data_provider,output_dir)
    if act=='plot_all':
        #plot a img to see how many lines we should detect, and estimate how to slice
        slice_data.plot_slice_line()
    elif act=='slice':
        #slice the data by a regular strides method, and dump
        slice_data.dump_slice()

    
if __name__ == '__main__':
    import argparse
    flags = argparse.ArgumentParser()
    flags.add_argument('--act',  required=True,help='set acting')
    FLAGS = flags.parse_args()
    main(FLAGS.act)