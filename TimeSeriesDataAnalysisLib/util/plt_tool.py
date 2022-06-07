import os
from typing import List,Dict
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid
import seaborn as sns
import TimeSeriesDataAnalysisLib.algorithm.single_ts_algorithm
TimeSeriesStep=TimeSeriesDataAnalysisLib.algorithm.single_ts_algorithm.TimeSeriesStep

#MARKER=[".","v","s","p","P","*","h","H","+","x","X","D","d","|","_","2",'$...$']
CMAP = plt.cm.get_cmap('jet')
def simple_draw_ts_point_lines(df:DataFrame,ts_point:DataFrame,enable_feature_names:List[str])->plt.Figure:
    if enable_feature_names is None:
        enable_feature_names=df.columns

    ax=df[enable_feature_names].plot(kind='line',legend=True,cmap=CMAP)
    if ts_point is not None:
        x=ts_point.loc[0, TimeSeriesStep.State1.value]
        if x!= None:
            plt.axvline(x=x,color='r',linestyle='dashed',label='State1')
        x=ts_point.loc[0, TimeSeriesStep.State2.value]
        if x!= None:
            plt.axvline(x=x,color='r',linestyle='dashdot',label='State2')
        x=ts_point.loc[0, TimeSeriesStep.State3.value]
        if x!= None:
            plt.axvline(x=x,color='r',linestyle='dotted',label='State3')
        x=ts_point.loc[0, TimeSeriesStep.State4.value]
        if x!= None:
            plt.axvline(x=x,color='r',linestyle='solid',label='State4')

    patches, labels = ax.get_legend_handles_labels()
    ax.legend(patches, labels, loc='right')
    return ax.get_figure()
    
def simple_batch_plt_ts_point_imgs(df_list:List[DataFrame],enable_feature_names:List[str]=None,ts_point_list:List[DataFrame]=None,title_list:List[str]=None)->List[plt.Figure]:
    sns.set()
    figs=[]
    for i,df in enumerate(df_list):
        if ts_point_list is None:
            fig=simple_draw_ts_point_lines(df,None,enable_feature_names=enable_feature_names)
        else:
            fig=simple_draw_ts_point_lines(df,ts_point_list[i],enable_feature_names=enable_feature_names)
        if title_list is not None:
            plt.title(title_list[i])
        #plt.tight_layout()
        fig.set_size_inches(16, 9)
        figs.append(fig)
    return figs
def simple_draw_change_point_lines(df:DataFrame,change_point:np.ndarray,enable_feature_names:List[str])->plt.Figure:
    if enable_feature_names is None:
        enable_feature_names=df.columns
    ax=df[enable_feature_names].plot(kind='line',legend=True,cmap=CMAP)
    if change_point is not None:
        for x in change_point:
            plt.axvline(x=x,color='r',linestyle='dashed')
    patches, labels = ax.get_legend_handles_labels()
    ax.legend(patches, labels, loc='right')
    return ax.get_figure()
    
def simple_batch_plt_change_point_imgs(df_list:List[DataFrame],enable_feature_names:List[str]=None,change_point_list:List[np.ndarray]=None,title_list:List[str]=None)->List[plt.Figure]:
    sns.set()
    figs=[]
    for i,df in enumerate(df_list):
        if change_point_list is None:
            fig=simple_draw_change_point_lines(df,None,enable_feature_names=enable_feature_names)
        else:
            fig=simple_draw_change_point_lines(df,change_point_list[i],enable_feature_names=enable_feature_names)
        if title_list is not None:
            plt.title(title_list[i])
        #plt.tight_layout()
        fig.set_size_inches(16, 9)
        figs.append(fig)
    return figs
def save_imgs(figs:List[plt.Figure],output_dir:str,filename_list:List[str])->None:
    #plt.show(figs)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i,fig in enumerate(figs):
        
        fig.savefig(os.path.join(output_dir,filename_list[i]),dpi = 120)
        #fig.clf()