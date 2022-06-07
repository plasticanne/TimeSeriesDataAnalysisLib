from functools import reduce
from TimeSeriesDataAnalysisLib.algorithm.tf_base import tf,M,L,K
from TimeSeriesDataAnalysisLib.algorithm.net.keras.base import BaseKerasModelBuilder
from TimeSeriesDataAnalysisLib.algorithm.net.base import get_name

class OneLayerMLP(BaseKerasModelBuilder):

    def __init__(self,inputs_shape:list,class_num:int):
        super().__init__(inputs_shape,class_num)
    def create_model(self,*args, **kwargs):
        inputs=[]
        outputs=[]
        ## build your model here ##
        inpt = L.Input(shape=self.inputs_shape[0])
        
        inputs.append(inpt)
        
        new_shape=reduce(lambda x,y:x * y, self.inputs_shape[0])
        x = L.Dense(new_shape, activation='relu')(inpt)
        x = L.Flatten()(x)
        x = L.Dense(self.class_num,activation='softmax')(x)
        
        outputs.append(x)
       
        ##
        self.model = M.Model(inputs=inputs, outputs=outputs)
        return  self.isModelExist()
        
    def compile(self):
        optimizer=M.optimizers.Adam(lr=1e-4)
        loss=M.losses.CategoricalCrossentropy()
        self.model.compile(optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                M.metrics.RootMeanSquaredError(name='rmse'),
                #ba.rmse_metrics,
                M.metrics.MeanAbsoluteError(name='mae'),
                #ba.mae_metrics,
                M.metrics.top_k_categorical_accuracy,
            ]) 
        # set the params logging
        self.params_log.update({
            'optimizer':get_name(optimizer),
            'loss':get_name(loss),
        })
        # map the metrics logging names
        self.metrics_names=[
            'loss',
            'acc',
            'RMSE',
            'MAE',
            'top_5_acc']



    