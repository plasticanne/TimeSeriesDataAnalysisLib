import TimeSeriesDataAnalysisLib.algorithm.batch_algorithm as ba
from TimeSeriesDataAnalysisLib.interface.trainable_data_provider import TrainableDataProvider
import TimeSeriesDataAnalysisLib.algorithm.util.tfgraph_ctrl as pg
import TimeSeriesDataAnalysisLib.algorithm.util.visualize as vis
from TimeSeriesDataAnalysisLib.algorithm.net.base import AbcWorkflow,AbcModelBuilder,my_raise,set_defaut,get_name

from sklearn.metrics import accuracy_score
import abc,os,shutil
import numpy as np
from typing import List,Dict,Union,Tuple,Generator
from TimeSeriesDataAnalysisLib.algorithm.tf_base import tf,M,L,K,one_hot_encoder

class BaseKerasModelBuilder(AbcModelBuilder):
    """Keras model builder for Keras training task

    *for create new model, you should override create_model() and compile()

    *for loading exist h5 model, you can just use this class

    args
    ----------
    inputs_shape : List[Tuple[int]]
        the shapes of inputs list

    class_num : int
        the classify number
    """
    def __init__(self,inputs_shape:List[Tuple[int]],class_num:int):
        self.inputs_shape=inputs_shape
        self.class_num=class_num
        self.params_log={}
        self.metrics_names=[]
        self.result_dir='result'
        self.model_dir='keras_model'
        self.tb_name='TensorBoard'
        self.model_filename='keras_model.h5'
        self.freeze_model_filename='freeze.pb'
        self.lite_model_filename='lite_model.tflite'
        self.quantized_model_filename='quantized_model.tflite'
        self.saved_model_dir='saved_model'
        self.visualized_lite_model_filename='lite_model.html'
        self.visualized_quantized_model_filename='quantized_model.html'
        self.report_filename='report.txt'
        self.sess=None
        self.sess_config= tf.ConfigProto()

    def create_model(self,*args, **kwargs)->bool:
        inputs=[]
        outputs=[]
        ## build your model here ##
        
        ## 
        self.model = M.Model(inputs=inputs, outputs=outputs)
        return self.isModelExist()
    def compile(self):
        ## build your compile here ##
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
        # this must be set for logging
        self.params_log.update({
            'optimizer':get_name(optimizer),
            'loss':get_name(loss),
        })
        # this must be set for logging
        self.metrics_names=(
            'loss',
            'acc',
            'RMSE',
            'MAE',
            'top_5_acc')
    def fit(self,*args, **kwargs):
        """same to keras"""
        return self.model.fit(*args, **kwargs)
    def evaluate(self, *args, **kwargs):  
        """same to keras"""
        return self.model.evaluate(*args,**kwargs)
    def fit_generator(self,*args, **kwargs):
        """same to keras"""
        return self.model.fit_generator(*args, **kwargs)
    def evaluate_generator(self,*args, **kwargs):
        """same to keras"""
        return self.model.evaluate_generator(*args, **kwargs)
    def summary(self):
        """same to keras"""
        self.model.summary()

    def load_model(self,model_dir:str):
        """load h5 format model, other formats you have to see evaluate_converted_model"""
        self.model=M.models.load_model(
            os.path.join(model_dir,self.model_filename))
        # rename outputs node 
        pg.package_graph(self.model,self.inputs_shape)
        return self.isModelExist()

    def save_model(self,model_dir:str):
        """save model as h5 format, you need h5 convert to other formats, which are in convert_model()"""
        M.models.save_model(
                self.model,
                os.path.join( model_dir, self.model_filename),
                overwrite=True,
                include_optimizer=True,
                save_format='h5',          
            )

    def set_gpu(self,index:int):
        """select gpu index which you want to use"""
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess_config= tf.ConfigProto(gpu_options=gpu_options)

    def init_new_kares_sess(self):
        """clean graph and sess before"""
        K.clear_session()
        sess = tf.Session(config=self.sess_config)
        K.set_session(sess) 
        return sess

    def start_new_kares_task(self):
        """start_new_kares_task is a Simplification of keras sess and default_graph control, 
        this will clean graph and sess before, the usage just like sess """
        sess=self.init_new_kares_sess()
        sess.as_default()
        self.sess=sess
        return self

    def start_new_tf_task(self):
        """start_new_tf_task is a Simplification of tf sess and default_graph control, 
        this will clean graph and sess before, the usage just like sess """
        self.end_task()
        sess = tf.Session()
        sess.as_default()
        self.sess=sess
        return self

    def end_task(self):
        """close sess and reset_default_graph
        or you can instead it with

        >>> with start_new_kares_task:
        >>>     do something...

         
        """
        self.sess.close()
        tf.reset_default_graph()
        self.model=None

    def __enter__(self):
        return self

    def __exit__(self,exc_type, exc_val, exc_tb):
        self.end_task()

    def convert_model(self,output_dir:str,frozen_graph:bool=False,lite_graph:bool=False,quantized_graph:bool=False,override:bool=False):
        """convert h5 -> saved_model -> frozen_graph,lite_graph,quantized_graph

        args
        ----------
        output_dir : str
            the shapes of inputs list

        frozen_graph : bool
            output frozen graph

        lite_graph : bool
            output lite graph  
        
        quantized_graph : bool
            output quantized graph

        override : bool
            override output file if exists
        """

        freeze_model=os.path.join(output_dir,self.freeze_model_filename)
        lite_model=os.path.join(output_dir,self.lite_model_filename)
        quantized_model=os.path.join(output_dir,self.quantized_model_filename)
        saved_model=os.path.join(output_dir,self.saved_model_dir)
        visualized_lite_model=os.path.join(output_dir,self.visualized_lite_model_filename)
        visualized_quantized_model=os.path.join(output_dir,self.visualized_quantized_model_filename)

        # load .h5 file
        self.start_new_kares_task()
        self.load_model(output_dir)

        # output savedModel
        try:
            pg.write_savedModel(self.sess,saved_model)
        except AssertionError as e:
            if override:
                shutil.rmtree(saved_model)
                pg.write_savedModel(self.sess,saved_model)
            else:
                raise e
        self.end_task()
        self.sess = tf.Session()
        # output frozen_graph
        if os.path.isfile(freeze_model) and not override:
            pass
        else:
            with self.sess.as_default(): 
                pg.load_savedModel(self.sess,saved_model)
                if frozen_graph:
                    pg.write_frozen_pb(self.sess, freeze_model)
            self.end_task()
        # output lite_graph
        if os.path.isfile(lite_model) and not override:
            pass
        else:
            if lite_graph:
                pg.write_lite_model_from_saved_model(saved_model, output_dir)
                vis.do_visualize(['',lite_model,visualized_lite_model]) 
            self.end_task()
        # output quantized_graph
        if os.path.isfile(quantized_model) and not override:
            pass
        else:
            if quantized_graph:
                pg.write_quantized_model_from_saved_model(saved_model, output_dir)
                vis.do_visualize(['',quantized_model,visualized_quantized_model]) 
            self.end_task()


    


class KerasWorkflow(AbcWorkflow):
    """workflow for training task detail 

    args
    ----------
    model_builder : BaseSKModelBuilder
        the keras ModelBuilder

    """
    def __init__(self,model_builder:BaseKerasModelBuilder,model_name=None,output_dir=None):
        self.model_builder=model_builder
        self.class_num=self.model_builder.class_num
        self.inputs_shape=self.model_builder.inputs_shape
        self.inputs_num=len(self.inputs_shape)
        self.logging_tb =None
        self.model_name=model_name
        self.output_dir=output_dir

    def set_params_log(self):
        train_size=self.data_provider.x_train_indexes.shape[0]
        valid_size=self.data_provider.x_valid_indexes.shape[0]
        test_size=self.data_provider.x_test_indexes.shape[0]
        size=train_size+valid_size+valid_size
        
        callbacks=set_defaut(self.kwargs,"callbacks",[])
        if self.logging_tb is not None:
            self.kwargs["callbacks"]=[self.logging_tb] + callbacks
        else:
            self.kwargs["callbacks"]= callbacks
        # log params
        params_log={
            'all':size,
            'train':train_size,
            'valid':valid_size,
            'test':test_size,
            "initial_epoch":set_defaut(self.kwargs,"initial_epoch",0),
            "epochs":set_defaut(self.kwargs,"epoch",2),
            'class_num':self.model_builder.class_num,
            'batch_size':self.batch_size,
            'shuffle':set_defaut(self.kwargs,"shuffle",True),
            'callbacks':list(map( get_name  , callbacks))
            }
        params_log.update(self.model_builder.params_log)
        return params_log
    def set_flow(self,data_provider:TrainableDataProvider,batch_size:int,*args,**kwargs):
        """set for training task detail incloud data input, evaluate, logging
        
        """
        self.data_provider=data_provider
        self.batch_size=batch_size
        self.args=args
        self.kwargs=kwargs
        self.logging_tb =None
    def set_tb_path(self,tb_path:str):
        self.logging_tb = M.callbacks.TensorBoard(log_dir=tb_path.replace('/','\\'))
        if not os.path.exists(tb_path):
            os.makedirs(tb_path)
    def predict_evaluate_flow(self):
        # data prepare
        if self.get_x_size('valid')>0 and self.get_y_size('valid')>0:
            x_valid=self.load_x_value_ndarray('valid')
            y_valid=one_hot_encoder(self.load_y_value_ndarray('valid'),self.class_num)
            validation_data=(x_valid, y_valid)
        else:
            validation_data=None    
        history=None
        def process(mode,do_fit):
            nonlocal history
            x_=self.load_x_value_ndarray(mode)
            # if you have multiple input node
            # x1= data_provider1.load_batch_x_values(...
            # x2= data_provider2.load_batch_x_values(...
            # x_=[x1,x2] 
            y_=one_hot_encoder(self.load_y_value_ndarray(mode),self.class_num)
            # if you have multiple input node
            # y1= data_provider1.load_batch_y_values(...
            # y2= data_provider2.load_batch_y_values(...
            # y_=[y1,y2]
            # fit
            if do_fit:
                history=self.model_builder.fit([x_,], [y_,],
                    batch_size=self.batch_size,
                    validation_data=validation_data,
                    verbose=1,
                    **self.kwargs
                    )
            
            # evaluate
            metrics_values = self.model_builder.evaluate([x_,],[y_,],verbose=1)
            return {"names":self.model_builder.metrics_names,"values":metrics_values} 

        
        metrics_log={
            "train":process("train",False),
            "valid": None,
            "test":process("test",False),
            
        }
        return metrics_log
    def train_evaluate_flow(self):
        """workflow for training task detail incloud data input, evaluate, logging
        
        notice that in this case, only have 1 input node and 1 output node
        if you need multiple, you should override this method
        """

        # data prepare
        if self.get_x_size('valid')>0 and self.get_y_size('valid')>0:
            x_valid=self.load_x_value_ndarray('valid')
            y_valid=one_hot_encoder(self.load_y_value_ndarray('valid'),self.class_num)
            validation_data=(x_valid, y_valid)
        else:
            validation_data=None    
        history=None
        def process(mode,do_fit):
            nonlocal history
            x_=self.load_x_value_ndarray(mode)
            # if you have multiple input node
            # x1= data_provider1.load_batch_x_values(...
            # x2= data_provider2.load_batch_x_values(...
            # x_=[x1,x2] 
            y_=one_hot_encoder(self.load_y_value_ndarray(mode),self.class_num)
            # if you have multiple input node
            # y1= data_provider1.load_batch_y_values(...
            # y2= data_provider2.load_batch_y_values(...
            # y_=[y1,y2]
            # fit
            if do_fit:
                history=self.model_builder.fit([x_,], [y_,],
                    batch_size=self.batch_size,
                    validation_data=validation_data,
                    verbose=1,
                    **self.kwargs
                    )
            
            # evaluate
            metrics_values = self.model_builder.evaluate([x_,],[y_,],verbose=1)
            return {"names":self.model_builder.metrics_names,"values":metrics_values} 

        
        metrics_log={
            "train":process("train",True),
            "valid": None,
            "test":process("test",False),
            
        }
        return history,metrics_log
    

    def evaluate_converted_model(self,model_dir:str,outputs_num:int):
        """evaluate saved_model,freeze_model,lite_model,quantized_model, then output report

        args
        ----------

        model_dir : str
            input model dir

        outputs_num : int
            output node number
        """
        report=os.path.join(model_dir,self.model_builder.report_filename)
        freeze_model=os.path.join(model_dir,self.model_builder.freeze_model_filename)
        lite_model=os.path.join(model_dir,self.model_builder.lite_model_filename)
        quantized_model=os.path.join(model_dir,self.model_builder.quantized_model_filename)
        saved_model=os.path.join(model_dir,self.model_builder.saved_model_dir)

        evaluate_result={}
        x_=self.data_provider.load_batch_x_values(self.data_provider.x_train_indexes)
        y_=one_hot_encoder( self.data_provider.load_batch_y_values(self.data_provider.y_train_values),self.model_builder.class_num)

        # evaluate original model
        # self.model_builder.start_new_kares_task()
        # if x_.shape[0] > 0  or  y_.shape[0] > 0:
        #     self.model_builder.load_model(model_dir) 
        #     self.model_builder.compile() 
        #     evaluate_result["keras_model"] = self.model_builder.evaluate(x_,y_)[1]
        # else:
        #     raise RuntimeError("test data should be set")
        # self.model_builder.end_task()

        
        inputs,outputs=pg.gen_in_out_node_name(self.inputs_num, outputs_num)
        # evaluate saved_model
        if os.path.exists(saved_model):
            if os.listdir(saved_model):
                print("--load saved_model--")
                self.model_builder.sess=tf.Session()
                with self.model_builder.sess.as_default(): 
                    pg.load_savedModel(self.model_builder.sess,saved_model)
                    evaluate_result["saved_model"]=pg.graph_evaluate(self.model_builder.sess,[x_],[y_,])
                self.model_builder.end_task()
        # evaluate freeze_model
        if os.path.isfile(freeze_model):
            print("--load freeze_model--")
            self.model_builder.sess=tf.Session()
            with  self.model_builder.sess.as_default():
                pg.load_frozen_pb(freeze_model)
                evaluate_result["freeze_model"]=pg.graph_evaluate(self.model_builder.sess,[x_,],[y_,])
            self.model_builder.end_task()
        # evaluate lite_model
        if os.path.isfile(lite_model):
            print("--load lite_model--")
            evaluate_result["lite_model"]=pg.lite_graph_evaluate(lite_model,[x_,],[y_,])
            self.model_builder.end_task()
        # evaluate quantized_model
        if os.path.isfile(quantized_model):
            print("--load quantized_model--")
            evaluate_result["quantized_model"]=pg.lite_graph_evaluate(quantized_model,[x_,],[y_,])
            self.model_builder.end_task()

        # log report
        with open( report , 'w', encoding='utf-8') as f:
            msg="""input nodes: {0}\noutput nodes: {1}\nbuilder tags: {2}\nsingature_def_map key: {3}\n\n""".format(inputs,outputs,pg.BUILDER_TAGS,pg.SINGATURE_DEF_MAP_KEY)
            for key in evaluate_result.keys():
                msg=msg+"{0} acc: {1}\n".format(key,evaluate_result[key])
            print('\n'+msg)
            print(msg, file=f)

class KerasGeneratorWorkflow(KerasWorkflow):
    def set_params_log(self):
        train_size=self.data_provider.x_train_indexes.shape[0]
        valid_size=self.data_provider.x_valid_indexes.shape[0]
        test_size=self.data_provider.x_test_indexes.shape[0]
        size=train_size+valid_size+valid_size
        
        callbacks=set_defaut(self.kwargs,"callbacks",[])
        if self.logging_tb is not None:
            self.kwargs["callbacks"]=[self.logging_tb] + callbacks
        else:
            self.kwargs["callbacks"]= callbacks
        # log params
        params_log={
            'all':size,
            'train':train_size,
            'valid':valid_size,
            'test':test_size,
            "initial_epoch":set_defaut(self.kwargs,"initial_epoch",0),
            "epochs":set_defaut(self.kwargs,"epoch",2),
            'class_num':self.model_builder.class_num,
            'batch_size':self.batch_size,
            'shuffle':set_defaut(self.kwargs,"shuffle",True),
            'callbacks':list(map( get_name  , callbacks))
            }
        params_log.update(self.model_builder.params_log)
        return params_log
    def evaluate_flow(self):
        """notice that in this case, only have 1 input node and 1 output node
        if you need multiple, you should override this method
        """
        train_size=self.get_x_size('train')
        valid_size=self.get_x_size('valid')
        test_size=self.get_x_size('test')
        # data prepare
        # if you have multiple input node
        # you should change data_generator as yield [x1,x2],[y1,y2]
        data_generator=self.data_provider.data_generator
        if valid_size>0 and self.data_provider.y_valid_values.shape[0]>0:
            validation_data=data_generator(self.data_provider.x_valid_indexes,self.data_provider.y_valid_values, self.model_builder.class_num,self.batch_size,)
            validation_steps=max(1, valid_size//self.batch_size)
        else:
            validation_data=None
            validation_steps=None
        # fit
        history=self.model_builder.fit_generator(
            data_generator(self.data_provider.x_train_indexes,self.data_provider.y_train_values, self.model_builder.class_num,self.batch_size,),
            steps_per_epoch=max(1, train_size//self.batch_size ),
            validation_data=validation_data,
            validation_steps=validation_steps,
            verbose=1,
            **self.kwargs
            )
        # evaluate
        train_metrics={
            "names":self.model_builder.metrics_names,
            "values":self.model_builder.evaluate_generator(
                    data_generator(self.data_provider.x_train_indexes,self.data_provider.y_train_values, self.model_builder.class_num,self.batch_size,shuffle=False),
                    steps=train_size,
                    verbose=1)
            }
        if test_size >0:
            test_metrics={
            "names":self.model_builder.metrics_names,
            "values":self.model_builder.evaluate_generator(
                data_generator(self.data_provider.x_test_indexes,self.data_provider.y_test_values, self.model_builder.class_num,self.batch_size, shuffle=False),
                steps=test_size,
                verbose=1)
            }
            
        else:
            test_metrics=None
        # log evaluate
        metrics_log={
            "train":train_metrics,
            "valid":None,
            "test":test_metrics,
            }
        return history,metrics_log