from TimeSeriesDataAnalysisLib.algorithm.tf_base import tf,M,L,K

import numpy as np
import os
import time
from typing import List,Dict
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib

from tensorflow.python.tools import freeze_graph 
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

import TimeSeriesDataAnalysisLib.algorithm.batch_algorithm as ba
from sklearn.metrics import accuracy_score
NPTYPE=np.float32
DTYPE=tf.float32
# INPUT_NODE_NAME="input0"
# INPUT_TENSOR_NAME="input0:0"
# OUTPUT_NODE_NAME="output0"
# OUTPUT_TENSOR_NAME="output0:0"
BUILDER_TAGS=[tag_constants.SERVING]
SINGATURE_DEF_MAP_KEY='predict'
def _input_node_name(index):
    return "input_{0}".format(index+1)
def _input_tensor_name(index):
    return "input_{0}:0".format(index+1)
def _output_node_name(index):
    return "output_{0}".format(index+1)
def _output_tensor_name(index):
    return "output_{0}:0".format(index+1)
def gen_in_out_node_name(inputs_num:int, outputs_num:int):
    # inputs=[]
    # for i in range(inputs_num):
    #     inputs.append(_input_node_name(i))
    # outputs=[]
    # for i in range(outputs_num):
    #     outputs.append(_output_node_name(i))
    return [_input_node_name(i) for i in range(inputs_num)],[_output_node_name(i) for i in range(outputs_num) ]
def get_in_out_from_graph(sess):
    #print([n.name for n in tf.get_default_graph().as_graph_def().node])
    inputs={}
    outputs={}
    count=0
    while True:
        try:
            tensor=sess.graph.get_tensor_by_name(_input_tensor_name(count))
            inputs[_input_node_name(count)]=tensor
            count+=1
        except KeyError:
            break
    count=0
    while True:
        try:
            tensor=sess.graph.get_tensor_by_name(_output_tensor_name(count))
            outputs[_output_node_name(count)]=tensor
            count+=1
        except KeyError:
            break
    #input_nodes =  [ for node_name in sess.graph.get_tensor_by_name(get_input_node_name(0)).name.split(":")[0]]
    #output_nodes = [sess.graph.get_tensor_by_name(get_output_node_name(0)).name.split(":")[0]]
    print("inputs: ", inputs)
    print("outputs: ", outputs)
    return inputs,outputs

def package_graph(keras_model,inputs_shape:list):
    #_inputs=[]
    #for i,input_shape in enumerate( inputs_shape):
    #    _inputs.append(tf.placeholder(shape=input_shape, name=get_input_node_name(i),dtype=DTYPE))
        #_input =  L.Input(shape=input_shape,name="input0",dtype=DTYPE)
    #_outputs = keras_model(_inputs)
    for i,output in enumerate( keras_model.outputs):
        tf.identity(output, name=_output_node_name(i))
    print("inputs node: ", keras_model.inputs)
    print("outputs node: ", keras_model.outputs)

def lite_converter():
    from tensorflow.lite import main
    
def write_frozen_pb(sess, output_pb_file):
    inputs, outputs=get_in_out_from_graph(sess)
    constant_graph = graph_util.convert_variables_to_constants(sess, 
                                                                sess.graph.as_graph_def(),
                                                                list(outputs.keys()))
    optimize_Graph = optimize_for_inference_lib.optimize_for_inference(
        constant_graph,
        list(inputs.keys()),  # an array of the input node(s)
        list(outputs.keys()),  # an array of output nodes
        DTYPE.as_datatype_enum)
    optimize_for_inference_lib.ensure_graph_is_valid(optimize_Graph)
    with tf.gfile.GFile( output_pb_file, "wb") as f:
        f.write(constant_graph.SerializeToString())
        print("output frozen graph at "+ output_pb_file)
def write_frozen_pb_from_meta(meta_dir,meta_name,output_pb_file,output_node_names):
    freeze_graph.freeze_graph(
			input_graph=meta_dir+'/'+meta_name+'.pb',
			input_binary=False, 
			input_checkpoint=meta_dir+'/'+meta_name+'.ckpt', 
			output_node_names=output_node_names,
			output_graph=output_pb_file,
			clear_devices=False,
			initializer_nodes='',
            input_saver='',
            filename_tensor_name='save/Const:0',
            restore_op_name='save/restore_all'
			)
#@profile
def load_frozen_pb(file_path):
    with tf.gfile.GFile(file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')



def write_meta(sess, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(
        dir, "meta.ckpt"))
    tf.train.write_graph(sess.graph_def,
                         dir, 'meta.pb')
    print("output meta graph at "+os.path.join(dir, 'meta'))
def load_meta(sess, dir):
    checkpoint = tf.train.get_checkpoint_state(
        dir).model_checkpoint_path
    saver = tf.train.import_meta_graph(
        checkpoint + '.meta', clear_devices=True)
    saver.restore(sess, checkpoint)
    tf.import_graph_def(sess.graph_def, name='')


def write_savedModel(sess, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    builder = tf.saved_model.builder.SavedModelBuilder(dir)
    inputs, outputs=get_in_out_from_graph(sess)
    signature = predict_signature_def(inputs=inputs,outputs=outputs)
    builder.add_meta_graph_and_variables(sess=sess,
                                    clear_devices=True,
                                     tags=[tag_constants.SERVING],
                                     signature_def_map={SINGATURE_DEF_MAP_KEY: signature})
    builder.save()
    print("output SavedModel at "+os.path.join(dir))
def write_simpleSavedModel(sess, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    inputs, outputs=get_in_out_from_graph(sess)
    tf.saved_model.simple_save(sess,dir,inputs=inputs,outputs=outputs)
    print("output SavedModel at "+os.path.join(dir))
def load_savedModel(sess, dir):
    tf.saved_model.loader.load(sess,[tag_constants.SERVING],dir)
    graph = tf.get_default_graph()
    


def write_opt_quantized_model(converter,dir,train_data=None):
    
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.SELECT_TF_OPS,
        #tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
                        ]
    converter.target_spec.supported_types = [tf.lite.constants.FLOAT]
    
    # Convert the model to the TensorFlow Lite format with quantization
    #converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    """def representative_data_gen():
        nonlocal train_data
        for input_value in train_data:
            yield [np.asarray([input_value])]
    converter.representative_dataset = representative_data_gen"""
    converter.post_training_quantize=True
    #converter.inference_type = tf.uint8
    converter.inference_input_type = DTYPE
    converter.inference_output_type = DTYPE
    tflite_qmodel = converter.convert()
    # Save the model to disk
    open(dir+'/quantized_model.tflite', "wb").write(tflite_qmodel)
def write_opt_lite_model(converter,dir):
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
                        ]
    tflite_model = converter.convert()
    # Save the model to disk
    open(dir+'/lite_model.tflite', "wb").write(tflite_model)

def write_lite_model(sess, output_dir):
    inputs, outputs=get_in_out_from_graph(sess)
    converter = tf.lite.TFLiteConverter.from_session(sess, inputs.values() , outputs.values())
    write_opt_lite_model(converter,output_dir)
def write_lite_model_from_saved_model(saved_model_dir,output_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir,signature_key=SINGATURE_DEF_MAP_KEY)
    write_opt_lite_model(converter,output_dir)
def write_quantized_model_from_saved_model(saved_model_dir,output_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir,signature_key=SINGATURE_DEF_MAP_KEY)
    write_opt_quantized_model(converter,output_dir)
def write_lite_model_from_fz(file,output_dir,inputs_num,outputs_num):
    inputs,outputs=gen_in_out_node_name(inputs_num, outputs_num)
    converter = tf.lite.TFLiteConverter.from_frozen_graph(file, inputs, outputs)
    write_opt_lite_model(converter,output_dir)
def write_lite_model_from_keras(model,output_dir):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    write_opt_lite_model(converter,output_dir)


def graph_evaluate(sess,x:List[np.ndarray],y:List[np.ndarray]):
    inputs, outputs=get_in_out_from_graph(sess)
    feed_dict={}
    for i,x_ in enumerate(x):
        feed_dict[list(inputs.values())[i]]=x_
    print(feed_dict.keys())
    outputs_v = sess.run(list(outputs.values()),
            feed_dict=feed_dict)
    result=[]
    for i,out in enumerate(outputs_v):
        y_pred=ba.score_to_onehot(out)
        r=accuracy_score( y[i],y_pred)
        result.append(r)
        print("y [{0}]: {1}".format(i,r))
    return result

def lite_graph_evaluate(lite_model,x:List[np.ndarray],y:List[np.ndarray]):
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=lite_model)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #print(input_details)
    #print(output_details)
    outputs_v=[]
    for i,x_ in enumerate(x):
        outputs_v.append([])
        for v in x_:
            interpreter.set_tensor(input_details[i]['index'], np.asarray([v],dtype=NPTYPE))
            interpreter.invoke()
            outputs_v[i].append(interpreter.get_tensor(output_details[i]['index']))
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
    result=[]
    for i,y_true in enumerate(y):
        y_pred=ba.score_to_onehot(np.vstack(outputs_v[i]))
        r=accuracy_score( y_true,y_pred)
        result.append(r)
        print("y [{0}]: {1}".format(i,r))
    return result
