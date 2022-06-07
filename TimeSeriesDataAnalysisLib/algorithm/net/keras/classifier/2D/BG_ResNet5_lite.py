import os
NAME=os.path.basename(__file__).split(".")[0]
DIMENSION='2D'
import tensorflow
tf=tensorflow.compat.v1
M=tf.keras
L=M.layers
K=M.backend

DTYPE=tf.float32
K.set_floatx('float32')
# def zeropad(x):
#     y = K.zeros_like(x)
#     return K.concatenate([x, y], axis=3)

class MyConv2D(L.Layer):
    def __init__(self, *args, **kwargs):
        super(MyConv2D, self).__init__()
        self.args=args
        self.kwargs=kwargs

    def call(self, input_node):
        with tf.variable_scope('nn'):
            nb_filter=self.args[0]
            kernel_size=self.args[1]
            w = tf.get_variable(
                name='weight',
                shape=[kernel_size[0], kernel_size[1],  input_node.shape[3] ,nb_filter],
                dtype=tf.float32)
            b = tf.get_variable(
                name='bias',
                shape=[nb_filter],
                dtype=tf.float32)
            
        return tf.nn.conv2d(input_node, filter=w, strides=self.kwargs["strides"], padding='SAME')+b

def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
 
    x = L.Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name,dtype=DTYPE)(x)
    #x = MyConv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name,dtype=DTYPE)(x)
    #x = L.BatchNormalization(axis=3,name=bn_name,dtype=DTYPE)(x,training=True)
    return x
 
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False, maxp=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')

    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=(1,1))
#         shortcut = zeropad(inpt)
        x = L.add([x,shortcut],dtype=DTYPE)
        return x
    else:
#         if maxp:
#             shortcut = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(inpt)
#     #         shortcut = zeropad(inpt)
#             x = add([x,shortcut])
#             return x
        
        x = L.add([x,inpt],dtype=DTYPE)
        return x
    
    
    return x
def get_model(inputCNNshape,nb_classes):
    # Constants for the AVletters dataset
    #min_n_frames = 1
    #inputCNNshape = (72, 72, min_n_frames)
    #nb_classes = 10
    # Constants for the Digits dataset
    inpt = L.Input(shape=inputCNNshape,dtype=DTYPE)
    if inputCNNshape[0]==48:
        pool_size=(3,3)
        
    elif inputCNNshape[0]==72:
        pool_size=(5,5)
    else:
        raise tf.errors.InvalidArgumentError
#     x = ZeroPadding2D((3,3))(inpt)
    x = Conv2d_BN(inpt,nb_filter=32,kernel_size=(3,3),strides=(2,2))
#     x = Conv2d_BN(inpt,nb_filter=32,kernel_size=(3,3))
#     x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    
    #(48,48,16)
    
#     x = Conv2d_BN(inpt,nb_filter=32,kernel_size=(3,3))
    
#     x = Conv_Block(x,nb_filter=16,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=16,kernel_size=(3,3))
     
        
    #(24,24,32)
#     x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))
    
    x = Conv2d_BN(inpt,nb_filter=32,kernel_size=(3,3),strides=(2,2))
   #     x = Conv_Block(x,nb_filter=32,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))

#     x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))



#     x = Conv_Block(x,nb_filter=32,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))
    
    
    #(12,12,64)
    x = Conv2d_BN(inpt,nb_filter=64,kernel_size=(3,3),strides=(2,2))
 #   x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))

#     x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
#     x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))

    
    #(6,6,128)
    x = Conv2d_BN(inpt,nb_filter=128,kernel_size=(3,3),strides=(2,2))
 #   x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))

#     x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
#     x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
#     x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
#     x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
#     (7,7,512)
#     x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
#     x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))

    x = L.AveragePooling2D(pool_size=pool_size,dtype=DTYPE)(x)
    x = L.Flatten(dtype=DTYPE)(x)
#     x = BatchNormalization()(x)
    #x = Dropout(0.2)(x, training=True)
    """x = Dense(128, 
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l1(0.01)
    )(x)"""
    x = L.Dense(128, activation='relu',dtype=DTYPE)(x)
#     x = BatchNormalization()(x)
    #x = Dropout(0.2)(x, training=True)
    
#     x = Dropout(0.5)(x)
    
    
    x = L.Dense(nb_classes,activation='softmax',dtype=DTYPE)(x)

    # Return the model object
    model = M.Model(inputs=inpt, outputs=x)
    return model