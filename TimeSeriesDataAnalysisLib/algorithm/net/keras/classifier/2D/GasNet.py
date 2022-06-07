import os
NAME=os.path.basename(__file__).split(".")[0]
DIMENSION='2D'
import tensorflow
tf=tensorflow.compat.v1
M=tf.keras
L=M.layers
K=M.backend

def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
        
    x = L.Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,name=conv_name)(x)
    
#     x = relu(x, alpha=0.0, max_value=None, threshold=0.0)

    x = L.BatchNormalization(axis=3,name=bn_name)(x)
    x = L.Activation('relu')(x)
    
#     x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
#     x = BatchNormalization(axis=3,name=bn_name)(x)
    
    return x

def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')

    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=(1,1))
        x = L.add([x,shortcut])
        return x
    else:
        x = L.add([x,inpt])
        return x
    
#     return x

def get_model(inputCNNshape,nb_classes):
    # Constants for the AVletters dataset
    #min_n_frames = 9
    #inputCNNshape = (48, 48, min_n_frames)
    #nb_classes = 10
    # Constants for the Digits dataset

    inpt = L.Input(shape=inputCNNshape)
    if inputCNNshape[0]==48:
        pool_size=(12,12)
    elif inputCNNshape[0]==72:
        pool_size=(18,18)
    else:
        raise tf.errors.InvalidArgumentError
#     x = ZeroPadding2D((3,3))(inpt)
    x = Conv2d_BN(inpt,nb_filter=32,kernel_size=(3,3))
    x = Conv2d_BN(x,nb_filter=32,kernel_size=(3,3))
#     x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    
    #(48,48,16)
    
#     x = Conv2d_BN(inpt,nb_filter=32,kernel_size=(3,3))
    
#     x = Conv_Block(x,nb_filter=16,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=16,kernel_size=(3,3))
     
        
    #(24,24,32)
#     x = Conv_Block(x,nb_filter=32,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))
    x = L.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)

    
    #(12,12,64)
#     x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    x = L.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
#     x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))

    
    #(6,6,128)
#     x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    
#     x = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same')(x)
#     x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
#     x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
#     x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
#     (7,7,512)
#     x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
#     x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))

    x = L.AveragePooling2D(pool_size=pool_size)(x)
    x = L.Flatten()(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x, training=True)
    
#     x = Dense(32, activation='relu')(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x, training=True)
    
#     x = Dropout(0.5)(x)
    
    
    x = L.Dense(nb_classes,activation='softmax')(x)

    # Return the model object
    model = M.Model(input=inpt, output=x)
    return model