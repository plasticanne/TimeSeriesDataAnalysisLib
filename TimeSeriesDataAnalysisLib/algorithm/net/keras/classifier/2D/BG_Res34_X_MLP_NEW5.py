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
 
    x = L.Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = L.BatchNormalization(axis=3,name=bn_name)(x)
    return x
 
    
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=(1,1))
#         shortcut = ZeroPadding2D(padding=(2, 2))(inpt)
        x = L.add([x,shortcut])
        return x
    else:
        x = L.add([x,inpt])
        return x

def get_model(inputCNNshape,nb_classes,inputMLPshape):
    # Constants for the AVletters dataset
    #min_n_frames = 9
    #inputCNNshape = (48, 48, min_n_frames)
    #inputMLPshape = (3 * 1,)
    #nb_classes = 10
    # Constants for the Digits dataset
    inpt = L.Input(shape=inputCNNshape)
    if inputCNNshape[0]==48:
        a1=4
        a2=3
        a3=8
        output_shape=(None, 6, 6, 8)
    elif inputCNNshape[0]==72:
        a1=5
        a2=5
        a3=64
        output_shape=(None, 9, 9, 64)
    else:
        raise tf.errors.InvalidArgumentError

#     x = ZeroPadding2D((3,3))(inpt)
    x = Conv2d_BN(inpt,nb_filter=32,kernel_size=(3,3),strides=(2,2))
    

# MLP stage 1
    # Build the MLP
    inputMLP = L.Input(shape=inputMLPshape)
    fcMLP = L.Dense(12, activation='relu')(inputMLP)
#     fcMLP = BatchNormalization()(fcMLP)
    

    fcMLP2 = L.Dense(12, activation='relu')(inputMLP)
#     fcMLP2 = BatchNormalization()(fcMLP2)
    
    
#     fcMLP3 = Dense(8, activation='relu')(inputMLP)
#     fcMLP3 = BatchNormalization()(fcMLP3)
    
    
    
# CNN stage 1
    #(48,48,16)
#     x = Conv_Block(x,nb_filter=16,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=16,kernel_size=(3,3))
    
    







    
#     fcMLP = Dropout(0.2)(fcMLP)




#     fcMLP = Dense(16, activation='relu')(fcMLP)
#     fcMLP = BatchNormalization()(fcMLP)
#     fcMLP = Dropout(0.2)(fcMLP)
    
    
# CNN stage 2
    #(24,24,32)
#     x = Conv_Block(x,nb_filter=32,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=32,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=32,kernel_size=(3,3))
#     x = Dropout(0.2)(x, training=True)






#     x12_k_st = Dense(5*5, activation='relu')(inputMLP)
# #     x12_k = PReLU()(x12_k)
#     x12_k_st = BatchNormalization()(x12_k_st)
# #     x12_k = Dropout(0.2)(x12_k, training=True)
#     x12_k_st = Reshape((5,5,1))(x12_k_st)
#     x12_k_st = Deconvolution2D(32, 8, 8, output_shape=(None, 12, 12, 32), border_mode='valid')(x12_k_st)
#     x12_k_st = PReLU(name='x1_to_cnn1k')(x12_k_st)
    
    
    
#     fcMLP2 = Dropout(0.2)(fcMLP2, training=True)
#     x12_k = Dense(5*5, activation='relu')(fcMLP3)
# #     x12_k = PReLU()(x12_k)
#     x12_k = BatchNormalization()(x12_k)
# #     x12_k = Dropout(0.2)(x12_k, training=True)
#     x12_k = Reshape((5,5,1))(x12_k)
#     x12_k = Deconvolution2D(4, 8, 8, output_shape=(None, 12, 12, 4), border_mode='valid')(x12_k)
#     x12_k = PReLU(name='x1_to_cnn2k')(x12_k)
    
# #     x12map = add([x12_k_st, x])
#     x12map = concatenate([x12_k, x] , axis=3)
#     x12map = BatchNormalization()(x12map)
#     x12map = Dropout(0.2)(x12map, training=True)





# CNN stage 3
    #(12,12,64)
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))




#     x12_k_st = Dense(5*5, activation='relu')(inputMLP)
# #     x12_k = PReLU()(x12_k)
#     x12_k_st = BatchNormalization()(x12_k_st)
# #     x12_k = Dropout(0.2)(x12_k, training=True)
#     x12_k_st = Reshape((5,5,1))(x12_k_st)
#     x12_k_st = Deconvolution2D(32, 2, 2, output_shape=(None, 6, 6, 32), border_mode='valid')(x12_k_st)
#     x12_k_st = PReLU(name='x1_to_cnn1')(x12_k_st)
    
    
    
#     fcMLP2 = Dropout(0.2)(fcMLP2, training=True)
    x12_k = L.Dense(a1*a1, activation='relu')(fcMLP2)
#     x12_k = PReLU()(x12_k)
#     x12_k = BatchNormalization()(x12_k)
#     x12_k = Dropout(0.2)(x12_k, training=True)
    x12_k = L.Reshape((a1,a1,1))(x12_k)
    x12_k = L.Deconvolution2D(a3, a2, a2, output_shape=output_shape, border_mode='valid')(x12_k)
    x12_k = L.PReLU(name='x1_to_cnn2')(x12_k)
    
#     x12map = add([x12_k_st, x])
    x12map = L.concatenate([x12_k, x] , axis=3)
#     x12map = BatchNormalization()(x12map)
#     x12map = Dropout(0.2)(x12map, training=True)
    

    
# CNN stage 4
    #(7,7,512)
    x = Conv_Block(x12map,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
#     x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
#     x = Dropout(0.2)(x, training=True)



    x = L.AveragePooling2D(pool_size=(3,3))(x)
    x = L.Flatten()(x)
    
    x = L.Dropout(0.2)(x, training=True)
    

    x = L.concatenate([fcMLP, x])
#     x = BatchNormalization()(x)
    
#     x = Dropout(0.2)(x, training=True)
    
    x = L.Dense(256, activation='relu')(x)
    x = L.Dropout(0.2)(x, training=True)
#     x = BatchNormalization()(x)
    
    x = L.Dense(128, activation='relu')(x)
    x = L.Dropout(0.2)(x, training=True)
#     x = BatchNormalization()(x)
    
#     x = Dropout(0.2)(x, training=True)
#     x = Dense(40, activation='relu')(x)
#     x = Dropout(0.2)(x, training=True)
    
    x = L.Dense(nb_classes,activation='softmax')(x)

    # Return the model object
    model = M.Model(input=[inpt, inputMLP], output=x)
    return model