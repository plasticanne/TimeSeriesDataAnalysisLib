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


def get_model(inputCNNshape,nb_classes):
    # Constants for the AVletters dataset
    #channel = 9
    #inputCNNshape = (48, 48, channel)
    #nb_classes = 10
    # Constants for the Digits dataset
    """channel  = 6
    inputCNNshape = (90, 120, channel )
    inputMLPshape = (26 * channel ,)
    nb_classes = 10"""
    # Build the CNN
    inputCNN = L.Input(shape=inputCNNshape)
#     inputNorm = BatchNormalization()(inputCNN)
    kernel_size=(3,3)


#     x = Conv2d_BN(inputNorm,nb_filter=32,kernel_size=(7,7),strides=(2,2),padding='valid')
#     pool = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    
    
    x = Conv2d_BN(inputCNN,nb_filter=32,kernel_size=kernel_size)
    x = Conv2d_BN(x,nb_filter=32,kernel_size=kernel_size)
    x = L.MaxPooling2D((2,2), strides=(2, 2))(x)
    
#     pool = Dropout(0.25)(pool)

    x = Conv2d_BN(x,nb_filter=32,kernel_size=kernel_size)
    x = Conv2d_BN(x,nb_filter=32,kernel_size=kernel_size)
    x = L.MaxPooling2D((2,2), strides=(2, 2))(x)
#     pool = Dropout(0.25)(pool)

    x = Conv2d_BN(x,nb_filter=64,kernel_size=kernel_size)
    x = Conv2d_BN(x,nb_filter=64,kernel_size=kernel_size)
    x = L.MaxPooling2D((2,2), strides=(2, 2))(x)
#     pool = Dropout(0.25)(pool)
    
    
    x = Conv2d_BN(x,nb_filter=128,kernel_size=kernel_size)
    x = Conv2d_BN(x,nb_filter=128,kernel_size=kernel_size)
    x = L.MaxPooling2D((2,2), strides=(2, 2))(x)
#     pool = Dropout(0.25)(pool)
    
    
    reshape = L.Flatten()(x)
#     reshape = BatchNormalization()(reshape)
#     reshape = Dropout(0.2)(reshape)
    
    fcCNN = L.Dense(128, activation='relu')(reshape)
#     fcCNN = BatchNormalization()(fcCNN)
#     fcCNN = Dropout(0.2)(fcCNN)


    #fc = Dense(512, activation='relu')(fcCNN)
    #fc = Dropout(0.5)(fc)

    out = L.Dense(nb_classes, activation='softmax')(fcCNN)

    # Return the model object
    model = M.Model(input=inputCNN, output=out)
    return model