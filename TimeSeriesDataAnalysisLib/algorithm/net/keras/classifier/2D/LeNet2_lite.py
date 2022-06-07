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
    #x = BatchNormalization(axis=3,name=bn_name)(x)
    return x

def get_model(inputCNNshape,nb_classes):
    #min_n_frames = 9
    #inputCNNshape = (48, 48, min_n_frames)
    #nb_classes = 10
    # Constants for the Digits dataset
    """min_n_frames = 6
    inputCNNshape = (90, 120, min_n_frames)
    inputMLPshape = (26 * min_n_frames,)
    nb_classes = 10"""
    # Build the CNN
    inputCNN = L.Input(shape=inputCNNshape)
    #inputNorm = BatchNormalization()(x)
    kernel_size=(3,3)


#     x = Conv2d_BN(inputNorm,nb_filter=32,kernel_size=(7,7),strides=(2,2),padding='valid')
#     pool = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    
    
    conv = L.Conv2D(32, kernel_size, padding='same', activation='relu')(inputCNN)
#     conv = Convolution2D(32, 3, 3, border_mode='same', activation='relu')(conv)
    #conv = BatchNormalization()(conv)
    pool = L.MaxPooling2D((2,2), strides=(2, 2))(conv)
    
#     pool = Dropout(0.25)(pool)

    conv = L.Conv2D(32, kernel_size, padding='same', activation='relu')(pool)
#     conv = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(conv)
    #conv = BatchNormalization()(conv)
    pool = L.MaxPooling2D((2,2), strides=(2, 2))(conv)
#     pool = Dropout(0.25)(pool)

    
    conv = L.Conv2D(64, kernel_size, padding='same', activation='relu')(pool)
#     conv = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(conv)
#     conv = Convolution2D(128, 3, 3, border_mode='same', activation='relu')(conv)
    #     conv = Convolution2D(128, 3, 3, border_mode='same', activation='relu')()BatchNormalization()(conv)
    pool = L.MaxPooling2D((2,2), strides=(2, 2))(conv)
#     pool = Dropout(0.25)(pool)
    
    
    conv = L.Conv2D(128, kernel_size, padding='same', activation='relu')(pool)
#     conv = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(conv)
#     conv = Convolution2D(256, 3, 3, border_mode='same', activation='relu')(conv)
    #conv = BatchNormalization()(conv)
    pool = L.MaxPooling2D((2,2), strides=(2, 2))(conv)
#     pool = Dropout(0.25)(pool)
    
    
    reshape = L.Flatten()(pool)
#     reshape = BatchNormalization()(reshape)
    #reshape = Dropout(0.2)(reshape)
    
    fcCNN = L.Dense(128, activation='relu')(reshape)
#     fcCNN = BatchNormalization()(fcCNN)
    #fcCNN = Dropout(0.2)(fcCNN)


    #fc = Dense(512, activation='relu')(fcCNN)
    #fc = Dropout(0.5)(fc)

    out = L.Dense(nb_classes, activation='softmax')(fcCNN)

    # Return the model object
    model = M.Model(inputs=inputCNN, outputs=out)
    return model