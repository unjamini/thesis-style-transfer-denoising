import pandas as pd
from keras.layers import concatenate, Activation, Lambda
from keras.layers import Conv2D, Input
from keras.optimizers import Adam
from keras import losses
import h5py
from keras.layers import BatchNormalization as BN
from utils.sobel import *
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
import tensorflow as tf
tf.config.run_functions_eagerly(True)


batch_size = 32
image_shape = (None, None, 3)


def PSNRLoss(y_true, y_pred):
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)


def save_history(history_dict, csv_filename):
    hist_df = pd.DataFrame(history_dict.history)

    with open(csv_filename, mode='w') as f:
        hist_df.to_csv(f)


def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('LDCT'))
        labels = np.array(hf.get('NDCT'))
        return data, labels


def sobel(x):
    dims = tf.shape(x)  # return tf.squeeze( tf.image.sobel_edges(x))
    return tf.reshape(sobel(x), [dims[0], dims[1], dims[2], 4])


def sobel_shape(input_shape):
    dims = [input_shape[0], input_shape[1], input_shape[2], 4]
    output_shape = tuple(dims)
    return output_shape


def create_model():
    inputs = Input(shape=(None, None, 1))
    edges = Lambda(sobel, output_shape=sobel_shape, name='sobel-edge')(inputs)

    input_edge = concatenate([inputs, edges], axis=3)

    conv1 = Conv2D(64, (5, 5), activation='relu', padding='same')(input_edge)

    conv2 = Conv2D(64, (3, 3), padding='same', dilation_rate=(2, 2))(conv1)
    conv2 = BN()(conv2)
    conv2 = Activation('relu')(conv2)

    conv3 = Conv2D(64, (3, 3), padding='same', dilation_rate=(3, 3))(conv2)
    conv3 = BN()(conv3)
    conv3 = Activation('relu')(conv3)

    conv4 = Conv2D(64, (3, 3), padding='same', dilation_rate=(4, 4))(conv3)
    conv4 = BN()(conv4)
    conv4 = Activation('relu')(conv4)

    conv5 = Conv2D(64, (3, 3), padding='same', dilation_rate=(3, 3))(conv4)
    conv5 = BN()(conv5)
    conv5 = Activation('relu')(conv5)

    conv6 = concatenate([conv5, conv2], axis=3)
    conv6 = Conv2D(64, (3, 3), padding='same', dilation_rate=(2, 2))(conv6)
    conv6 = BN()(conv6)
    conv6 = Activation('relu')(conv6)

    conv7 = concatenate([conv6, conv1], axis=3)
    conv7 = Conv2D(1, (3, 3), padding='same')(conv7)

    conv8 = concatenate([conv7, input_edge], axis=3)
    outputs = Conv2D(3, (3, 3), padding='same')(conv8)

    return Model(inputs=[inputs], outputs=[outputs, outputs])


def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    selectedLayers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']
    selectedOutputs = [vgg.get_layer(i).output for i in selectedLayers]
    loss_model = Model(inputs=vgg.input, outputs=selectedOutputs)
    loss_model.trainable = False
    mse = K.variable(value=0)
    for i in range(0, 3):
        mse = mse + K.mean(K.square(loss_model(y_true)[i] - loss_model(y_pred)[i]))
    return mse


def train_cnn_vgg(dataset_path):
    data, labels = read_hdf5(dataset_path)
    data = (data[:, :, :, None] / 4095).astype(np.float32)
    labels = (labels[:, :, :, None] / 4095).astype(np.float32)
    labels_3 = np.concatenate((labels, labels, labels), axis=-1)

    model_edge_p_mse = create_model()
    loss = [perceptual_loss, losses.mean_squared_error]
    loss_weights = [70, 30]

    tb_callback1 = tf.keras.callbacks.TensorBoard(
        log_dir='./logs1', histogram_freq=1)

    tb_callback2 = tf.keras.callbacks.TensorBoard(
        log_dir='./logs2', histogram_freq=1)

    ADAM = Adam(lr=0.001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model_edge_p_mse.compile(optimizer=ADAM, loss=loss, loss_weights=loss_weights, metrics=[PSNRLoss, 'loss'])
    hist_adam1 = model_edge_p_mse.fit(x=data, y=[labels_3, labels_3], batch_size=batch_size, epochs=20
                                     , validation_split=0, verbose=1, shuffle=True, callbacks=[tb_callback1])
    save_history(hist_adam1, 'hist_1.csv')
    model_edge_p_mse.save_weights('Weights/weights_DRL_edge4d_adam1_perceptual70_mse30_pig.h5')

    ADAM = Adam(lr=0.0001, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model_edge_p_mse.compile(optimizer=ADAM, loss=loss, loss_weights=loss_weights, metrics=[PSNRLoss, 'loss'])
    hist_adam2 = model_edge_p_mse.fit(x=data, y=[labels_3, labels_3], batch_size=batch_size, epochs=20
                                     , validation_split=0, verbose=1, shuffle=True, callbacks=[tb_callback2])
    save_history(hist_adam2, 'hist_2.csv')
    model_edge_p_mse.save_weights('Weights/weights_DRL_edge4d_adam2_perceptual70_mse30_pig.h5')


train_cnn_vgg('train_data.hdf5')
