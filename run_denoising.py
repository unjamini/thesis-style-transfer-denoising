import numpy as np
import h5py
import matplotlib.pyplot as plt

from train_cnn_vgg import create_model
from utils.create_datasets import compile_image

model_path = './res/Weights/weights_DRL_edge4d_adam2_perceptual70_mse30_pig.h5'


def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('LDCT'))
        labels = np.array(hf.get('NDCT'))
        ld_patches = np.array(hf.get('LD_patches'))
        return data, labels, ld_patches


def plotting(input, label, result):
    fig, axs = plt.subplots(1, 3)
    axs[1].imshow(result, cmap='gray')
    axs[1].set_title('Результат')
    axs[0].imshow(input, cmap='gray')
    axs[0].set_title('LDCT')
    axs[2].imshow(label, cmap='gray')
    axs[2].set_title('NDCT')
    plt.show()



model_edge_p_mse = create_model()
model_edge_p_mse.load_weights(model_path)

data_test, labels_test, ld_patches_test = read_hdf5('./test_data.hdf5')
data_test = (data_test[:, :, :, None] / 4095).astype(np.float32)
labels_test = (labels_test[:, :, :, None] / 4095).astype(np.float32)
ld_patches_test = (ld_patches_test[:, :, :, None] / 4095).astype(np.float32)
[labels_pred_test, labels_pred_test] = model_edge_p_mse.predict(ld_patches_test[:, :, :, :], batch_size=8, verbose=1)

result = compile_image(labels_pred_test[:256, :, :, 0])
result1 = compile_image(np.sum(labels_pred_test[:256, ...], axis=-1))
