import numpy as np
import os
import h5py
from PIL import Image

directory = ''


def extract_images(patient):
    # Для каждого пациента используется 5 равномерно выбранных слайса
    images_num, h, w = patient.shape
    slice_idxs = [int(np.floor(images_num / 4)),
                  int(np.floor(3 * images_num / 8)),
                  int(np.floor(images_num / 2)),
                  int(np.floor(5 * images_num / 8)),
                  int(np.floor(3 * images_num / 4))]
    out = patient[slice_idxs, :, :]
    return out


def save_lib_hdf5(images, filename="PC1_10.hdf5"):
    with h5py.File(filename, "w") as f:
        f.create_dataset("LDCT", data=images)


if __name__ == '__main__':
    new_data = np.empty((0, 512, 512))

    # использую каждое 25е изображение (так как датасет очень обширный)
    for patient in os.listdir(directory)[::25]:
        if patient != '.DS_Store' and patient.endswith('.png'):
            path = os.path.join(directory, patient)
            im2 = Image.open(path).convert('L')
            im2 = np.array(im2)

            if im2.shape == (512, 512):
                new_data = np.concatenate((new_data, im2[np.newaxis, ...]))
    print(new_data.shape)
    save_lib_hdf5(new_data, 'COVID_dset.hdf5')

