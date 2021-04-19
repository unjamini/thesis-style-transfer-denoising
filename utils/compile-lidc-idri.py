import numpy as np
import os
import h5py
from utils.create_datasets import load_samples


directory = '../../PCsub1-20090909/'


def extract_images(patient):
    # оставляется по 5 изображений для каждого пациента
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
    for patient in os.listdir(directory):
        if patient != '.DS_Store':
            path = os.path.join(directory, patient)
            pat = load_samples(path)
            data = extract_images(pat)
            new_data = np.concatenate((new_data, data))
    save_lib_hdf5(new_data)

