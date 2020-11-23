import numpy as np
import h5py


def female_male_saving(male_array, female_array, data_path, data_name):
    max_written = np.where(female_array[:, 0] == 0)[0][0]
    result_array_female = np.resize(female_array, (max_written, 768))
    result_array_male = np.resize(male_array, (max_written, 768))

    h5f = h5py.File(data_path + 'data_out.h5', 'w')
    h5f.create_dataset("female_embeddings_{}".format(data_name), data=result_array_female)
    h5f.create_dataset("male_embeddings_{}".format(data_name), data=result_array_male)
    h5f.close()
