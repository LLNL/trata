from trata.kosh_sampler import KoshSampler
import kosh
import h5py
import numpy as np
from sklearn import datasets
import os


def test_discrete_koshSampler():

    Nsamples = 15
    Ndims = 4

    X = datasets.load_iris().data
    h5f = h5py.File('iris_data.h5', 'w')
    h5f.create_dataset('dataset_1', data=X)
    h5f.close()

    # Create a new store (erase if exists)
    store = kosh.connect("kosh_test.sql", delete_all_contents=True)
    dataset = store.create("kosh_example1")
    dataset.associate('iris_data.h5', "hdf5")

    # @kosh.numpy_operator
    # def VStack(*data):
    #     return np.vstack(data).transpose()

    # stacked = VStack(dataset["dataset_1"])

    data_subsample = KoshSampler(stacked, method="LatinHyperCubeSampler", num_points=Nsamples)[:]
    assert data_subsample.shape[0] == Nsamples

    data_subsample = KoshSampler(stacked, method="CornerSampler", num_points=Nsamples)[:]
    assert data_subsample.shape[0] == Nsamples

    data_subsample = KoshSampler(stacked, method="SamplePointsSampler")[:]
    assert data_subsample.shape[0] == stacked[:].shape[0]

    data_subsample = KoshSampler(stacked, method="CartesianCrossSampler", num_divisions=Ndims)[:]
    assert data_subsample.shape[0] == Ndims ** 4

    data_subsample = KoshSampler(stacked, method="FractionalFactorialSampler", resolution=Ndims)[:]
    assert data_subsample.shape[0] <= Nsamples

    # Cleanup
    os.remove('iris_data.h5')
    store.close()


def test_continuous_KoshSampler():

    import trata.sampler as sampler
    from sklearn.gaussian_process import GaussianProcessRegressor
    import string
    import random

    rand_n = 7

    test_box = [[-5, 5], [-5, 5]]
    starting_data = sampler.LatinHyperCubeSampler.sample_points(box=test_box,
                                                                num_points=25,
                                                                seed=2018)
    starting_output = starting_data[:, 0] * starting_data[:, 1] + 50.0

    gpm = GaussianProcessRegressor()
    model = gpm.fit(starting_data, starting_output)

    # generate random strings
    res = ''.join(random.choices(string.ascii_uppercase +
                                 string.digits, k=rand_n))
    fileName = 'data_' + str(res) + '.h5'

    res = ''.join(random.choices(string.ascii_uppercase +
                                 string.digits, k=rand_n))
    fileName2 = 'data_' + str(res) + '.h5'

    h5f = h5py.File(fileName, 'w')
    h5f.create_dataset('inputs', data=starting_data.astype(np.float64))
    h5f.close()

    h5f = h5py.File(fileName2, 'w')
    h5f.create_dataset('outputs', data=starting_output.astype(np.float64))
    h5f.close()

    # Create a new store (erase if exists)
    store = kosh.connect("kosh_test.sql", delete_all_contents=True)
    dataset = store.create("kosh_example1")
    dataset.associate([fileName, fileName2], "hdf5")

    num_points = 7
    ndim = starting_data.shape[1]

    new_points = KoshSampler(dataset['inputs'],
                                          method='DeltaSampler',
                                          num_points=7,
                                          model=model,
                                          Y=dataset['outputs'],
                                          num_cand_points=20,
                                          box=test_box)[:]
    assert new_points.shape == (num_points, ndim)

    new_points = KoshSampler(dataset['inputs'],
                                          method='ExpectedImprovementSampler',
                                          num_points=7,
                                          model=model,
                                          Y=dataset['outputs'],
                                          num_cand_points=20,
                                          box=test_box)[:]
    assert new_points.shape == (num_points, ndim)

    new_points = KoshSampler(dataset['inputs'],
                                          method='LearningExpectedImprovementSampler',
                                          num_points=7,
                                          model=model,
                                          Y=dataset['outputs'],
                                          num_cand_points=20,
                                          box=test_box)[:]
    assert new_points.shape == (num_points, ndim)

    # Cleanup
    os.remove(fileName)
    os.remove(fileName2)
    store.close()
