from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import pytest
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as gpr

import trata.sampler
import trata.adaptive_sampler


def helper_function(np_input):
    np_input = np_input.astype(float)
    out = np.sin(np_input[:, 0]) + np.cos(np_input[:, 1]) + np.tanh(np_input[:, 2])
    return out.reshape(-1, 1)

ls_test_box = [[0.0, 5.0], [-5.0, 0.0], [-5.0, 5.0]]
np_train_input = trata.sampler.LatinHyperCubeSampler.sample_points(num_points=200,
                                                                                   box=ls_test_box,
                                                                                   seed=2018)
np_train_output = helper_function(np_train_input)
surrogate_model = gpr().fit(np_train_input, np_train_output)
np_candidate_points = trata.sampler.LatinHyperCubeSampler.sample_points(num_points=200,
                                                                                        box=ls_test_box,
                                                                                        seed=2019)

def test_ActiveLearningSampler_valid():
    np_actual_values = trata.adaptive_sampler. \
        ActiveLearningSampler.sample_points(num_points=5,
                                            cand_points=np_candidate_points,
                                            model=surrogate_model)
    np_expected_values = [[ 3.55452106, -4.11850436,  4.85333288],
                          [ 0.05094415, -3.38346499, -1.49442071],
                          [ 0.16384714, -3.29100777, -4.8358629 ],
                          [ 0.28111892, -4.91649867,  0.87429284],
                          [ 4.26925302, -4.71142410,  4.96952397]]
    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

def test_ActiveLearningSampler_invalid():
    # num_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.ActiveLearningSampler.sample_points,
                      cand_points=np_candidate_points,
                      model=surrogate_model)
    # cand_points or box/num_cand_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.ActiveLearningSampler.sample_points,
                      num_points=5,
                      model=surrogate_model)
    # model not given
    pytest.raises(TypeError, trata.adaptive_sampler.ActiveLearningSampler.sample_points,
                      num_points=5,
                      num_cand_points=np_candidate_points)

def test_DeltaSampler_valid():
    np_actual_values = trata.adaptive_sampler. \
        DeltaSampler.sample_points(num_points=5,
                                   cand_points=np_candidate_points,
                                   model=surrogate_model,
                                   X=np_train_input,
                                   Y=np_train_output)
    np_expected_values = [[ 1.99899988, -1.81340482,  0.29633238],
                          [ 0.53101766, -2.32897332, -0.87512164],
                          [ 4.73630199, -2.02933796, -0.17188005],
                          [ 3.76551458, -4.65149758,  4.77579306],
                          [ 3.55452106, -4.11850436,  4.85333288]]
    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

def test_DeltaSampler_invalid():
    # num_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.DeltaSampler.sample_points,
                      cand_points=np_candidate_points,
                      model=surrogate_model,
                      X=np_train_input,
                      Y=np_train_output)
    # cand_points or box/num_cand_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.DeltaSampler.sample_points,
                      num_points=5,
                      model=surrogate_model,
                      X=np_train_input,
                      Y=np_train_output)
    # model not given
    pytest.raises(TypeError, trata.adaptive_sampler.DeltaSampler.sample_points,
                      num_points=5,
                      cand_points=np_candidate_points,
                      X=np_train_input,
                      Y=np_train_output)
    # X not given
    pytest.raises(TypeError, trata.adaptive_sampler.DeltaSampler.sample_points,
                      num_points=5,
                      cand_points=np_candidate_points,
                      model=surrogate_model,
                      Y=np_train_output)
    # Y not given
    pytest.raises(TypeError, trata.adaptive_sampler.DeltaSampler.sample_points,
                      num_points=5,
                      cand_points=np_candidate_points,
                      model=surrogate_model,
                      X=np_train_input)

def test_ExpectedImprovementSampler_valid():
    np_actual_values = trata.adaptive_sampler. \
        ExpectedImprovementSampler.sample_points(num_points=5,
                                                 cand_points=np_candidate_points,
                                                 model=surrogate_model,
                                                 X=np_train_input,
                                                 Y=np_train_output)
    np_expected_values = [[ 4.73630199, -2.02933796, -0.17188005],
                          [ 1.18354594, -4.79710851, -3.98877172],
                          [ 0.53101766, -2.32897332, -0.87512164],
                          [ 3.76551458, -4.65149758,  4.77579306],
                          [ 3.55452106, -4.11850436,  4.85333288]]
    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

def test_ExpectedImprovementSampler_invalid():
    # num_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                      cand_points=np_candidate_points,
                      model=surrogate_model,
                      X=np_train_input,
                      Y=np_train_output)
    # cand_points or box/num_cand_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                      num_points=5,
                      model=surrogate_model,
                      X=np_train_input,
                      Y=np_train_output)
    # model not given
    pytest.raises(TypeError, trata.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                      num_points=5,
                      cand_points=np_candidate_points,
                      X=np_train_input,
                      Y=np_train_output)
    # X not given
    pytest.raises(TypeError, trata.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                      num_points=5,
                      cand_points=np_candidate_points,
                      model=surrogate_model,
                      Y=np_train_output)
    # Y not given
    pytest.raises(TypeError, trata.adaptive_sampler.ExpectedImprovementSampler.sample_points,
                      num_points=5,
                      cand_points=np_candidate_points,
                      model=surrogate_model,
                      X=np_train_input)

def test_LearningExpectedImprovementSampler_valid():
    np_actual_values = trata.adaptive_sampler. \
        LearningExpectedImprovementSampler.sample_points(num_points=5,
                                                         cand_points=np_candidate_points,
                                                         model=surrogate_model,
                                                         X=np_train_input,
                                                         Y=np_train_output)
    np_expected_values = np.array([[1.3852934963238415, -1.96129782751626, -4.139007365070207],
                                   [1.6067869490067022, -3.8305063466119877, -1.7085071793320115],
                                   [1.2908547354629034, -1.380016083614, 4.916344285338901],
                                   [1.3331056988837797, -0.809908644848071, -2.6064294864629867],
                                   [1.8097170706070012, -4.398370404395289, 2.0687736158314527]],
                                    dtype=object)
    np.testing.assert_array_almost_equal(np_actual_values, np_expected_values)

def test_LearningExpectedImprovementSampler_invalid():
    # num_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                      cand_points=np_candidate_points,
                      model=surrogate_model,
                      X=np_train_input,
                      Y=np_train_output)
    # cand_points or box/num_cand_points not given
    pytest.raises(TypeError, trata.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                      num_points=5,
                      model=surrogate_model,
                      X=np_train_input,
                      Y=np_train_output)
    # model not given
    pytest.raises(TypeError, trata.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                      num_points=5,
                      cand_points=np_candidate_points,
                      X=np_train_input,
                      Y=np_train_output)
    # X not given
    pytest.raises(TypeError, trata.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                      num_points=5,
                      cand_points=np_candidate_points,
                      model=surrogate_model,
                      Y=np_train_output)
    # Y not given
    pytest.raises(TypeError, trata.adaptive_sampler.LearningExpectedImprovementSampler.sample_points,
                      num_points=5,
                      cand_points=np_candidate_points,
                      model=surrogate_model,
                      X=np_train_input)
