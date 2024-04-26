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
    np_expected_values = [[2.647625702856798, -3.688256076131667, 1.9770490017748559],
                          [0.910786401471699, -4.584000218711989, -2.1662730274767754],
                          [3.765514576073261, -4.651497579471858, 4.775793056412519],
                          [1.5281075159512072, -0.4872863970040626, 2.315967227324471],
                          [4.9888063946430306, -4.056622552061053, 2.6788616344376965]]
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
