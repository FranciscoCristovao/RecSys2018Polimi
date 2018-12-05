from ParameterTuning.BayesianSearch import BayesianSearch
from ParameterTuning.BayesianSearch import DictionaryKeys
from ParameterTuning.AbstractClassSearch import EvaluatorWrapper
from loader.loader import train_data, test_data, tracks_data, target_data
from utils.auxUtils import buildURMMatrix
from slimRS.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from Base.Evaluation.Evaluator import SequentialEvaluator

hyperparamethers_range_dictionary = {}
hyperparamethers_range_dictionary["topK"] = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
#hyperparamethers_range_dictionary["epochs"] = [1, 5, 10, 20, 30, 50, 70, 90, 110]
hyperparamethers_range_dictionary["sgd_mode"] = ["adagrad", "adam"]
hyperparamethers_range_dictionary["lambda_i"] = [0.0, 1e-3, 1e-6, 1e-9]
hyperparamethers_range_dictionary["lambda_j"] = [0.0, 1e-3, 1e-6, 1e-9]

recommenderDictionary = {DictionaryKeys.CONSTRUCTOR_POSITIONAL_ARGS: [train_data],
                         DictionaryKeys.CONSTRUCTOR_KEYWORD_ARGS: {'URM_validation': buildURMMatrix(test_data)},
                         DictionaryKeys.FIT_POSITIONAL_ARGS: dict(),
                         DictionaryKeys.FIT_KEYWORD_ARGS: {"playlist_ids": target_data['playlist_id'],
                                                           "validation_every_n": 5,
                                                           "stop_on_validation": True,
                                                           "lower_validatons_allowed": 5},
                         DictionaryKeys.FIT_RANGE_KEYWORD_ARGS: hyperparamethers_range_dictionary}

evaluator_validation = SequentialEvaluator(buildURMMatrix(test_data), cutoff_list=[10])

evaluator_validation = EvaluatorWrapper(evaluator_validation)

parameterSearch = BayesianSearch(SLIM_BPR_Cython, evaluator_validation)

n_cases = 2
metric_to_optimize = "MAP"
output_root_path = "output/"

best_parameters = parameterSearch.search(recommenderDictionary,
                                         n_cases = n_cases,
                                         output_root_path = output_root_path,
                                         metric=metric_to_optimize)


