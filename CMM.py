import os, sys
sys.path.append(os.getcwd() + '/DataSynthesizer/DataSynthesizer')
from DataDescriber import DataDescriber
from DataGenerator import DataGenerator
import pandas as pd
from ModelInspector import ModelInspector
from lib.utils import read_json_file, display_bayesian_network
from ReadDatasetFunctions import *
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

mode = 'correlated_attribute_mode'
description_file = 'description.json'
synthetic_data = 'sythetic_data.csv'

def get_synthetic_data(df):
    # An attribute is categorical if its domain size is less than this threshold.
    # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
    threshold_value = 1

    # specify categorical attributes
    categorical_attributes = {}

    # specify which attributes are candidate keys of input dataset.
    candidate_keys = {}

    # A parameter in Differential Privacy. It roughly means that removing a row in the input dataset will not
    # change the probability of getting the same output more than a multiplicative difference of exp(epsilon).
    # Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
    epsilon = 0.1

    # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
    degree_of_bayesian_network = 2

    num_tuples_to_generate = len(df)*20  # Here 32561 is the same as input dataset, but it can be set to another number.
    input_data = 'temp_train.csv'
    df.to_csv(input_data,index=False)
    describer = DataDescriber(category_threshold=threshold_value)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                            epsilon=epsilon,
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=categorical_attributes,
                                                            attribute_to_is_candidate_key=candidate_keys)
    describer.save_dataset_description_to_file(description_file)
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)
    synth_data = pd.read_csv(synthetic_data)
    return synth_data

def train_dt_for_synthetic_data(synth_data,x_columns,y_column,rf):
    synth_data[y_column] = rf.predict(synth_data[x_columns].values)
    parameters = {'criterion': ['entropy', 'gini'],
                  'max_depth': [3, 5, 10, 20],
                  'min_samples_leaf': [1, 2, 5, 10]}
    dt = DecisionTreeClassifier()
    clfGS = GridSearchCV(dt, parameters, cv=10)
    clfGS.fit(synth_data[x_columns].values, synth_data[y_column])
    dt = clfGS.best_estimator_
    return dt

