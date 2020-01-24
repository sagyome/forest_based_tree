import numpy as np
import datetime
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from ConjunctionSet import ConjunctionSet
from Branch import Branch
from DataPreperation import *
import os
from ReadDatasetFunctions import *
from NewModelBuilder import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score
import pickle
from CMM import *



class ExperimentSetting():
    def __init__(self,number_of_branches_threshold,df_names,number_of_estimators,fixed_params,
                 num_of_iterations=30):
        self.num_of_iterations=num_of_iterations
        self.number_of_branches_threshold = number_of_branches_threshold
        self.df_names = df_names
        self.fixed_params = fixed_params
        self.number_of_estimators = number_of_estimators
    def run(self):
        self.experiments = []
        for threshold in self.number_of_branches_threshold:
            for df_name in self.df_names:
                df, x_columns, y_column, feature_types = get_dataset_by_string(df_name)
                d = {}
                d['max_number_of_branches'] = threshold
                d['df_name'] = df_name
                d['number_of_estimators'] = self.number_of_estimators
                print(d)
                self.run_experiment(threshold, df, x_columns, y_column, feature_types, d)
    def run_experiment(self,branch_probability_threshold,df,x_columns,y_column,feature_types,hyper_parameters_dict):
        for i in range(self.num_of_iterations):
            print(i)
            np.random.seed(i)
            num_of_estimators=hyper_parameters_dict['number_of_estimators']
            result_dict=dict(hyper_parameters_dict)
            result_dict['iteration']=i
            output_path = 'pickles_200trees/' + str(hyper_parameters_dict['df_name']) + '_' + str(result_dict['iteration'])
            if os.path.isfile(output_path):
                continue

            trainAndValidation_x, trainAndValidation_y, test_x, test_y = divide_to_train_test(df, x_columns, y_column)
            train_x = trainAndValidation_x[:int(len(trainAndValidation_x) * 0.8)]
            train_y = trainAndValidation_y[:int(len(trainAndValidation_x) * 0.8)]
            validation_x = trainAndValidation_x[int(len(trainAndValidation_x) * 0.8):]
            validation_y = trainAndValidation_y[int(len(trainAndValidation_x) * 0.8):]



            #Train random forest
            start_temp=datetime.datetime.now()
            rf = RandomForestClassifier(n_estimators=num_of_estimators,max_depth=5,min_samples_leaf=max(1,int(0.02*len(train_x))), **self.fixed_params)
            #rf = ExtraTreesClassifier(n_estimators=num_of_estimators, max_depth=3,min_samples_leaf=max(1, int(0.02 * len(train_x))), **self.fixed_params)
            rf.fit(trainAndValidation_x, trainAndValidation_y)
            result_dict['random forest training time']=(datetime.datetime.now()-start_temp).total_seconds()
            self.classes_=rf.classes_

            #Create the conjunction set
            start_temp = datetime.datetime.now()
            cs = ConjunctionSet(x_columns, trainAndValidation_x,trainAndValidation_x,trainAndValidation_y, rf, feature_types,
                                hyper_parameters_dict['max_number_of_branches'])
            result_dict['conjunction set training time'] = (datetime.datetime.now() - start_temp).total_seconds()
            result_dict['number of branches per iteration'] = cs.number_of_branches_per_iteration
            result_dict['number_of_branches'] = len(cs.conjunctionSet)

            #Train the new model
            start_temp = datetime.datetime.now()
            branches_df = cs.get_conjunction_set_df().round(decimals=5)
            result_dict['number_of_features_for_new_model'] = len(branches_df.columns)
            for i in range(2):
                branches_df[rf.classes_[i]] = [probas[i] for probas in branches_df['probas']]
            df_dict = {}
            for col in branches_df.columns:
                df_dict[col] = branches_df[col].values
            new_model = Node([True]*len(branches_df))
            new_model.split(df_dict)
            result_dict['new model training time'] = (datetime.datetime.now() - start_temp).total_seconds()

            #Train a decision tree
            start_temp = datetime.datetime.now()
            decision_tree_model=self.fit_decision_tree_model(trainAndValidation_x, trainAndValidation_y)
            result_dict['decision tree training time'] = (datetime.datetime.now() - start_temp).total_seconds()

            #Train CMM tree
            start_temp = datetime.datetime.now()
            cmm_data = pd.DataFrame(trainAndValidation_x,columns=x_columns)
            cmm_data[y_column] = trainAndValidation_y
            cmm_dt = self.fit_cmm_tree(cmm_data,x_columns,y_column,rf)
            result_dict['cmm tree training time'] = (datetime.datetime.now() - start_temp).total_seconds()

            #record experiment results
            result_dict.update(self.ensemble_measures(test_x,test_y,rf))
            result_dict.update(self.new_model_measures(test_x,test_y,new_model,branches_df))
            result_dict.update(self.decision_tree_measures(test_x,test_y,decision_tree_model))
            result_dict.update(self.cmm_tree_measures(test_x,test_y,cmm_dt))

            with open(output_path,'wb') as fp:
                pickle.dump(result_dict, fp)
            self.experiments.append(result_dict)
    def decision_tree_measures(self,X,Y,dt_model):
        result_dict={}
        probas=[]
        depths=[]
        for inst in X:
            pred,dept=self.tree_depth_and_prediction(inst,dt_model.tree_)
            probas.append(pred)
            depths.append(dept)
        predictions=dt_model.predict(X)
        result_dict['decision_tree_average_depth'] = np.mean(depths)
        result_dict['decision_tree_min_depth'] = np.min(depths)
        result_dict['decision_tree_max_depth'] = np.max(depths)
        result_dict['decision_tree_accuracy'] = np.sum(predictions == Y) / len(Y)
        result_dict['decision_tree_auc'] = self.get_auc(Y, np.array(probas),dt_model.classes_)
        result_dict['decision_tree_kappa'] = cohen_kappa_score(Y,predictions)
        return result_dict
    def cmm_tree_measures(self,X,Y,dt_model):
        return {k.replace('decision_tree','cmm_tree'):v for k,v in self.decision_tree_measures(X,Y,dt_model).items()}
    def new_model_measures(self,X,Y,new_model,branches_df):
        result_dict={}
        probas,depths=[],[]
        for inst in X:
            prob,depth=new_model.predict_probas_and_depth(inst,branches_df)
            probas.append(prob)
            depths.append(depth)
        predictions=[self.classes_[i] for i in np.array([np.argmax(prob) for prob in probas])]
        result_dict['new_model_average_depth']=np.mean(depths)
        result_dict['new_model_min_depth'] = np.min(depths)
        result_dict['new_model_max_depth'] = np.max(depths)
        result_dict['new_model_accuracy'] = np.sum(predictions==Y) / len(Y)
        result_dict['new_model_auc'] = self.get_auc(Y,np.array(probas),self.classes_)
        result_dict['new_model_kappa'] = cohen_kappa_score(Y,predictions)
        result_dict['new_model_number_of_nodes'] = new_model.number_of_children()
        result_dict['new_model_probas'] = probas

        return result_dict
    def ensemble_measures(self,X,Y,rf):
        result_dict={}
        predictions,depths=self.ensemble_prediction(X,rf)
        result_dict['ensemble_average_depth']=np.mean(depths)
        result_dict['ensemble_min_depth'] = np.min(depths)
        result_dict['ensemble max_depth'] = np.max(depths)
        ensemble_probas=rf.predict_proba(X)
        result_dict['ensemble_accuracy'] = np.sum(rf.predict(X)==Y)/len(Y)
        result_dict['ensemble_auc'] = self.get_auc(Y,ensemble_probas,rf.classes_)
        result_dict['ensemble_kappa'] = cohen_kappa_score(Y,rf.predict(X))
        result_dict['ensemble_probas'] = ensemble_probas
        return result_dict
    def ensemble_prediction(self,X, rf):
        predictions = []
        depths = []
        for inst in X:
            pred = []
            depth = 0
            for base_model in rf.estimators_:
                res = self.tree_depth_and_prediction(inst, base_model.tree_)
                pred.append(res[0])
                depth += res[1]
            predictions.append(np.array(pred).mean(axis=0))
            depths.append(depth)
        return predictions, depths
    def tree_depth_and_prediction(self,inst, t):
        indx = 0
        depth = 0
        epsilon = 0.0000001
        # epsilon: thresholds may be shifted by a very small floating points. For example: x1 <= 2.6 may become x1 <= 2.5999999
        # and then x1 = 2.6 won't be captured
        while t.feature[indx] >= 0:
            if inst[t.feature[indx]] <= t.threshold[indx] + epsilon:
                indx = t.children_left[indx]
            else:
                indx = t.children_right[indx]
            depth += 1
        return np.array([i / np.sum(t.value[indx][0]) for i in t.value[indx][0]]), depth
    def get_auc(self,Y,y_score,classes):
        y_test_binarize=np.array([[1 if i ==c else 0 for c in classes] for i in Y])
        fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
        return auc(fpr, tpr)

    def fit_decision_tree_model(self,train_x, train_y):
        """
        This function gets train data and conducts a gridsearch for the best decision tree
        out of several options. It returns the fitted tree
        """
        parameters = {'max_depth': [3, 10, 20],
                      'criterion':['gini','entropy'],
                      'min_samples_leaf': [1, 2, 10]}
        model = DecisionTreeClassifier()
        clfGS = GridSearchCV(model, parameters, cv=10)
        clfGS.fit(train_x, train_y)
        model = clfGS.best_estimator_
        return model
    def fit_cmm_tree(self, df,x_columns,y_column, rf):
        synthetic_data = get_synthetic_data(df)
        cmm_dt = train_dt_for_synthetic_data(synthetic_data,x_columns,y_column,rf)
        return cmm_dt

