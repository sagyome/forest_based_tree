import numpy as np
import datetime
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from ConjunctionSet import ConjunctionSet
from Branch import Branch
from DataPreperation import *
from ReadDatasetFunctions import *
from NewModelBuilder import *
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

class ExperimentSetting():
    def __init__(self,number_of_branches_threshold,df_names,number_of_estimators_list,fixed_params,
                 num_of_iterations=50,filter_approaches=['probability','number_of_samples','combination']):
        self.num_of_iterations=num_of_iterations
        self.number_of_branches_threshold = number_of_branches_threshold
        self.df_names = df_names
        self.fixed_params = fixed_params
        self.number_of_esitmators = number_of_estimators_list
        self.filter_approaches = filter_approaches
    def run(self):
        self.experiments=[]
        for threshold in self.number_of_branches_threshold:
            for df_name in self.df_names:
                df,x_columns,y_column=get_dataset_by_string(df_name)
                for num_of_estimators in self.number_of_esitmators:
                    for filter_approach in self.filter_approaches:
                        d={}
                        d['number_of_estimators']=num_of_estimators
                        d['probability_threshold']=threshold
                        d['df_name']=df_name
                        d['filter_approach']=filter_approach
                        print(d)
                        self.run_experiment(threshold,df,x_columns,y_column,d)

    def run_experiment(self,branch_probability_threshold,df,x_columns,y_column,hyper_parameters_dict):
        for i in range(self.num_of_iterations):
            print(i)
            num_of_estimators=hyper_parameters_dict['number_of_estimators']
            filter_approach=hyper_parameters_dict['filter_approach']
            result_dict=dict(hyper_parameters_dict)
            result_dict['iteration']=i
            train_x, train_y, test_x, test_y = divide_to_train_test(df, x_columns, y_column)

            #Training random forest
            start_temp=datetime.datetime.now()
            rf = RandomForestClassifier(n_estimators=num_of_estimators,**self.fixed_params)
            rf.fit(train_x, train_y)
            result_dict['random forest training time']=(datetime.datetime.now()-start_temp).microseconds
            self.classes_=rf.classes_

            #Create the conjunction set
            start_temp = datetime.datetime.now()
            cs = ConjunctionSet(x_columns, df, rf,hyper_parameters_dict['threshold'],filter_approach)
            result_dict['conjunction set training time'] = (datetime.datetime.now() - start_temp).microseconds
            result_dict['number of branches per iteration'] = cs.number_of_branches_per_iteration
            result_dict['number_of_branches'] = len(cs.conjunctionSet)

            #Train the new model...
            start_temp = datetime.datetime.now()
            branches_df = cs.get_conjunction_set_df().round(decimals=5)
            for i in range(2):
                branches_df[rf.classes_[i]] = [probas[i] for probas in branches_df['probas']]
            df_dict = {}
            for col in branches_df.columns:
                df_dict[col] = df[col].values
            new_model = Node([True]*len(branches_df))
            new_model.split(df_dict)
            result_dict['new model training time'] = (datetime.datetime.now() - start_temp).microseconds


            #Train a decision tree
            start_temp = datetime.datetime.now()
            decision_tree_model=self.fit_decision_tree_model(train_x, train_y)
            result_dict['decision tree training time'] = (datetime.datetime.now() - start_temp).microseconds
            #record experiment results
            result_dict.update(self.ensemble_measures(test_x,test_y,rf))
            result_dict.update(self.new_model_measures(test_x,test_y,new_model,branches_df))
            result_dict.update(self.decision_tree_measures(test_x,test_y,decision_tree_model))
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
        return result_dict
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
        result_dict['new_model_auc']=self.get_auc(Y,np.array(probas),self.classes_)
        return result_dict
    def ensemble_measures(self,X,Y,rf):
        result_dict={}
        predictions,depths=self.ensemble_prediction(X,rf)
        result_dict['ensemble_average_depth']=np.mean(depths)
        result_dict['ensemble_min_depth'] = np.min(depths)
        result_dict['ensemble max_depth'] = np.max(depths)
        #result_dict['ensemble_predictions_sklearn']=rf.predict_proba(X)
        #result_dict['ensemble_predictions']=predictions
        ensemble_probas=rf.predict_proba(X)
        result_dict['sklearn_vs_our_ensemble_predictions_disagreements']=np.sum([np.sum(i)!=0 for i in ensemble_probas-predictions])
        result_dict['ensemble_accuracy']=np.sum(rf.predict(X)==Y)/len(Y)
        result_dict['ensemble_auc']=self.get_auc(Y,ensemble_probas,rf.classes_)
        if result_dict['sklearn_vs_our_ensemble_predictions_disagreements']>0:
            result_dict['disagreement_instance'] = [str(i)+' | '+ str(j) for i,j in zip(ensemble_probas,predictions) if np.sum(i=j)!=0]
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
        parameters = {'criterion': ['entropy', 'gini'],
                      'max_depth': [3,5,10, 20, 50],
                      'min_samples_leaf': [1, 2, 5, 10]}
        model = DecisionTreeClassifier()
        clfGS = GridSearchCV(model, parameters, cv=3)
        clfGS.fit(train_x, train_y)
        model = clfGS.best_estimator_
        model.f