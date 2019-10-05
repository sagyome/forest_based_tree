import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from Node import *
from sklearn.model_selection import GridSearchCV

class NewModelBuilder():
    def __init__(self,df):
        self.df=df
    def train_new_model(self):
        root=Node(self.df.inex)
        try:
            Y = self.df['label']
        except:
            print(self.df)
        weights = self.df['weight'].values
        self.model = DecisionTreeClassifier(criterion='entropy')
        self.model.fit(X, Y,sample_weight=weights)
    def new_model_processing(self):
        self.tree_features=[]
        self.tree_thresholds=[]
        original_features = [(int(i.split('<')[0]), float(i.split('<')[1])) for i in self.input_feature_names]
        t=self.model.tree_
        for feature, threshold in zip(t.feature, t.threshold):
            if feature < 0:
                self.tree_features.append(feature)
                self.tree_thresholds.append(threshold)
                continue
            self.tree_features.append(original_features[feature][0])
            self.tree_thresholds.append(original_features[feature][1])
    def predict_instance_probas(self,inst):
        indx=0
        depth=0
        while self.tree_features[indx]>=0:
            if inst[self.tree_features[indx]]<=self.tree_thresholds[indx]:
                indx=self.model.tree_.children_right[indx]
            else:
                indx = self.model.tree_.children_left[indx]
            depth+=1
        probas=np.array(self.model.tree_.value[indx][0])
        probas=probas/np.sum(probas)
        return probas,depth
    def predict_instance(self,inst):
        probas,depth=self.predict_instance_probas(self,inst)
        return np.argmax(probas)
