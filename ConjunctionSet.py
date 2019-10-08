from Branch import Branch
import numpy as np
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import entropy
from pruningFunctions import *

class ConjunctionSet():
    def __init__(self,feature_names,original_data, pruning_x, pruning_y,
                 model, feature_types, amount_of_branches_threshold,filter_approach='probability', exclusion_starting_point=10,
                 minimal_forest_size=10,exclusion_threshold=0.8):
        self.amount_of_branches_threshold = amount_of_branches_threshold
        self.model = model
        self.feature_names = feature_names
        self.exclusion_threshold=exclusion_threshold
        self.label_names = model.classes_
        self.relevant_indexes = reduce_error_pruning(self.model,pruning_x,pruning_y,minimal_forest_size)
        self.feature_types = feature_types
        self.filter_approach = filter_approach
        self.exclusion_starting_point = exclusion_starting_point
        self.set_ecdf(original_data)
        self.get_ranges(original_data)
        self.generateBranches()
        #self.get_associative_leaves(np.concatenate((original_data,pruning_x)))
        self.number_of_branches_per_iteration = []
        self.buildConjunctionSet()

    def generateBranches(self):
        trees=[estimator.tree_ for estimator in self.model.estimators_]
        self.branches_lists=[self.get_tree_branches(tree_,i) for i,tree_ in enumerate(trees) if i in self.relevant_indexes]
        for list_indx,branch_list in enumerate(self.branches_lists):
            for leaf_index,branch in enumerate(branch_list):
                branch.leaves_indexes=[str(list_indx)+'_'+str(leaf_index)]
    def get_tree_branches(self,tree_,tree_index):
        leaf_indexes = [i for i in range(tree_.node_count) if tree_.children_left[i] == -1 and tree_.children_right[i] == -1]
        branches=[self.get_branch_from_leaf_index(tree_,leaf_index) for leaf_index in leaf_indexes]
        return branches
    def get_branch_from_leaf_index(self,tree_,leaf_index):
        sum_of_probas=np.sum(tree_.value[leaf_index][0])
        label_probas=[i/sum_of_probas for i in tree_.value[leaf_index][0]]
        new_branch=Branch(self.feature_names,self.feature_types,self.label_names,label_probas=label_probas,
                          number_of_samples=tree_.n_node_samples[leaf_index])#initialize branch
        node_id=leaf_index
        while node_id: #iterate over all nodes in branch
            ancesor_index=np.where(tree_.children_left==node_id)[0] #assuming left is the default for efficiency purposes
            bound='upper'
            if len(ancesor_index)==0:
                bound='lower'
                ancesor_index = np.where(tree_.children_right == node_id)[0]
            new_branch.addCondition(tree_.feature[ancesor_index[0]], tree_.threshold[ancesor_index[0]], bound)
            node_id=ancesor_index[0]
        return new_branch
    def buildConjunctionSet(self):
        conjunctionSet=self.branches_lists[0]
        excluded_branches=[]
        for i,branch_list in enumerate(self.branches_lists[1:]):
            print('Iteration '+str(i+1)+": "+str(len(conjunctionSet))+" conjunctions")
            conjunctionSet=self.merge_branch_with_conjunctionSet(branch_list,conjunctionSet)
            #print('i='+str(i))
            if i >= self.exclusion_starting_point:
                conjunctionSet,this_iteration_exclusions=self.exclude_branches_from_cs(conjunctionSet,self.exclusion_threshold)
                excluded_branches.extend(this_iteration_exclusions)
                #print('Number of exclusions: '+str(len(excluded_branches)))
                #print('Number of remained: '+str(len(conjunctionSet)))
        self.conjunctionSet=excluded_branches+conjunctionSet
    def exclude_branches_from_cs(self,cs,threshold):
        filtered_cs=[]
        excludable_brancehs=[]
        for branch in cs:
            if branch.is_excludable_branch(threshold):
                excludable_brancehs.append(branch)
            else:
                filtered_cs.append(branch)
        return filtered_cs,excludable_brancehs
    def filter_conjunction_set(self,cs):
        if len(cs) <= self.amount_of_branches_threshold:
            return cs
        if self.filter_approach=='association':
            cs = [b for b in cs if b.is_valid_association(self.association_leaves)]
            return cs
        if self.filter_approach=='probability':
            branches_metrics=[b.calculate_branch_probability_by_ecdf(self.ecdf_dict) for b in cs]
        elif self.filter_approach=='number_of_samples':
            branches_metrics = [b.number_of_samples for b in cs]
        elif self.filter_approach=='probability_entropy':
            branches_metrics = [b.calculate_branch_probability_by_ecdf(self.ecdf_dict)*(1-entropy(b.label_probas)) for b in cs]
        elif self.filter_approach=='entropy':
            branches_metrics = [-entropy(b.label_probas) for b in cs]
        elif self.filter_approach=='range':
            branches_metrics = [b.calculate_branch_probability_by_range(self.ranges) for b in cs]
        elif self.filter_approach=='association_probability':
            branches_metrics = [b.is_valid_association(self.association_leaves)*b.calculate_branch_probability_by_ecdf(self.ecdf_dict) for b in cs]
        threshold=sorted(branches_metrics,reverse=True)[self.amount_of_branches_threshold-1]
        return [b for b,metric in zip(cs,branches_metrics) if metric >= threshold][:self.amount_of_branches_threshold]

    def merge_branch_with_conjunctionSet(self,branch_list,conjunctionSet):
        new_conjunction_set=[]
        for b1 in conjunctionSet:
            new_conjunction_set.extend([b1.mergeBranch(b2) for b2 in branch_list if b1.contradictBranch(b2)==False])
        #print('number of branches before filterring: '+str(len(new_conjunction_set)))
        new_conjunction_set=self.filter_conjunction_set(new_conjunction_set)
        #print('number of branches after filterring: ' + str(len(new_conjunction_set)))
        self.number_of_branches_per_iteration.append(len(new_conjunction_set))
        return new_conjunction_set
    def get_conjunction_set_df(self):
        return pd.DataFrame([b.get_branch_dict(self.ecdf_dict) for b in self.conjunctionSet])
    def predict(self,X):
        predictions=[]
        for inst in X:
            for conjunction in self.conjunctionSet:
                if conjunction.containsInstance(inst):
                   predictions.append(self.label_names[conjunction.getLabel()])
        return predictions
    def get_instance_branch(self,inst):
        for conjunction in self.conjunctionSet:
            if conjunction.containsInstance(inst):
                return conjunction
    def set_ecdf(self,data):
        self.ecdf_dict={i:ECDF(data.transpose().T[i])for i in range(len(self.feature_names))}
    def group_by_label_probas(self,conjunctionSet):
        probas_hashes={}
        for i,b in enumerate(conjunctionSet):
            probas_hash = hash(tuple(b.label_probas))
            if probas_hash not in probas_hashes:
                probas_hashes[probas_hash]=[]
            probas_hashes[probas_hash].append(i)
        return probas_hashes
    def get_ranges(self,original_data):
        self.ranges = [max(v)-min(v) for v in original_data.transpose()]
    def get_associative_leaves(self,X):
        association_dict ={}
        for indx,inst in enumerate(X):
            association_dict[indx]=[]
            for tree_indx,tree_ in enumerate(self.branches_lists):
                for leaf_indx,leaf in enumerate(tree_):
                    if leaf.containsInstance(inst):
                        association_dict[indx].append(str(tree_indx)+'_'+str(leaf_indx))
        association_dict
        self.association_leaves={}
        for tree_indx1,tree_1 in enumerate(self.branches_lists):
            for tree_indx2,tree_2 in enumerate(self.branches_lists):
                if tree_indx1==tree_indx2:
                    continue
                for leaf_index1,leaf1 in enumerate(tree_1):
                    for leaf_index2,leaf2 in enumerate(tree_2):
                        self.association_leaves[str(tree_indx1)+'_'+str(leaf_index1)+'|'+str(tree_indx2)+'_'+str(leaf_index2)]=0
        for inst in association_dict:
            for leaf1 in association_dict[inst]:
                for leaf2 in association_dict[inst]:
                    if leaf1 == leaf2:
                        continue
                    else:
                        self.association_leaves[leaf1+'|'+leaf2]+=1