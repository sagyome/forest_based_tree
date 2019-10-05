from sklearn.metrics import roc_curve, auc
import numpy as np

def get_auc(Y,y_score,classes):
    y_test_binarize=np.array([[1 if i == c else 0 for c in classes] for i in Y])
    fpr, tpr, _ = roc_curve(y_test_binarize.ravel(), y_score.ravel())
    return auc(fpr, tpr)
def predict_with_included_trees(model,included_indexes,X):
    predictions=[]
    for inst in X:
        predictions.append(predict_instance_with_included_tree(model,included_indexes,inst))
    return np.array(predictions)
def predict_instance_with_included_tree(model,included_indexes,inst):
    v=np.array([0]*model.n_classes_)
    for i,t in enumerate(model.estimators_):
        if i in included_indexes:
            v = v + t.predict_proba(inst.reshape(1, -1))[0]
    return v/np.sum(v)
def select_index(rf,current_indexes,validation_x,validation_y):
    options_auc = {}
    for i in range(len(rf.estimators_)):
        if i in current_indexes:
            continue
        predictions = predict_with_included_trees(rf,current_indexes+[i],validation_x)
        options_auc[i] = get_auc(validation_y,predictions,rf.classes_)
    best_index = max(options_auc, key=options_auc.get)
    best_auc = options_auc[best_index]
    return best_auc,current_indexes+[best_index]
def reduce_error_pruning(model,validation_x,validation_y,min_size):
    best_auc,current_indexes = select_index(model,[],validation_x,validation_y)
    while len(current_indexes) <= model.n_estimators:
        new_auc, new_current_indexes = select_index(model, current_indexes,validation_x,validation_y)
        if new_auc <= best_auc and len(new_current_indexes) > min_size:
            break
        best_auc, current_indexes = new_auc, new_current_indexes
        print(best_auc, current_indexes)
    print('Finish pruning')
    return current_indexes
