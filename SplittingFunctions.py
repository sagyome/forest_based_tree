import numpy as np
from scipy.stats import entropy
from joblib import delayed, Parallel

def select_split_feature_parallel(df, features, mask):
    feature_to_value = {}
    feature_to_metric = {}
    features_values_list = Parallel(n_jobs=-1, verbose=0)(delayed(check_feature_split_value)(df, feature,mask) for feature in features)
    for item in features_values_list:
        feature, value, metric = item[0],item[1],item[2]
        feature_to_value[feature] = value
        feature_to_metric[feature] = metric
    feature = min(feature_to_metric, key=feature_to_metric.get)
    return feature, feature_to_value[feature]
def check_feature_split_value(df, feature, mask):
    value_to_metric = {}
    values = list(set(list(df[str(feature) + '_upper'][mask]) + list(df[str(feature) + '_lower'][mask])))
    np.random.shuffle(values)
    values = values[:3]
    class_probas = np.array([np.array(l) / np.sum(l) for l in df['probas'][mask]])
    classes = set(np.array([np.argmax(x) for x in class_probas]))
    has_same_class = len(classes)==1
    for value in values:
        left_mask = [True if upper <= value  else False for upper in df[str(feature) + "_upper"]]
        right_mask = [True if lower >= value else False for lower in df[str(feature) + '_lower']]
        both_mask = [True if value < upper and value > lower else False for lower, upper in
                     zip(df[str(feature) + '_lower'], df[str(feature) + "_upper"])]
        if has_same_class:
            value_to_metric[value] = get_value_metric(df, left_mask, right_mask, both_mask, mask)
        else:
            value_to_metric[value] = get_value_metric_accuracy(df, left_mask, right_mask, both_mask, mask)
    val = min(value_to_metric, key=value_to_metric.get)
    return feature,val, value_to_metric[val]

def get_value_metric(df,left_mask,right_mask,both_mask,original_mask):
    l_df_mask=np.logical_and(np.logical_or(left_mask,both_mask),original_mask)
    r_df_mask=np.logical_and(np.logical_or(right_mask,both_mask),original_mask)
    if np.sum(l_df_mask)==0 or np.sum(r_df_mask)==0:
        return np.inf
    l_entropy,r_entropy=calculate_entropy(df,l_df_mask),calculate_entropy(df,r_df_mask)
    l_prop=np.sum(l_df_mask)/len(l_df_mask)
    r_prop=np.sum(r_df_mask)/len(l_df_mask)
    return l_entropy*l_prop+r_entropy*r_prop

def get_value_metric_accuracy(df,left_mask,right_mask,both_mask,original_mask):
    l_df_mask=np.logical_and(np.logical_or(left_mask,both_mask),original_mask)
    r_df_mask=np.logical_and(np.logical_or(right_mask,both_mask),original_mask)
    if np.sum(l_df_mask)==0 or np.sum(r_df_mask)==0:
        return np.inf
    l_entropy,r_entropy=calculate_entropy_accuracy(df,l_df_mask),calculate_entropy_accuracy(df,r_df_mask)
    l_prop=np.sum(l_df_mask)/len(l_df_mask)
    r_prop=np.sum(r_df_mask)/len(l_df_mask)
    return l_entropy*l_prop+r_entropy*r_prop

def calculate_entropy(test_df,test_df_mask):
    class_probas = np.array([np.array(l) / np.sum(l) for l in test_df['probas'][test_df_mask]])

    class_probas = class_probas.mean(axis=0)
    probas_sum = np.sum(class_probas)
    class_probas = [i / probas_sum for i in class_probas]
    return entropy(class_probas)

def calculate_entropy_accuracy(test_df,test_df_mask):
    class_probas = np.array([np.array(l) / np.sum(l) for l in test_df['probas'][test_df_mask]])
    print(type(class_probas))
    print(class_probas)
    print('*' * 100)
    values = np.array([np.argmax(x) for x in class_probas])
    values, counts = np.unique(values, return_counts=True)
    probas = counts / np.sum(counts)
    return entropy(probas)
