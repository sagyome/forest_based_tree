import numpy as np
import pandas as pd
import glob
from sklearn.datasets import load_iris
from sklearn.feature_extraction import DictVectorizer
def read_iris_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data[:], columns=iris.feature_names)
    data['class'] = [iris.target_names[i] for i in iris.target]
    y_column='class'
    feature_types = ['float']*4
    x_columns=iris.feature_names
    return data,x_columns,y_column,feature_types

def dummify_data(data,feature_cols, label_col):
    """
    This function identify categorical features and convert them into multiple binary features (one for each category)

    :param data: pandas dataframe
    :param feature_cols: feature names
    :param label_col: label column name
    :return: pandas dataframe with dummy features, names of the new features
    """

    #The following line identifies the categorical features in the dataset
    char_cols = data[feature_cols].dtypes.pipe(lambda x: x[x == 'object']).index

    #convert the categorical features to multiple dummy variables and add them to the dataframe
    for col in char_cols:
        dummies = pd.get_dummies(data[col])
        dummies.columns = [col + '=' + category for category in dummies.columns]
        data = pd.concat([data, dummies], axis=1)

    #Remove the original categorical features from the dataframe
    data = data.drop(char_cols,axis=1)
    feature_cols = [col for col in data.columns if col != label_col]
    return data, feature_cols

def read_winery_data():
    x_columns=['Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium',
                                     'Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins',
                                    'Color intensity','Hue','OD280/OD315 of diluted wines','Proline']
    y_column='class'
    data = pd.read_csv("datasets/wine.data",names=[y_column]+x_columns)
    #feature_types = ['float'] * 4 + ['int'] + ['float'] * 7 + ['int']
    feature_types = ['float']*len(x_columns)
    return data, x_columns, y_column,feature_types
def read_breast_cancer_data():
    x_columns = ['Clump_thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size'
        , 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']
    y_column='class'
    feature_types = ['float'] * 9
    data = pd.read_csv('datasets/breast-cancer-wisconsin.data',names=x_columns+[y_column])
    data = data[data['Bare Nuclei'] != '?']
    data['Bare Nuclei'] = [int(i) for i in data['Bare Nuclei']]
    return data, x_columns, y_column, feature_types
def read_tic_tac_toe_dataset1():
    x_columns=['top - left - square','top - middle - square','top - right - square','middle - left - square',
               'middle - middle - square','middle - right - square','bottom - left - square','bottom - middle - square','bottom - right - square']
    y_column='class'
    data=pd.read_csv('datasets/tic-tac-toe.data',names=x_columns+[y_column])
    dv=DictVectorizer()
    dv_data=dv.fit_transform([dict(row) for index,row in data[x_columns].iterrows()])
    dv_data=pd.DataFrame(dv_data.toarray(),columns=dv.feature_names_)
    dv_data[y_column]=data[y_column]
    data=dv_data
    feature_types = ['int']*len(dv.feature_names_)
    return data, dv.feature_names_, y_column, feature_types
def read_tic_tac_toe_dataset():
    feature_cols=['top - left - square','top - middle - square','top - right - square','middle - left - square',
               'middle - middle - square','middle - right - square','bottom - left - square','bottom - middle - square','bottom - right - square']
    label_col='class'
    data=pd.read_csv('datasets/tic-tac-toe.data',names=feature_cols+[label_col])
    data, feature_cols = dummify_data(data, feature_cols, label_col)
    feature_types = ['int'] * len(feature_cols)
    return data, feature_cols, label_col, feature_types
def read_australian():
    x_columns = ["A" + str(i) for i in range(14)]
    y_column='class'
    feature_types = ['int', 'float', 'float', 'int', 'int', 'int', 'float', 'int', 'int', 'int', 'int', 'int', 'int',
                     'int', 'int']
    feature_types = ['float']*len(x_columns)
    data = pd.read_csv("datasets/australian.dat", sep=" ", names=x_columns+['class'])
    return data,x_columns,y_column,feature_types
def read_nurse():
    x_columns = ['x' + str(i) for i in range(8)]
    y_column='class'
    data = pd.read_csv("datasets/post-operative.data", names=x_columns+[y_column])
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    data = dv_data
    feature_types = ['int'] * len(dv.feature_names_)
    return data, dv.feature_names_, y_column, feature_types
def read_kohkiloyeh():
    data = pd.read_excel('datasets/kohkiloyeh.xlsx')
    y_column = 'pb'
    x_columns = [col for col in data.columns if col!=y_column]
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    data = dv_data
    feature_types = ['int'] * len(dv.feature_names_)
    return data, dv.feature_names_, y_column, feature_types
def read_haberman():
    data = pd.read_csv('datasets/haberman.data',names=['Age','year_of_operation','number_of_positive_axiilary_nodes','class'])
    y_column='class'
    x_columns=[col for col in data.columns if col!='class']
    feature_types = ['int']*len(x_columns)
    return data, x_columns, y_column, feature_types
def read_balance_scale():
    data = pd.read_csv('datasets/balance-scale.data', names=['class', 'left-weight', 'left-dist', 'right-weight', 'right-dist'])
    x_columns=data.columns[1:]
    y_column='class'
    feature_types = ['int']*4
    return data, x_columns, y_column, feature_types
def read_spambase():
    x_columns = []
    with open('datasets/spambase.names', 'r') as f:
        for line in f:
            x_columns.append(line.replace('\n', ''))
    y_column = 'is_spam'
    data = pd.read_csv('datasets/spambase.data', names=x_columns + [y_column])
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types
def read_zoo():
    x_columns = ['x'+str(i) for i in range(17)]
    y_column = 'class'
    data = pd.read_csv('datasets/zoo.data',names=x_columns+[y_column])
    x_columns = x_columns[1:]
    feature_types = ['int']*16
    return data, x_columns, y_column, feature_types
def read_segment():
    data = pd.read_csv('datasets/segment.csv',index_col=0)
    data['class'] = data.index.values
    data.index = [i for i in range(len(data))]
    x_columns = [col for col in data.columns[:-1]]
    y_column = 'class'
    feature_types = ['float']*2+['int']+['float']*16
    return data,x_columns,y_column,feature_types
def read_german():
    x_columns = ['x'+str(i) for i in range(20)]
    y_column = 'class'
    data = pd.read_csv('datasets/german.data',sep=' ', names = x_columns+[y_column])
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    feature_types = ['int'] * len(dv.feature_names_)
    data = dv_data
    return data, dv.feature_names_, y_column, feature_types
def read_pima():
    x_columns = ['x'+str(i) for i in range(8)]
    y_column = 'class'
    data = pd.read_csv('datasets/pima.scv', names=x_columns+[y_column])
    #feature_types = ['int']*5+['float']*2+['int']
    feature_types = ['float']*len(x_columns)
    return data,x_columns,y_column,feature_types
def read_letter():
    x_columns = ['x'+str(i) for i in range(16)]
    y_column = 'class'
    data = pd.read_csv('datasets/letter-recognition.data',names = [y_column]+x_columns)
    feature_types = ['int']*len(x_columns)
    return data, x_columns, y_column, feature_types
def read_ionosphere():
    data = pd.read_csv('datasets/dataset_59_ionosphere.csv')
    x_columns = data.columns[:-1]
    y_column = 'class'
    feature_types = ['int']*2+['float']*32
    return data,x_columns,y_column,feature_types
def read_autism_adult():
    x_columns = ['x'+str(i) for i in range(20)]
    y_column = 'class'
    data = pd.read_csv('datasets/autism_adult.txt',names=x_columns+[y_column])
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    feature_types = ['int'] * len(dv.feature_names_)
    data = dv_data
    return data, dv.feature_names_, y_column, feature_types
def read_bank():
    y_column = 'class'
    data = pd.read_csv('datasets/bank.txt'  ,sep=';')
    x_columns = data.columns[:-1]
    data = data.rename({'y': 'class'}, axis=1)
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    feature_types = ['int'] * len(dv.feature_names_)
    data = dv_data
    return data, dv.feature_names_, y_column, feature_types
def read_banknote():
    x_columns = ['x' + str(i) for i in range(4)]
    y_column = 'class'
    data = pd.read_csv('datasets/banknote.txt', names=x_columns + [y_column])
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types
def read_car():
    x_columns = ['x' + str(i) for i in range(6)]
    y_column = 'class'
    data = pd.read_csv('datasets/car.data', names=x_columns + [y_column])
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    feature_types = ['int'] * len(dv.feature_names_)
    data = dv_data
    return data, dv.feature_names_, y_column, feature_types
def read_adult():
    x_columns = ['x' + str(i) for i in range(14)]
    y_column = 'class'
    data = pd.read_csv('datasets/adult.data', names=x_columns + [y_column])
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    feature_types = ['int'] * len(dv.feature_names_)
    data = dv_data
    return data, dv.feature_names_, y_column, feature_types
def read_connect():
    x_columns = ['x' + str(i) for i in range(42)]
    y_column = 'class'
    data = pd.read_csv('datasets/connect-4.data', names=x_columns + [y_column])
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    feature_types = ['int'] * len(dv.feature_names_)
    data = dv_data
    return data, dv.feature_names_, y_column, feature_types
def read_credit():
    x_columns = ['x' + str(i) for i in range(15)]
    y_column = 'class'
    data = pd.read_csv('datasets/credit.data', names=x_columns + [y_column])
    for col in ['x1','x2','x7','x13']:
        data[col] = data[col].replace('?', -1000).astype(float).values
        data = data[data[col] > -1000]
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    #feature_types = ['int' if i not in ['x1','x2','x7','x13'] else float for i in dv.feature_names_]
    data = dv_data
    data = data[data['class'].apply(lambda x: isinstance(x, str))]
    feature_types = ['float'] * len(dv.feature_names_)
    return data, dv.feature_names_, y_column, feature_types
def read_ctherapy():
    data = pd.read_csv('datasets/cryotherapy.csv')
    data = data.rename({'Result_of_Treatment':'class'},axis=1)
    y_column = 'class'
    x_columns = data.columns[:-1]
    feature_types = ['int']*len(x_columns)
    return data, x_columns, y_column, feature_types
def read_internet_trust():
    x_columns = ['x' + str(i) for i in range(4)]
    y_column = 'class'
    data = pd.read_csv('datasets/disshonest_internet.txt.txt', names=x_columns + [y_column], sep=' ')
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    feature_types = ['int'] * len(dv.feature_names_)
    data = dv_data
    return data, dv.feature_names_, y_column, feature_types
def read_dress():
    data = pd.read_csv('datasets/dress.csv')
    data = data[data.columns[1:]]
    data = data.rename({'Recommendation': 'class'}, axis=1)
    x_columns = data.columns[:-1]
    y_column = 'class'
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    data = dv_data
    data = data.drop(['Decoration', 'FabricType', 'Material','Price' ,'NeckLine', 'Pattern Type', 'Season', 'SleeveLength', 'waiseline','Decoration'], axis=1)
    x_columns = data.columns[:-1]
    feature_types = ['int' if '=' in col else 'float' for col in x_columns]
    return data, x_columns , y_column, feature_types
def read_cad():
    data = pd.read_csv('datasets/cad_Detection.csv')
    data = data.rename({'Cath': 'class'}, axis=1)
    x_columns = data.columns[:-1]
    y_column = 'class'
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    data = dv_data
    feature_types = ['int' if col not in ['BMI', 'CR', 'HB', 'K'] else 'float' for col in dv.feature_names_]
    return data, dv.feature_names_, y_column, feature_types
def read_forest():
    data = pd.read_csv('datasets/forsttypes.csv')
    x_columns = data.columns[1:]
    y_column = 'class'
    feature_types = ['int' if col.startswith('b') else 'float' for col in x_columns]
    return data, x_columns, y_column, feature_types
def read_gesture():
    pathes = glob.glob('datasets/gesture/*')
    data = pd.DataFrame()
    for p in pathes:
        data = pd.concat([data, pd.read_csv(p)])
    data = data.rename({'Phase': 'class'}, axis=1)
    x_columns = data.columns[:-1]
    y_column = 'class'
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types
def read_glass():
    x_columns = ['x'+str(i) for i in range(10)][1:]
    y_column='class'
    data = pd.read_csv('datasets/glass.data',names=x_columns+[y_column])
    feature_types = ['float']*len(x_columns)
    return data, x_columns, y_column, feature_types
def read_htru():
    x_columns = ['x' + str(i) for i in range(8)]
    y_column = 'class'
    data = pd.read_csv('datasets/HTRU_2.csv', names=x_columns + [y_column])
    feature_types = ['float'] * len(x_columns)
    return data, x_columns, y_column, feature_types
def read_liver():
    x_columns = ['x' + str(i) for i in range(10)]
    y_column = 'class'
    data = pd.read_csv('datasets/liver.csv', names=x_columns + [y_column])
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    data = dv_data.dropna()
    feature_types = ['float' if '=' not in col else 'int' for col in data.columns]
    return data, dv.feature_names_, y_column, feature_types
def read_parkinson():
    data = pd.read_csv('datasets/parkinsons.data')
    data = data.drop(['name'],axis=1)
    data = data.rename({'status': 'class'}, axis=1)
    x_columns = [col for col in data.columns if col != 'class']
    y_column = 'class'
    dv = DictVectorizer()
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    feature_types = ['int' if '=' in col else 'float' for col in dv.feature_names_]
    return data, dv.feature_names_, y_column, feature_types
def read_vegas():
    dv = DictVectorizer()
    data = pd.read_csv('datasets/LasVegasTripAdvisorReviews-Dataset.csv', sep=';')
    data
    data = data.rename({'Score': 'class'}, axis=1)
    data = data.drop(['User country', 'Hotel name'], axis=1)
    y_column = 'class'
    x_columns = [col for col in data.columns if col != 'class']
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    feature_types = ['float' if 'Nr.' in col else 'int' for col in dv.feature_names_]
    return dv_data, dv.feature_names_, y_column, feature_types
def read_magic():
    x_columns = ['x' + str(i) for i in range(10)]
    y_column = 'class'
    data = pd.read_csv('datasets/magic04.data', names=x_columns + [y_column]).sample(2000)
    feature_types = ['float'] * len(x_columns)
    return data,x_columns,y_column,feature_types
def read_mamo():
    x_columns = ['x' + str(i) for i in range(5)]
    y_column = 'class'
    data = pd.read_csv('datasets/mammographic_masses.data', names=x_columns + [y_column])
    data = data[['?' not in str(row) for indx, row in data.iterrows()]]
    for col in x_columns:
        data[col] = data[col].astype(int)
    feature_types = ['int']*len(x_columns)
    return data, x_columns, y_column, feature_types
def read_mushroom():
    dv = DictVectorizer()
    x_columns = ['x' + str(i) for i in range(22)]
    y_column = 'class'
    data = pd.read_csv('datasets/mushroom.data', names=x_columns + [y_column])
    data = data[['?' not in str(row) for indx, row in data.iterrows()]]
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    dv_data = dv_data[~dv_data[y_column].isnull()]
    for col in dv.feature_names_:
        dv_data[col] = dv_data[col].astype(int)
    feature_types = ['int'] * len(dv.feature_names_)
    return dv_data,dv.feature_names_,y_column,feature_types
def read_occupancy():
    data = pd.read_csv('datasets/occupancy.txt')
    data = data.drop(['date'], axis=1)
    data = data.rename({'Occupancy': 'class'},axis=1)
    x_columns = data.columns[:-1]
    y_column = 'class'
    feature_types = ['float'] * len(x_columns)
    return data,x_columns,y_column,feature_types
def read_biodeg():
    x_columns = ['x' + str(i) for i in range(41)]
    y_column = 'class'
    data = pd.read_csv('datasets/biodeg.txt', sep=';', names=x_columns + [y_column])
    feature_types = ['float']*len(x_columns)
    return data, x_columns, y_column, feature_types
def read_seismic():
    dv = DictVectorizer()
    x_columns = ['x' + str(i) for i in range(18)]
    y_column = 'class'
    data = pd.read_csv('datasets/seismic.arff', names=x_columns + [y_column])
    dv_data = dv.fit_transform([dict(row) for index, row in data[x_columns].iterrows()])
    dv_data = pd.DataFrame(dv_data.toarray(), columns=dv.feature_names_)
    dv_data[y_column] = data[y_column]
    feature_types = ['int'] * len(dv.feature_names_)
    return dv_data, dv.feature_names_, y_column, feature_types
def read_abalone():
    x_columns = ['x' + str(i) for i in range(8)]
    y_column = 'class'
    data = pd.read_csv('datasets/abalone.data', names=x_columns + [y_column])
    data['x0'] = [1 if i == 'M' else 0 for i in data['x0']]
    feature_types = ['int'] + ['float'] * 7
    return data, x_columns, y_column, feature_types
def read_ecoli():
    x_columns = ['x'+str(i) for i in range(8)]
    y_column = 'class'
    data = pd.read_csv('datasets/ecoli1.data',names=x_columns+[y_column])
    x_columns = x_columns[1:]
    feature_types = ['float']*len(x_columns)
    return data, x_columns, y_column, feature_types
def read_divorce():
    data = pd.read_csv('datasets/divorce.csv',sep=';')
    x_columns = data.columns[:-1]
    y_column = data.columns[-1]
    feature_types = len(x_columns)*['float']
    return data, x_columns, y_column, feature_types

def read_acute_inflam():
    path = 'datasets/acute.csv'
    data = pd.read_csv(path,header=None)
    data = data[data.columns[:-1]]
    data.columns = ['x'+str(i) for i in data.columns[:-1]]+['class']
    for col in data.columns[1:-1]:
        data[col] = data[col].apply(lambda x: x=='yes').astype(int)
    feature_cols = data.columns[:-1]
    label_col = 'class'
    feature_types = ['float']+['int']*5
    return data,feature_cols,label_col,feature_types

def read_acute_nephritis():
    path = 'datasets/acute.csv'
    data = pd.read_csv(path,header=None)
    new_cols = [0,1,2,3,4,5,7]
    data = data[new_cols]
    data.columns = ['x'+str(i) for i in data.columns[:-1]]+['class']
    for col in data.columns[1:-1]:
        data[col] = data[col].apply(lambda x: x=='yes').astype(int)
    feature_cols = data.columns[:-1]
    label_col = 'class'
    feature_types = ['float']+['int']*5
    return data,feature_cols,label_col,feature_types
def read_hayes_roth():
    data = pd.read_csv('datasets/hayes-roth.data',names=['name','hobby','age','edu','martial_status','class'])
    data.drop(columns = 'name',inplace=True)
    feature_cols = data.columns
    label_col = 'class'
    feature_types = ['float']*len(feature_cols)
    return data,feature_cols,label_col,feature_types

def get_dataset_by_string(s):
    if s=='iris':
        return read_iris_data()
    elif s=='winery':
        return read_winery_data()
    elif s=='divorce':
        return read_divorce()
    elif s=='breast cancer':
        return read_breast_cancer_data()
    elif s == 'aust_credit':
        return read_australian()
    elif s == 'nurse':
        return read_nurse()
    elif s == 'kohkiloyeh':
        return read_kohkiloyeh()
    elif s =='haberman':
        return read_haberman()
    elif s=='tic-tac-toe':
        return read_tic_tac_toe_dataset()
    elif s=='balance_scale':
        return read_balance_scale()
    elif s=='spambase':
        return read_spambase()
    elif s=='zoo':
        return read_zoo()
    elif s=='segment':
        return read_segment()
    elif s=='german':
        return read_german()
    elif s=='pima':
        return read_pima()
    elif s=='letter':
        return read_letter()
    elif s=='ionosphere':
        return read_ionosphere()
    elif s=='autism':
        return read_autism_adult()
    elif s=='bank':
        return read_bank()
    elif s=='banknote':
        return read_banknote()
    elif s=='car':
        return read_car()
    elif s=='adult':
        return read_adult()
    elif s=='connect':
        return read_connect()
    elif s=='credit':
        return read_credit()
    elif s=='cryotherapy':
        return read_ctherapy()
    elif s=='internet':
        return read_internet_trust()
    elif s=='dress':
        return read_dress()
    elif s=='cad':
        return read_cad()
    elif s=='forest':
        return read_forest()
    elif s=='gesture':
        return read_gesture()
    elif s=='glass':
        return read_glass()
    elif s=='htru':
        return read_htru()
    elif s=='liver':
        return read_liver()
    elif s=='vegas':
        return read_vegas()
    elif s=='magic':
        return read_magic()
    elif s=='mamographic':
        return read_mamo()
    elif s=='occupancy':
        return read_occupancy()
    elif s=='biodeg':
        return read_biodeg()
    elif s=='seismic':
        return read_seismic()
    elif s=='abalone':
        return read_abalone()
    elif s=='ecoli':
        return read_ecoli()
    elif s=='acute-inflam':
        return read_acute_inflam()
    elif s=='acute-nephitis':
        return read_acute_nephritis()
    elif s=='hayes-roth':
        return read_hayes_roth()