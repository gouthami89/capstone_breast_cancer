# Data Exploratory functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display # Allows the use of display() for DataFrames
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
import re

def split_data(data):
    output = data['diagnosis']
    features = pd.DataFrame(data=data)
    cols = ['id', 'diagnosis']
    for col in cols:
        if col in features.columns:
            features = features.drop(col, axis=1)
    return output, features

# For each feature print the rows which have outliers in all features 
def print_outliers(data, how_far=2, worst_th=6, to_display=False):
    # Select the last 10 features as they are the worst collected during measurements
    data = data.iloc[:,11:30]
    really_bad_data = defaultdict(int)
    for col in data.columns:
        Q1 = np.percentile(data[col], 25)
        Q3 = np.percentile(data[col], 75)
        step = (Q3-Q1)*how_far
        bad_data = list(data[~((data[col]>=Q1-step)&(data[col]<=Q3+step))].index)
        for i in bad_data:
            really_bad_data[i]+= 1
        # Display the outliers
    max_ind = max(really_bad_data.values())
    worst_points = [k for k, v in really_bad_data.iteritems() if v > max_ind-worst_th]
    if to_display:
        print "Data points considered outliers are:"  
        display(data.ix[worst_points,:])
    return worst_points
    
def transform_log_minmax(data):
    cols = data.columns
    data_transformed = pd.DataFrame(data=data)
    scaler = MinMaxScaler()
    for col in cols:
        data_transformed[col] = data[col].apply(lambda x: np.log(x+1))
        data_transformed[col] = scaler.fit_transform(data[col].values.reshape(-1,1))
    return data_transformed

def encode_diagnosis(d):
    if d== 'B':
        ed = 0
    else:
        ed = 1
    return ed
         
def return_reduced_data(good_data, pca):
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    reduced_data = pd.DataFrame(data=pca.transform(good_data), columns=dimensions)
    return reduced_data

def return_select_cols(data, **kwargs):
    checks = ['radius', 'area', 'perimeter']
    cols = [c for c in data.columns for ch in checks if re.search('{}(.)'.format(ch), c)]
    if kwargs['which']=='mean_non_dims':
        cols = [c for c in data.columns if c not in cols]
        cols = [c for c in cols if re.search('(.)_mean', c)]
    elif kwargs['which']=='se_non_dims':
        cols = [c for c in data.columns if c not in cols]
        cols = [c for c in cols if re.search('(.)_se', c)]
    elif kwargs['which']=='worst_non_dims':
        cols = [c for c in data.columns if c not in cols]
        cols = [c for c in cols if re.search('(.)_worst', c)]
    elif kwargs['which']=='all':
        cols = data.columns
    return cols

def print_evaluation_metrics(clf, x, y, scoring, cv=5, only_times=True, print_times=True):
    scores = cross_validate(clf, x, y, cv=cv, scoring=scoring, return_train_score=True)
    if print_times:
        print('Average fit time is:   {:.3f}s'.format(np.mean(scores['fit_time'])))
        print('Average score time is: {:.3f}s\n'.format(np.mean(scores['score_time'])))
    if not only_times:
        print(' {: >7} {: >10} |  {: >3}    |  {: >3}    |  {: >3}    |'.format(' ', ' ', 'Avg', 'Min', 'Max'))
        for f in ['train', 'test']:
            for s in scoring:
                key = [sc for sc in scores.keys() if re.search('{}(.){}'.format(f,s),sc)]
                print(' {: >7} {: >10} |  {: >.3f}  |  {: >.3f}  |  {: >.3f}  |'.format(f, s, np.mean(scores[key[0]]), np.min(scores[key[0]]), np.max(scores[key[0]])))
               
        
        