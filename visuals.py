###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import importlib
importlib.import_module('mpl_toolkits.mplot3d').Axes3D
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
import analysis as als
import seaborn as sns
import re
from sklearn.model_selection import cross_validate
import math

def distribution(data):
    """
    Visualization code for displaying skewed distributions of features
    """
    
    # Create figure
    fig = pl.figure(figsize = (18,15));

    # Skewed feature plotting
    for i, feature in enumerate(data.columns[:10]):
        ax = fig.add_subplot(5, 5, i+1)
        ax.hist(data[feature], bins = 25, color = '#00A0A0')
        ax.set_title("'%s'"%(feature), fontsize = 14)
        ax.set_xlabel("Value")
        ax.set_ylabel("Number of Records")
        #ax.set_ylim((0, 2000))
        #ax.set_yticks([0, 500, 1000, 1500, 2000])
        #ax.set_yticklabels([0, 500, 1000, 1500, ">2000"])

        fig.suptitle("Distributions Features", \
            fontsize = 16, y = 1.03)

    fig.tight_layout()
    fig.show()

def show_output_classes(data, data_clean):
    '''
    Visualization code for histogram of classes
    '''
    
    # Create figure
    fig = pl.figure(figsize=(10,6))
    
    encoded_data = data.apply(lambda x: als.encode_diagnosis(x))
    encoded_data_clean = data_clean.apply(lambda x: als.encode_diagnosis(x))

    ax = fig.add_subplot(111)
    n, bins, patches = ax.hist(encoded_data, bins=np.arange(3), alpha=0.5, color='b', label='Lost data', width=0.5)
    n_c, bins_c, patches_c = ax.hist(encoded_data_clean, bins=np.arange(3), color='k', label = 'Data filtered for outliers',width=0.5)
    '''
    colors = ['r', 'g']
    for i in range(2):
        patches[i].set_fc(colors[i])
        patches_c[i].set_fc(colors[i])
    '''    
    ax.set_title('Barplot of output classes', fontsize=16)
    ax.set_xticks([b+0.25 for b in bins[:-1]])
    ax.set_xticklabels(['Benign', 'Malign'], fontsize=16)
    
    ax.legend(fontsize=16)
    ax.set_ylabel('Number of records', fontsize=16)
    fig.tight_layout()
    fig.show()

def violin_swarm_plots(data, **kwargs):
    cols = ['diagnosis'] + als.return_select_cols(data, which=kwargs['which'])
    d = pd.melt(data[cols], id_vars = 'diagnosis', var_name = 'features', value_name = 'value')
    
    sns.set(font_scale=1.5)
    fig, ax = pl.subplots(figsize=(15,10))
    ax = sns.violinplot(x='features', y = 'value', hue='diagnosis', data=d, split=True, inner='quart')
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)    
    fig.tight_layout()
    pl.subplots_adjust(bottom=0.2)
    pl.show()

def observe_correlations(data, **kwargs):
    cols = als.return_select_cols(data, which=kwargs['which'])
    fig,ax = pl.subplots(figsize=(10,7))
    sns.heatmap(data[cols].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
    fig.tight_layout()
    pl.show()
        
def plot_pca_variance(pca):
    x = np.arange(1,len(pca.components_)+1)
    fig, ax = pl.subplots(figsize=(10,6))
    
    # plot the cumulative variance
    ax.plot(x, np.cumsum(pca.explained_variance_ratio_), '-o', color='black')

    # plot the components' variance
    ax.bar(x, pca.explained_variance_ratio_, align='center', alpha=0.5)

    # plot styling
    ax.set_ylim(0, 1.05)
    
    for i,j in zip(x, np.cumsum(pca.explained_variance_ratio_)):
        ax.annotate(str(j.round(2)),xy=(i+.2,j-.02))
    ax.set_xticks(range(1,len(pca.components_)+1))
    ax.set_xlabel('PCA components')
    ax.set_ylabel('Explained Variance')
    
    fig.tight_layout()
    pl.show()
    
def pca_results(good_data, pca, **kwargs):
    cols = als.return_select_cols(good_data, which=kwargs['which'])
    cols_indices = [i for i, j in enumerate(good_data.keys()) if j in cols]
    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_[:,cols_indices], 4), columns = good_data.keys()[cols_indices])
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = pl.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)
    
def scatter_two_dimensions(reduced_features, output_float):
    fig, ax = pl.subplots(figsize=(8,5))
    ax.scatter(reduced_features.loc[:,'Dimension 1'], reduced_features.loc[:, 'Dimension 2'], c=output_float, cmap='winter')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Projections of features on first two principal components')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    fig.tight_layout()
    pl.show()
    
def biplot(good_data, reduced_data, output_float, pca):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.
    
    good_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)
    
    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    fig = pl.figure(figsize = (18,22))
    ax = fig.add_subplot(111, projection='3d')
    # scatterplot of the reduced data    
    xs = reduced_data.loc[:, 'Dimension 1']
    ys = reduced_data.loc[:, 'Dimension 2']
    zs = reduced_data.loc[:, 'Dimension 3']
    
    ax.scatter(xs, ys, zs, c=output_float, cmap='winter')
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 5.6, 6

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.plot([0, arrow_size*v[0]], [0, arrow_size*v[1]], [0, arrow_size*v[2]], lw=1.5, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, v[2]*text_pos, good_data.columns[i], color='black', 
                 ha='center', va='center', fontsize=14)

    ax.set_xlabel("Dimension 1", fontsize=18)
    ax.set_ylabel("Dimension 2", fontsize=18)
    ax.set_zlabel("Dimension 3", fontsize=18)

    ax.set_title("PC plane with original feature projections.", fontsize=18);
    
    fig.tight_layout()
    pl.show()
    
def plot_evaluation_metrics(clfs, clf_labels, x, y, cv=5):
    scoring = ['accuracy', 'precision', 'recall']
    scores = {}
    for label, clf in zip(clf_labels, clfs):
        scores[label] = cross_validate(clf, x, y, cv=cv, scoring=scoring, return_train_score=True)
    colors = ['b', 'g', 'r', 'k', 'c']
    lab2 = ['Avg', 'Min', 'Max']
    fig, ax = pl.subplots(2,3, figsize = (20,15))
    for i, f in enumerate(['train', 'test']):
        for j, s in enumerate(scoring):
            minval=1
            for k, lab1 in enumerate(clf_labels):
                scs = scores[lab1]
                key = [sc for sc in scs.keys() if re.search('{}(.){}'.format(f,s),sc)][0]
                alphac = [1,0.2,0.5]
                for l, lval in enumerate([np.mean(scs[key]), np.min(scs[key]), np.max(scs[key])]):
                    lab = lab1 + ' ' + lab2[l]
                    if lval < minval:
                        minval = lval
                    ax[i,j].bar(k+1+(0.23*l), lval, 0.23, color=colors[k], label=lab, alpha=alphac[l]) 
            ax[i,j].legend()
            ax[i,j].set_ylim(minval-0.01,1)
            ax[i,j].set_xlim(0,8)
            ax[i,j].set_title('{} {} scores'.format(f,s))
            ax[i,j].set_ylabel('Score')
            ax[i,j].set_xticklabels([])
    fig.tight_layout()
    pl.show()

def feature_plot(importances, X_train, y_train):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:5]]
    values = importances[indices][:5]

    # Creat the plot
    fig = pl.figure(figsize = (9,5))
    pl.title("Normalized Weights for First Five Most Predictive Features", fontsize = 16)
    pl.bar(np.arange(5), values, width = 0.6, align="center", color = '#00A000', \
          label = "Feature Weight")
    pl.bar(np.arange(5) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
          label = "Cumulative Feature Weight")
    pl.xticks(np.arange(5), columns)
    pl.xlim((-0.5, 4.5))
    pl.ylabel("Weight", fontsize = 12)
    pl.xlabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper center')
    pl.tight_layout()
    pl.show()  
