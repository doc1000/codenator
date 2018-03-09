
from math import ceil
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from basis_expansions import NaturalCubicSpline
from regression_tools.dftransformers import (
    ColumnSelector, Identity, FeatureUnion, MapFeature, Intercept)
from regression_tools.plotting_tools import (
    plot_univariate_smooth,
    bootstrap_train,
    display_coef,
    plot_bootstrap_coefs,
    plot_partial_depenence,
    plot_partial_dependences,
    predicteds_vs_actuals)

#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')


#class regression_holder(object):
#def panda_eda():
show_plots = False
filename = '~/galvanize/dsi-solns-g64/linear-regression/balance.csv'
data_df = pd.read_csv(filename)
print(data_df.shape)
print(data_df.head())
print(data_df.info())
print(data_df.describe())
print('The count of nulls is: \n',data_df[data_df.isnull()].count())
print('What should be done with null values?')


def plot_univariate_bynumeric(target_column):
    fig, axs = plt.subplots(len(numeric_predictors), figsize=(14, 14))
    for name, ax in zip(numeric_predictors, axs.flatten()):
        plot_univariate_smooth(ax,
        data_df[name].values.reshape(-1, 1),
        data_df[target_column],
        bootstrap=100)
    ax.set_title(name)
    fig.tight_layout()
    plt.show()

def simple_spline_specification(name, knots):
    '''
        making a lot of these simple spline specifications, let's write a small function to make them for us.
        income_spec = Pipeline([
            ('Income_select', ColumnSelector(name="Income")),
            ('Income_spline', NaturalCubicSpline(knots=[25, 50, 75, 100, 125]))
        ])
    '''
    select_name = "{}_select".format(name)
    spline_name = "{}_spline".format(name)
    return Pipeline([
        (select_name, ColumnSelector(name=name)),
        (spline_name, NaturalCubicSpline(knots=knots))
    ])


if __name__ == '__main__':
    if show_plots == True:
        _ = scatter_matrix(data_df, alpha=0.2, figsize=(20, 20), diagonal='kde')
        plt.show()

    target_column = 'Balance'
    if len(target_column)==0:
        target_column = str(input('What is the target column name?\n'))
    y_values = data_df[target_column]

    drop_columns = ['Unnamed: 0']
    if len(drop_columns) == 0:
        drop_columns = split(str(input('What columns should be dropped?\n')),',').strip
    data_df = data_df.drop(drop_columns,axis=1)
    #maybe figure out how to store this in a temp csv so you don't ask each time
    numeric_predictors = data_df.drop(target_column, axis=1).select_dtypes(exclude=[np.object]).columns

    if show_plots == True:
        plot_univariate_bynumeric(target_column)
    print('Correlation matrix: \n',data_df.corr())

    log_transform = 'y'
    if len(log_transform)==0:
        log_transform = str(input('Should the target be transformed by log? (y/n)\n'))
    if log_transform == 'y':
        log_target_column = 'Log'+target_column
        data_df.loc[:, log_target_column] = np.log(data_df[target_column])
        if show_plots == True: plot_univariate_bynumeric(log_target_column)

    '''
    specify pipelines for each feature you want to include.  even if small and limited
    '''
