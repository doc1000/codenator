code snippents

import matplotlib.pyplot as plt
import seaborn
import numpy as np
%matplotlib inline
ipython --pylab #keeps graphics open

import seaborn as sns

import pandas_profiling as pp
from scipy import stats

### How about some legible text in our graph?
import matplotlib as mpl
mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
})

#File handling
with open('data/bay_area_bikeshare/201402_weather_data_v2.csv') as f:
    labels = f.readline().strip().split(',')
data_labels =[(i, label) for i, label in enumerate(labels)]
data_labels

filepath = 'data/bay_area_bikeshare/201402_weather_data_v2.csv'
weather = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=cols)
data_labels = [data_labels[x] for x in cols]

#to write to file
mt_cars.to_csv('mtcars_adj.csv')


#conver jupyter notebook to python script
!jupyter nbconvert --to script config_template.ipynb
#open html from terminal
def open_html():
    firefox power-calculation.html



def bootstrap():
    x = [2,4,3,5,7,2]
    n = len(x)
    bootstraps = 20
    mean_x = np.array([np.mean([x[np.random.randint(n)] for i in np.arange(n)]) for i in np.arange(bootstraps)])
    np.mean(mean_x)
fig = plt.figure()
ax = fig.add_subplot(111)


def bike_scatter(ax):
    plt.xlabel('Humidity')
    plt.ylabel('Dew Point')
    x = weather[:,2]
    y = weather[:,1]
    ax.scatter(x,y)


def make_scatter(ax):
    #The following code generates two arrays populated with integers from 0 to 999
    a = np.random.randint(1000, size=50)
    b = np.random.randint(1000, size=50)
    c = np.add(a, b)
    col_keys = []
    for num in c:
      if num%2 ==0:
          col_keys.append('b')
      else:
          col_keys.append('r')
    #initialize canvas


    #generate scatter plot
    ax.scatter(a, b, c = col_keys)



# In[3]:


def make_lines(ax):

    #Plot the functions y = 3x + 0.5 and y = 5*sqrt(x) on the same figure for values of x between 0 and 5.
    #Remember that ax.plot() takes an array of x-values and an array of y-values.
    #You may find np.arange() or np.linspace() helpful.

    x = np.arange(6)

    y1 = 3 * x + .5
    y2 = 5*(x**.5)

    ax.plot(x, y1, 'k:|',label = "Linear")
    ax.plot(x, y2, 'y-.p',label = "Exponent")

    #Add a legend using ax.legend(). Note that you'll have to specify label='something' for each ax.plot() command.
    ax.legend()

    #How does this grph look with x and/or y on a log scale? Use ax.set_xscale()

    ax.set_xscale('log')

    #Change the color, line style and marker style using the "format string" shorthand


    # In[4]:


def make_bar(ax):

    barheights = [3,5,1]
    barlabels = ['grapes', 'oranges', 'hockey pucks']
    ax.bar(np.arange(len(barheights)), barheights)
    x_pos = np.arange(len(barheights))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(barlabels, rotation=0, size = 8)

    #How would you change the x-position of the labels?
    #We can change the position of the xticks, or the labels of the xticks, but not the position of the labels

#historgram
def bike_hist(ax,d_index):
    '''creates histogram of bike weather data
    d_index is the index position in weather dataset
    '''
    x = weather[:,d_index]
    ax.hist(x, bins=20, label = 'temp')

#panda plotting
df_weather = pd.read_csv(filepath)

df_weather.plot(kind='scatter',x = 'mean_temperature_f',y='mean_humidity')

df_weather['min_temperature_f'].hist()

df.info()
df.describe()

dfbikes['start_date'].dt.hour.hist(bins=24);
dfWeekenddata[dfWeekenddata['duration'] < 3000]['duration'].hist(bins=100);
#panda plotting and eda
dfWeekdaydata = dfbikes[dfbikes['start_date'].dt.weekday < 5]
dfWeekenddata = dfbikes[dfbikes['start_date'].dt.weekday >= 5]
dfWeekdayhour = dfWeekdaydata['start_date'].dt.hour
dfWeekendhour = dfWeekenddata['start_date'].dt.hour
dfbikewithstationsubset.hist(column='duration', by='landmark', bins=100, figsize=(16,10));

#Nupmpy array`
class numpy_stats_by_axis(object):
    def __init__(self, nparray):
        self.row_max = nparray.max(axis=0)
        self.row_min = nparray.min(axis=0)
        self.row_avg = nparray.mean(axis=0)
        self.row_std = nparray.std(axis=0)
        self.col_max = nparray.max(axis=1)
        self.col_min = nparray.min(axis=1)
        self.col_avg = nparray.mean(axis=1)
        self.col_std = nparray.std(axis=1)
        self.column_convert_date

class panda_eda(object):
    def __init__(self,df):
        self.df = df
        self.column_convert_date()

    def p_groupby(self):
        self.df_str = self.df.select_dtypes(include=[np.object])
        self.groupings = []
        for col in self.df_str.columns:
            df_group = self.df.groupby(col)
            df_group.sum()
            df_group.mean()
            groupings.append((col,df_group))
            return groupings

    def create_bins(self):
        self.labels = [ "{0} - {1}".format(i, i + 9) for i in range(0, 100, 10) ]
        df['group'] = pd.cut(self.df.value, range(0, 105, 10), right=False, labels=labels)

    def column_convert_date(self):
        '''automatically tries to select columns that looks like dates and converts them'''
        self.df = self.df.apply(lambda col: pd.to_datetime(col, errors='ignore')
              if col.dtypes == object
              else col,
              axis=0)

    def add_date_columns(self, column_name):
        #df_num.select_dtypes(include=[np.datetime64]).columns
        self.df[self.column_name + '_year'] = pd.DatetimeIndex(self.df[self.column_name]).year
        self.df[self.column_name + '_month'] = pd.DatetimeIndex(df[self.column_name]).month
        self.df[self.column_name + '_day'] = pd.DatetimeIndex(df[self.column_name]).day
        self.df[self.column_name + '_hour'] = pd.DatetimeIndex(df[self.column_name]).hour
        self.df[self.column_name + '_dayofweek'] = pd.DatetimeIndex(df[self.column_name]).dayofweek
        self.df[self.column_name + '_weekday'] = pd.DatetimeIndex(df[self.column_name]).weekday_name
        #self.df[self.column_name + '_time'] = pd.DatetimeIndex(df[self.column_name]).time
        #self.df[self.column_name + '_minute'] = pd.DatetimeIndex(df[self.column_name]).minute

#visualization
#https://pandas.pydata.org/pandas-docs/stable/visualization.html

#should get the basics locked down and in place to plot a particular column
#also get in place 2,2 plots to cover basics in mass,  hist should work but not sure it does

def make_draws(dist, params, size=200):
    """Return array of samples from dist with given params.

    Draw samples of random variables from a specified distribution, dist, with
    given parameters, params, return these in an array.

    Parameters
    ----------
    dist: Scipy.stats distribution object
        Distribution with a .rvs method

    params: dict
        Parameters to define the distribution dist.
                e.g. if dist = scipy.stats.binom then params could be:
                {'n': 100, 'p': 0.25}

    size: int, optional (default=200)
        The number of random variables to draw.

    Returns
    -------
    Numpy array: Sample of random variables
    """
    return dist(**params).rvs(size)


def plot_bootstrapped_statistics(dist, params, stat_function=np.mean, size=200, repeats=5000):
    """Plot distribtuion of sample means for repeated draws from distribution.

    Draw samples of specified size from Scipy.stats distribution and calculate
    the sample mean.  Repeat this a specified number of times to build out a
    sampling distribution of the sample mean.  Plot the results.

    Parameters
    ----------
    dist: Scipy.stats distribution object
            Accepts: scipy.stats: .uniform, .poisson, .binom, .expon, .geom

    params: dict
        Parameters to define the distribution dist.
            e.g. if dist = scipy.stats.binom then params could be:
            {'n': 100, 'p': 0.25}

    stat_function: function
        Statistic to be calculated on bootrapped samples, e.g., np.max or np.mean

    size: int, optional (default=200)
        Number of examples to draw.

    repeats: int, optional (default=5000)
        Number of sample means to calculate.

    Returns
    -------
    ax: Matplotlib axis object
    """
    dist_instance = dist(**params)

    bootstrapped_statistics = []
    for _ in xrange(repeats):
        values = dist_instance.rvs(size)
        bootstrapped_statistics.append(stat_function(values))

    d_label = {
        stats.uniform: ['Uniform', 'Mean of randomly drawn values from a uniform'],
        stats.poisson: ['Poisson', 'Mean events happening in an interval'],
        stats.binom: ['Binomial', 'Mean number of successes'],
        stats.expon: ['Exponential', 'Mean of waiting time before an event'],
        stats.geom: ['Geometric', 'Mean trials until first success']
        }

    dist_name, xlabel = d_label[dist]
    title_str = 'Mean of {0} samples with size {1} drawn from {2} distribution'
    title_str = title_str.format(repeats, size, dist_name)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.hist(bootstrapped_statistics, bins=30)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Counts')
    ax.set_title(title_str, fontsize=14)

    return ax
def sample_sd(arr):
    """Sample Standard Deviation.

    ddof=1 means Delta Degrees of Freedom, changes denom. to N-1.

    Parameters
    ----------
    arr: Numpy array
        Array of data.

    Returns
    -------
    float
    """
    return np.std(arr, ddof=1)


def standard_error(arr):
    """Compute standard errror of arr.

    Parameters
    ----------
    arr: Numpy array

    Returns
    -------
    float
    """
    return sample_sd(arr) / np.sqrt(len(arr))


def bootstrap(arr, iterations=10000):
    """Create a series of bootstrapped samples of an input array.

    Parameters
    ----------
    arr: Numpy array
        1-d numeric data

    iterations: int, optional (default=10000)
        Number of bootstrapped samples to create.

    Returns
    -------
    boot_samples: list of arrays
        A list of length iterations, each element is array of size of input arr
    """
    if type(arr) != np.ndarray:
        arr = np.array(arr)

    if len(arr.shape) < 2:
        arr = arr[:, np.newaxis]
        # [:, np.newaxis] increases the dimension of arr from 1 to 2

    nrows = arr.shape[0]
    boot_samples = []
    for _ in xrange(iterations):
        row_inds = np.random.randint(nrows, size=nrows)
        # because of the [:, np.newaxis] above
        # the following will is a 1-d numeric data with the same size as the input arr
        boot_sample = arr[row_inds, :]
        boot_samples.append(boot_sample)

    return boot_samples


def bootstrap_confidence_interval(sample, stat_function=np.mean, iterations=1000, ci=95):
    """Calculate the CI of chosen sample statistic using bootstrap sampling.

    CI = confidence interval

    Parameters
    ----------
    sample: Numpy array
        1-d numeric data

    stat_function: function, optional (default=np.mean)
        Function for calculating as sample statistic on data

    iterations: int, optional (default=1000)
        Number of bootstrap samples to create

    ci: int, optional (default=95)
        Percent of distribution encompassed by CI, 0<ci<100

    Returns
    -------
    tuple: lower_ci(float), upper_ci(float), bootstrap_samples_statistic(array)
        Lower and upper bounds of CI, sample stat from each bootstrap sample
    """
    boostrap_samples = bootstrap(sample, iterations=iterations)
    bootstrap_samples_stat = map(stat_function, boostrap_samples)
    low_bound = (100 - ci) / 2
    high_bound = 100 - low_bound
    lower_ci, upper_ci = np.percentile(bootstrap_samples_stat,
                                       [low_bound, high_bound])
    return lower_ci, upper_ci, bootstrap_samples_stat



def plot_hist(df, title, color):
    df.hist(figsize=(12, 5), sharey=True, grid=False, color=color, alpha=0.5)
    plt.suptitle(title, size=18, weight='bold')
    plt.tight_layout()
    plt.show()


def plot_hist_basic(df, col):
    """Return a Matplotlib axis object with a histogram of the data in col.

    Plots a histogram from the column col of dataframe df.

    Parameters
    ----------
    df: Pandas DataFrame

    col: str
        Column from df with numeric data to be plotted

    Returns
    -------
    ax: Matplotlib axis object
    """
    data = df[col]
    ax = data.hist(bins=20, normed=1, edgecolor='none', figsize=(10, 7), alpha=.5)
    ax.set_ylabel('Probability Density')
    ax.set_title(col)

    return ax


def get_sample_mean_var(df, col):
    """Calculate sample mean and sample variance of a 1-d array (or Series).

    Parameters
    ----------
    df: Pandas DataFrame

    col: str
        Column from df with numeric data to be plotted

    Returns
    -------
    tuple: (sample mean (float), sample variance (float))
    """

    # by default np.var returns population variance.
    # ddof=1 to get sample var (ddof: delta degrees of freedom)
    data = df[col]
    return data.mean(), data.var(ddof=1)

class methond_of_moments(object):
    """Cacluates and plots gamma and normal model method of moments estimates"""

    def __init__(self):
        """Construct methond_of_moments class"""
        pass

    def fit(self, df, col):
        """Fit Normal and Gamma models to the data using Method of Moments

        Parameters
        ----------
        df: Pandas DataFrame

        col: str
             Column from df with numeric data for Method of Moments
             distribution estimation and plotting
        """

        self.df = df
        self.col = col
        self.samp_mean, self.samp_var = get_sample_mean_var(self.df, self.col)
        self._fit_gamma()
        self._fit_normal()

    def _fit_gamma(self):
        """Fit Normal and Gamma models to the data using Method of Moments"""
        self.alpha = self.samp_mean**2 / self.samp_var
        self.beta = self.samp_mean / self.samp_var

    def _fit_normal(self):
        """Fit Normal and Gamma models to the data using Method of Moments"""
        self.samp_std = self.samp_var**0.5

    def plot_pdf(self, ax=None, gamma=True, normal=True, xlim=None, ylim=None):
        """Plot distribution PDFs using MOM against histogram of data in df[col].

        Parameters
        ----------
        ax: Matplotlib axis object, optional (default=None)
            Used for creating multiple subplots

        gamma: boolean, optional (default=True)
               Fit and plot a Gamma Distribution

        normal: boolean, optional (default=True)
                Fit and plot a Normal Distribution

        xlim: None, or two element tuple
              If not 'None', these limits are used for the x-axis

        ylim: None, or two element tuple
              If not 'None', these limits are used for the y-axis

        Returns
        -------
        ax: Matplotlib axis object
        """

        if ax is None:
            ax = plot_hist_basic(self.df, self.col)

        x_vals = np.linspace(self.df[self.col].min(), self.df[self.col].max())

        if gamma:
            gamma_rv = stats.gamma(a=self.alpha, scale=1/self.beta)
            gamma_p = gamma_rv.pdf(x_vals)
            ax.plot(x_vals, gamma_p, color='b', label='Gamma MOM', alpha=0.6)

        if normal:
            # scipy's scale parameter is standard dev.
            normal_rv = stats.norm(loc=self.samp_mean, scale=self.samp_std)
            normal_p = normal_rv.pdf(x_vals)
            ax.plot(x_vals, normal_p, color='k', label='Normal MOM', alpha=0.6)

        ax.set_ylabel('Probability Density')
        ax.legend()

        if not xlim is None:
            ax.set_xlim(*xlim)

        if not ylim is None:
            ax.set_ylim(*ylim)

        return ax



def plot_year(df, cols, estimation_method_list, gamma=True, normal=True):
    """Loop over 12 columns of data and plot fits to each column.

    Requires plot_method_of_moments or maximum_likelihood_estimation objects

    Parameters
    ----------
    df: Pandas DataFrame

    cols: list of str
        Columns from df with numeric data to be plotted

    estimation_method_list: list of model estimation objects
        plot_method_of_moments or maximum_likelihood_estimation objects

    gamma: boolean, optional (default=True)
        Fit and plot a Gamma Distribution

    normal: boolean, optional (default=True)
        Fit and plot a Normal Distribution

    Returns
    -------
    ax: 4x3 list of Matplotlib axis objects
    """
    # assuming we are plotting a 3x4 grid, so check
    if len(cols) != 12:
        ex_str = 'Expecting 12 monthly columns, got: {}'.format(len(cols))
        raise Exception(ValueError)

    # Pandas does a lexicographic sort on the column names when plotting
    # multiple histograms (with no obvious way to turn that off).
    # So to ensure consistency between the histograms and the fits,
    # use the sorted version of the columns here and in plotting the fits.
    #
    # The three lines indicated below for removal and the  inclusion
    # of the following two will implement the above discussed solution.
    # cols_srt = sorted(cols)
    # axes = df[cols_srt].hist(bins=20, normed=1,
    #                          grid=0, edgecolor='none',
    #                          figsize=(15, 10),
    #                          layout=(3,4))

    # The following two lines removed if lexographical ordering preferred
    fig, axes = plt.subplots(4,3, figsize=(15,10))
    cols_srt = cols
    for month, ax in zip(cols_srt, axes.flatten()):
        # the following line removed if lexographical ordering preferred
        ax.hist(df[month], bins=20, normed=1, edgecolor='none')
        for estimation_method in estimation_method_list:
            estimation_method.fit(df, month)
            estimation_method.plot_pdf(ax=ax, normal=normal, gamma=gamma,
                                       xlim=[0, 16], ylim=[0, 0.35])
    plt.tight_layout()

    return axes


class likelihood_estimation(object):
    """Calculates and plots likelihoods for the Poisson distribution paramter"""

    def __init__(self):
        """Construct likelihood_estimation class"""
        pass

    def _calculate_likelihood(self, lam, ks):
        """Compute the poisson log likelihood of observing ks for paramater lam.

        Parameters
        ----------
        lam: float
            Lambda rate parameter of a Poisson distribution

        ks: Numpy array
            Discrete count data observations assumed to be from a Poisson distribution

            Returns
            -------
            likelihood: float
            """
        return stats.poisson(lam).pmf(ks)


    def fit(self, data, lambda_range):
        """Approximate the log likelihood function for Poisson distrubtion given data

        Parameters
        ----------
        data: Numpy array
            Discrete count data observations assumed to be from a Poisson distribution

        lambda_range: Numpy array
            Different rate parameters (lambda values) at which to evaluate likelihood
            Possibly created using np.linspace
        """

        self.lambda_range = lambda_range
        self.sum_logs = []
        for lam in lambda_range:
            likes = self._calculate_likelihood(lam, data)
            sum_log_liklihood = np.sum(np.log10(likes))
            self.sum_logs.append(sum_log_liklihood)


    def get_maximum_likelihood_estimate(self):
        """Returns the MLE for the poisson distribution of the current fit of the data

        This method should be called after the .fit method

        Returns
        -------
        float: lambda value corresponding to the Maximum Likelihood of the data
            or a message to use the .fit method first.
        """

        maxlik_ind = np.argmax(self.sum_logs)
        return self.lambda_range[maxlik_ind]


    def plot_likelihood_function(self):
        """Plot the estimated log likelihood function

        Returns
        -------
        ax: Matplotlib axis object
        """

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(self.lambda_range, self.sum_logs)
        ax.set_ylabel('$log(L(x | \lambda))$', fontsize=14)
        ax.set_xlabel('$\lambda$', fontsize=14)

        return ax


class maximum_likelihood_estimation(object):
    """Cacluates and plots gamma and normal model maximum likelihood estimation"""

    def __init__(self):
        """Construct maximum_likelihood_estimation class"""
        pass

    def fit(self, df, col):
        """Fit Normal/Gamma models to the data using Maximum Likelihood Estimation

        Parameters
        ----------
        df: Pandas DataFrame

        col: str
             Column from df with numeric data for Maximum Likelihood
             Estimation distribution estimation and plotting
        """
        self.df = df
        self.col = col
        self._fit_gamma()
        self._fit_normal()

    def _fit_gamma(self):
        """Fit Gamma model to the data using Maximum Likelihood Estimation"""
        self.ahat, self.loc, self.bhat = stats.gamma.fit(self.df[self.col], floc=0)

    def _fit_normal(self):
        """Fit Normal model to the data using Maximum Likelihood Estimation"""
        self.mean_mle, self.std_mle = stats.norm.fit(self.df[self.col])


    def plot_pdf(self, ax=None, gamma=True, normal=True, xlim=None, ylim=None):
        """Plot distribution PDFs using MLE against histogram of data in df[col].

        Use Maximum Likelihood Estimation to fit Normal and/or Gamma Distributions
        to the data in df[col] and plot their PDFs against a histogram of the data.

        Parameters
        ----------
        ax: Matplotlib axis object, optional (default=None)
            Used for creating multiple subplots

        gamma: boolean, optional (default=True)
               Fit and plot a Gamma Distribution

        normal: boolean, optional (default=True)
                Fit and plot a Normal Distribution
                xlim: None, or two element tuple

              If not 'None', these limits are used for the x-axis

        ylim: None, or two element tuple
              If not 'None', these limits are used for the y-axis

        Returns
        -------
        ax: Matplotlib axis object
        """

        if ax is None:
            print 'sup'
            ax = plot_hist_basic(self.df, self.col)

        x_vals = np.linspace(self.df[self.col].min(), self.df[self.col].max())

        if gamma:
            gamma_rv = stats.gamma(a=self.ahat, loc=self.loc, scale=self.bhat)
            gamma_p = gamma_rv.pdf(x_vals)
            ax.plot(x_vals, gamma_p, color='k', alpha=0.7, label='Gamma MLE')

        if normal:
            normal_rv = stats.norm(loc=self.mean_mle, scale=self.std_mle)
            normal_p = normal_rv.pdf(x_vals)
            ax.plot(x_vals, normal_p, color='b', label='Normal MLE', alpha=0.6)

        ax.set_ylabel('Probability Density')
        ax.legend()

        if not xlim is None:
            ax.set_xlim(*xlim)

        if not ylim is None:
            ax.set_ylim(*ylim)

        return ax


def plot_kde(df, col):
    """Fit a Gaussian KDE to input data, plot fit over histogram of the data.

    Parameters
    ----------
    df: Pandas DataFrame

    col: str
        Column from df with numeric data to be plotted

    Returns
    -------
    ax: Matplotlib axis object
    """
    ax = plot_hist_basic(df, col)
    data = df[col]
    density = stats.kde.gaussian_kde(data)
    x_vals = np.linspace(data.min(), data.max(), 100)
    kde_vals = density(x_vals)

    ax.plot(x_vals, kde_vals, 'b-')

    return ax

def basic_ttest(df_column1,df_column2):
    #stats.ttest_ind(df_signed_in.CTR, df_not_signed_in.CTR, equal_var=False)
    stats.ttest_ind(df_column1, df_column2, equal_var=False)


def plot_t_test(group_1_df, group_2_df, group_1_name, group_2_name):

    fig = plt.figure()
    group_1_mean = group_1_df['CTR'].mean()
    group_2_mean = group_2_df['CTR'].mean()

    print '%s Mean CTR: %s' % (group_1_name, group_1_mean)
    print '%s Mean CTR: %s' % (group_2_name, group_2_mean)
    print 'diff in mean:', abs(group_2_mean-group_1_mean)
    p_val = stats.ttest_ind(group_1_df['CTR'], group_2_df['CTR'], equal_var=False)[1]
    print 'p value is:', p_val

    group_1_df['CTR'].hist(normed=True, label=group_1_name, color='g', alpha=0.3)
    group_2_df['CTR'].hist(normed=True, label=group_2_name, color='r', alpha=0.3)
    plt.axvline(group_1_mean, color='r', alpha=0.6, lw=2)
    plt.axvline(group_2_mean, color='g', alpha=0.6, lw=2)

    plt.ylabel('Probability Density')
    plt.xlabel('CTR')
    plt.legend()
    plt.grid('off')
    plt.show()

plot_t_test(df_signed_in, df_not_signed_in, 'Signed In', 'Not Signed In')


def bin_group_example():
    df_signed_in['age_groups'] = pd.cut(df_signed_in['Age'],
                                    [7, 18, 24, 34, 44, 54, 64, 1000],
                                    include_lowest=True)

    df_signed_in['age_groups'].value_counts().sort_index().plot(kind='bar',
                                                                grid=False)
    plt.xlabel('Age Group')
    plt.ylabel('Number of users')
    plt.tight_layout()
    plt.show()


def bin_group_example2():
    #calc low, med, high from data, and then add a new column
    mt_cars['hp_cat'] = pd.cut(mt_cars.hp,[low,med,high],labels=['low','high'],include_lowest=True)
    mt_cars.groupby('hp_cat').count()

def combo_pvalue_example():
    results = pd.DataFrame()
    combos = combinations(pd.unique(df_signed_in['age_groups']), 2)
    for age_group_1, age_group_2 in combos:
        age_group_1_ctr = df_signed_in[df_signed_in.age_groups == age_group_1]['CTR']
        age_group_2_ctr = df_signed_in[df_signed_in.age_groups == age_group_2]['CTR']
        p_value = stats.ttest_ind(age_group_1_ctr, age_group_2_ctr, equal_var=True)[1]
        age_group_1_ctr_mean = age_group_1_ctr.mean()
        age_group_2_ctr_mean = age_group_2_ctr.mean()
        diff = age_group_1_ctr_mean-age_group_2_ctr_mean
        absolute_diff = abs(age_group_1_ctr_mean-age_group_2_ctr_mean)
        results = results.append({
                  'first_age_group':age_group_1, 'second_age_group':age_group_2,
                  'first_group_mean':age_group_1_ctr_mean, 'second_group_mean':age_group_2_ctr_mean,
                  'mean_diff':diff, 'absolute_mean_diff':absolute_diff, 'p_value':p_value},
                  ignore_index=True)

    results = results[['first_age_group', 'second_age_group',
                       'first_group_mean', 'second_group_mean',
                       'mean_diff', 'absolute_mean_diff', 'p_value']]
    results[results['p_value'] < alpha].sort_values('absolute_mean_diff', ascending=False)
    results[results['p_value'] < alpha].sort_values('p_value', ascending=False)
    results[results['p_value'] > alpha].sort_values('absolute_mean_diff', ascending=False)
    results[results['p_value'] > alpha].sort_values('p_value', ascending=False)





# 4.
def find_mismatch(ab_cell, landing_page_cell):
    """Checks if A/B test treatment/control encoding is accurate

    Parameters
    ----------
    ab_cell: str
        Treatment/Control encoding

    landing_page_cell: str
        Page version

    Returns
    -------
    int: indicator of match (1) or mismatch (0)
    """

    if ab_cell == 'treatment' and landing_page_cell == 'new_page':
        return 0
    elif ab_cell == 'control' and landing_page_cell == 'old_page':
        return 0
    else:
        return 1


# 5.
def calc_conversion_rate(data, page_type):
    """Counts proportion of tatal visits resulting in a conversion

    Parameters
    ----------
    data: Pandas DataFrame
        A/B testing storage DataFrame with columns 'converted' (1=yes, 0=no)
        and 'landing_page' with values "new_page" or "old_page"

    page_type: str ("new" or "old")
        corresponding to the "new_page"/"old_page"

    Returns
    -------
    float: proportion of total visits converted for input page_type
    """

    total_vis = data[data['landing_page'] == page_type + '_page']
    converted = total_vis[total_vis['converted'] == 1].shape[0]
    return float(converted) / total_vis.shape[0], total_vis.shape[0]


# 8.
def plot_pval(data):
    """plots p-value based on hourly testing of running A/B test

    Parameters
    ----------
    data: Pandas DataFrame
        A/B testing storage DataFrame with columns 'hour' converted' and 'landing_page'

    Returns
    -------
    None: A plot is produced buy no axis object is returned
    """

    pval_lst = []
    datetime = data.ts.astype('datetime64[s]')
    hour = datetime.apply(lambda x: x.hour)
    data['hour'] = hour
    # Run the test as the data accumulates hourly
    for hr in hour.unique():
        hr_data = data[data['hour'] <= hr]
        # data for old landing page
        old = hr_data[hr_data['landing_page'] == 'old_page']['converted']
        old_nrow = old.shape[0]
        old_conversion = old.mean()
        # data for new landing page
        new = hr_data[hr_data['landing_page'] == 'new_page']['converted']
        new_nrow = new.shape[0]
        new_conversion = new.mean()
        # Run the z-test
        p_val = z_test(old_conversion, new_conversion,
                       old_nrow, new_nrow, effect_size=0.001,
                       two_tailed=True, alpha=.05)
        pval_lst.append(p_val[1])

    # Make the plot
    plt.plot(pval_lst, marker='o')
    plt.ylabel('p-value', fontweight='bold', fontsize=14)
    plt.xlabel('Hour in the day', fontweight='bold', fontsize=14)
    plt.axhline(0.05, linestyle='--', color='r')


# 9.
def read_country_and_merge(data, filename):
    """Reads in csv file with 'country' and 'user_id' columns and merges with data

    Parameters
    ----------
    data: Pandas DataFrame
        A/B testing storage DataFrame with columns 'user_id'

    filename: str
        Path to csv file with columns named 'user_id' and 'country'

    Returns
    -------
    Pandas DataFrame: orginal input 'data' merged with the csv file at 'filename'
    """
    country = pd.read_csv(filename)
    merged_df = pd.merge(data, country, left_on='user_id',
                         right_on='user_id', how='left')
    merged_df['country'] = merged_df['country'].map(str)
    return merged_df


def run_test(data):
    """Does an A/B test based on 'calc_conversion_rate' and 'z_test.z_test'

    Parameters
    ----------
    data: Pandas DataFrame
        A/B testing storage DataFrame with columns 'converted' (1=yes, 0=no)
        and 'landing_page' with values "new_page" or "old_page"

    Returns
    -------
    tuple: p_val, old_conversion, new_conversion
        The p-value from an A/B test along with the A and B conversion rates
    """

    new_convert, new_nrow = calc_conversion_rate(data, 'new')
    old_conversion, new_nrow = calc_conversion_rate(data, 'old')
    #alpha needs to be reduced to account for 4 countries
    modified_alpha = 0.05 / 4
    p_val = z_test(old_conversion, new_conversion,
       old_nrow, new_nrow, effect_size = 0.001, alpha = modified_alpha)[1]

    return p_val, old_conversion, new_conversion


def run_country_test(data):
    """Runs a separate A/B test based on 'run_test' for each country

    Parameters
    ----------
    data: Pandas DataFrame
        A/B testing storage DataFrame with columns
        'converted',  'landing_page', and 'country'

    Returns
    -------
    None: Conversion rate difference (new-old) and
        test p-value are printed for each country
    """
    results = {}
    for country in data['country'].unique():
        country_df = data[data['country'] == country]
        p_val, old_conversion, new_conversion = run_test(country_df)
        results[country] = [p_val, new_conversion - old_conversion]

    for country, lst in results.iteritems():
        p_val, conversion_diff = lst
        print '%s | conversion increase: %s | p_val: %s' % (country, conversion_diff, p_val)



def z_test(ctr_old, ctr_new, nobs_old, nobs_new,
           effect_size=0., two_tailed=True, alpha=.05):
    """Perform z-test to compare two proprtions (e.g., click through rates (ctr)).

        Note: if you set two_tailed=False, z_test assumes H_A is that the effect is
        non-negative, so the p-value is computed based on the weight in the upper tail.

        Arguments:
            ctr_old (float):    baseline proportion (ctr)
            ctr_new (float):    new proportion
            nobs_old (int):     number of observations in baseline sample
            nobs_new (int):     number of observations in new sample
            effect_size (float):    size of effect
            two_tailed (bool):  True to use two-tailed test; False to use one-sided test
                                where alternative hypothesis if that effect_size is non-negative
            alpha (float):      significance level

        Returns:
            z-score, p-value, and whether to reject the null hypothesis
    """
    conversion = (ctr_old * nobs_old + ctr_new * nobs_new) / \
                 (nobs_old + nobs_new)

    se = sqrt(conversion * (1 - conversion) * (1 / nobs_old + 1 / nobs_new))

    z_score = (ctr_new - ctr_old - effect_size) / se

    if two_tailed:
        p_val = (1 - stat.norm.cdf(abs(z_score))) * 2
    else:
        # H_A is examining if estimated effect_size > hypothesized effect_size
        p_val = 1 - stat.norm.cdf(z_score)

    reject_null = p_val < alpha
    print 'z-score: %s, p-value: %s, reject null: %s' % (z_score, p_val, reject_null)
    return z_score, p_val, reject_null



2. Scatter plot of `age` against `expense`

   ```python
   plt.scatter(df['age'], df['expense'], edgecolor='none', alpha=0.2)
   plt.ylim([0, 2000])
   plt.xlabel('age')
   plt.ylabel('expense')
   slope, intercept = np.polyfit(df['age'], df3['expense'], 1)
   plt.plot(df['age'], df['age'] * slope + intercept, alpha=0.5, color='r')
   ```

   ![qu1](imgs/qu_1.png)

3. Histogram of `expense`

   ```python
   # Using the regular matplotlib api
   plt.hist(df['expense'].tolist(), bins=15, normed=True, edgecolor='none')
   # Using the pandas plot api
   # df['expense'].hist(bins=15, normed=True, edgecolor='none')
   plt.xlabel('expense')
   plt.ylabel('frequency')
   ```

   ![qu2](imgs/qu_2.png)

4. Barplot of male/female ratio

   ```python
   # Getting percentage of male and female
   male_num = df[df['gender'] == 'M'].shape[0]
   female_num = df[df['gender'] == 'F'].shape[0]
   male_percent = male_num / ((male_num + female_num) * 1.)
   female_percent = female_num / ((female_num + female_num) * 1.)
   ```

   ```python
   # Getting the percentage of male and female for expense over 800
   over800_df = df[df['expense'] > 800]
   male_over800_num = over800_df[over800_df['gender'] == 'M'].shape[0]
   female_over800_num = over800_df[over800_df['gender'] == 'F'].shape[0]
   male_over800_percent = \
     male_over800_num / ((male_over800_num + female_over800_num) * 1.)
   female_over800_percent = \
     female_over800_num / ((male_over800_num + female_over800_num) * 1.)
   ```

   ```python
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
   #Plotting bar chart for overall female/male percent
   ax1.bar(range(2), [male_percent, female_percent],
        width=.2, color=['b', 'r'], alpha=.3, align='center')
   ax1.set_xticks(range(2))
   ax1.set_xticklabels(['M', 'F'])
   ax1.set_ylabel('Percentage of Male/Female')
   # Plotting bar chart for overall female/male percent over 800 expense
   ax2.bar(range(2), [male_over800_percent, female_over800_percent],
         width=.2, color=['b', 'r'], alpha=.3, align='center')
   ax2.set_xticks(range(2))
   ax2.set_xticklabels(['M', 'F'])
   ax2.set_ylabel('Percentage of Male/Female')



import sys


def make_draws(distribution, parameters, size=200):
    """
    - distribution(STR) [specify which distribution to draw from]
    - parameters(DICT) [dictionary with different params depending]
    - size(INT) [the number of values we draw]
    """
    if distribution == 'uniform':
        a, b = parameters['a'], parameters['b']
        values = scs.uniform(a, b).rvs(size)

    elif distribution == 'poisson':
        lam = parameters['lam']
        values = scs.poisson(lam).rvs(size)

    elif distribution == 'binomial':
        n, p = parameters['n'], parameters['p']
        values = scs.binom(n, p).rvs(size)

    elif distribution == 'exponential':
        lam = parameters['lam']
        values = scs.expon(lam).rvs(size)

    elif distribution == 'geometric':
        p = parameters['p']
        values = scs.geom(p).rvs(size)

    return values


def plot_means(distribution, parameters, size=200, repeats=5000):
    """
    - distribution(STR) [specify which distribution to draw from]
    - parameters(DICT) [dictionary with different params depending]
    - size(INT) [the number of values we draw]
    - repeat(INT) [the times we draw values]
    """
    mean_vals = []
    for _ in range(repeats):
        values = make_draws(distribution, parameters, size=200)
        mean_vals.append(np.mean(values))

    d_xlabel = {'uniform': 'Mean of randomly drawn values from a uniform',
                'poisson': 'Mean events happening in an interval',
                'binomial': 'Mean number of success',
                'exponential': 'Mean of waiting time before an event happens',
                'geometric': 'Mean rounds of failures before a success'
                }
    xlabel = d_xlabel[distribution]
    plt.hist(mean_vals, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title('Mean of %s samples with size %s drawn from %s distribution' %
              (repeats, size, distribution.capitalize()), fontsize=14)
    plt.show()

#if __name__ == '__main__':
plot_means('uniform', {'a': 10, 'b': 20})
plot_means('poisson', {'lam': 2})
plot_means('binomial', {'n': 100, 'p': 0.1})
plot_means('exponential', {'lam': 2})
plot_means('geometric', {'p': 0.1})



'''
##Part 1: Identifying Distributions

1. A typist makes on average 2 mistakes per page.

   What is the probability of a particular page having no errors on it?

   ```
   X ~ Poisson(2)

   P(X = 0)
   = (2 ^ 0 / 0!) * e ^ -2
   = 0.135
   ```


2. Components are packed in boxes of 20. The probability of a component being
   defective is 0.1.

   What is the probability of a box containing 2 defective components?

   ```
   X ~ Binomial(20, 0.1)

   P(X = 2)
   # 20 C 2 is 20 choose 2
   = (20 C 2) * (0.1 ^ 2) * (1 - 0.1) ^ 18
   = (20! / 2! * 18!) * (0.1 ^ 2) * (1 - 0.1) ^ 18
   = 0.285
   ```

3. Patrons arrive at a local bar at a mean rate of 30 per hour.

   What is the probability that the bouncer has to wait more than 3 minutes to card the
   next patron?

   ```
   X = waiting time
   X ~ Exponential(0.5)  # 30 per hour = 0.5 per minute

   P(X > 3)
   = 1 - P(X <= 3)
   = 1 - CDF
   = exp(-0.5 * 3)
   = 0.2231
   ```

4. A variable is normally distributed with a mean of 120 and a standard
   deviation of 5. One score is randomly sampled.

   What is the probability the score is above 127?

   (_**Hint: Define new random variable**_ `Z`, `P(X > 127) = P(Z > ?)`. Refer
   to `standard_normal_table.pdf` to convert `Z` to probability.)

   ```
   X ~ Normal(120, 5)

   P(X > 127)
   = 1 - P(X <= 127)        Z = (127 - 120) / 5
   = 1 - P(Z <= 1.4))
   = 0.081
   ```

5. You need to find a tall person, at least 6 feet tall, to help you reach
   a cookie jar. 8% of the population is 6 feet or taller.

   If you wait on the sidewalk, how many people would you expect to have passed
   you by before you'd have a candidate to reach the jar?

   ```
   X ~ Geometric(p = 0.08)

   E(X)
   = 1 / 0.08
   = 12.5
   # Note: the reason 1/p is because 1/p is equal to the mean of the Geometric Distribution.
   ```

6. A harried passenger will miss by several minutes the scheduled 10 A.M.
   flight to NYC. Nevertheless, he might still make the flight, since boarding
   is always allowed until 10:10 A.M., and extended boarding is sometimes
   permitted as long as 20 minutes after that time.

   Assuming the extended boarding time is **uniformly distributed** over the above
   limits, find the probability that the passenger will make his flight,
   assuming he arrives at the boarding gate at 10:25.

   ```
   X ~ Uniform(10, 30)

   P(X > 25)
   = (30 - 25) / (30 - 10) = 0.25
   ```

##Part 2: Covariance and Joint Distribution

1. Read in data

   ```python
   df = pd.read_csv('data/admissions.csv')
   ```

2. Covariance Function

   ```python
   def covariance(x1, x2):
       return np.sum((x1 - np.mean(x1)) * (x2 - np.mean(x2))) / (len(x1) - 1.)

   print covariance(df.family_income, df.gpa)
   print covariance(df.family_income, df.parent_avg_age)
   print covariance(df.gpa, df.parent_avg_age)

   #Check your results
   df.cov()
   ```

3. Normalizing covariance to compute correlation

   ```python
   def correlation(x1, x2):
       std_prod = np.std(x1) * np.std(x2)
       covar = covariance(x1, x2)
       return covar / std_prod

   print correlation(df.family_income, df.gpa)
   print correlation(df.family_income, df.parent_avg_age)
   print correlation(df.gpa, df.parent_avg_age)

   #Check your results
   df.corr()
   ```

4. Computing the GPA threshold for each income class

   ```python
   # Categorize the family income
   def income_category(income):
       if income <= 26832:
           return 'low'
       elif income <= 37510:
           return 'medium'
       else:
           return 'high'
   # Apply the categorization and define a new column
   df['family_income_cat'] = df.family_income.apply(income_category)

   # Alternatively, we can use pandas' cut function to bin the data
   max_income = df['family_income'].max()
   df['family_income_cat']=pd.cut(np.array(df['family_income']), [0,26382,37510,max_income],
                                         labels=["low","medium","high"])
   ```

   ```python
   # Get the conditional distribution of GPA given an income class
   low_income_gpa = df[df['family_income_cat'] == 'low'].gpa
   medium_income_gpa = df[df['family_income_cat'] == 'medium'].gpa
   high_income_gpa = df[df['family_income_cat'] == 'high'].gpa

   # Plot the distributions
   from scipy.stats.kde import gaussian_kde
   def plot_smooth(gpa_samp, label):
       my_pdf = gaussian_kde(gpa_samp)
       x = np.linspace(min(gpa_samp) , max(gpa_samp))
       plt.plot(x, my_pdf(x), label=label)

   fig = plt.figure(figsize=(12, 5))
   plot_smooth(low_income_gpa, 'low income')
   plot_smooth(medium_income_gpa, 'medium income')
   plot_smooth(high_income_gpa, 'high income')
   plt.xlabel('GPA', fontsize=14, fontweight='bold')
   plt.ylabel('Probability Density', fontsize=14, fontweight='bold')
   plt.legend()
   ```

5. 90th percentile GPA for each income class

   ```python
   # The 90th percentile GPA for each class
   print '90th percentile GPA for low income class', np.percentile(low_income_gpa, 90)
   print '90th percentile GPA for medium income class', np.percentile(medium_income_gpa, 90)
   print '90th percentile GPA for high income class', np.percentile(high_income_gpa, 90)
   ```

##Part 3: Pearson Correlation vs Spearman Correlation

1. Load in the new data

   ```python
   df2 = pd.read_csv('data/admissions_with_study_hrs_and_sports.csv')
   ```

2. Make a scatter plot of GPA against Hours Studied

   ```python
   plt.scatter(df2['gpa'], df2['hrs_studied'], alpha=.01, edgecolor='none')
   slope, intercept, r_value, p_value, std_err = sc.linregress(df2['gpa'], df2['hrs_studied'])
   plt.plot(df2['gpa'], slope * df2['gpa'] + intercept, color='r', alpha=.4)
   plt.xlabel('GPA', fontsize=14, fontweight='bold')
   plt.ylabel('Hours Studied', fontsize=14, fontweight='bold')
   ```

   ![image](imgs/scatter_hr_gpa.png)

3. Correlation between `gpa` and `hrs_studied`

   ```python
   print sc.pearsonr(df2['gpa'], df2['hrs_studied'])
   print sc.spearmanr(df2['gpa'], df2['hrs_studied'])
   # The spearman correlation shows a more positive coefficient since it captures the non-linear relationship
   ```

4. Correlation between `gpa` and `hrs_studied`

   ```python
   print sc.pearsonr(df2['gpa'], df2['sport_performance'])
   print sc.spearmanr(df2['gpa'], df2['sport_performance'])
   # There is s strong relationship between gpa and sports perf. , but the values of the
   # two variables are not monotonically increasing together. Therefore, the coefficients are low
   ```

##Part 4: Distribution Simulation

1. Define the distributions

   ```python
   # Define number of sales to be a uniform from 5000 to 6000
   sales = sc.uniform(5000, 1000)
   # Define conversion percent as a binomial distribution
   conversion = sc.binom
   # Profit PMF
   profit_ = sc.binom
   #OLD, WRONG
   def profit_rvs():
       if random.random() >= 0.2:
           return 50
       else:
           return 60
   ```

2. Simulate the profit distribution and plot it

   ```python
   def simulate_sales():
       sales_draw = sales.rvs()
       conversion_draw = conversion(sales_draw, 0.12).rvs()
       wholesale_proportion = profit_(conversion_draw, .2).rvs()
       profit = conversion_draw * wholesale_proportion * 50
       return  profit + (conversion_draw-wholesale_proportion)*60

   dist = [simulate_sales() for _ in range(10000)]
   plt.hist(dist)
   plt.xlabel('Profit', fontsize=14, fontweight='bold')
   plt.ylabel('Freq.', fontsize=14, fontweight='bold')
   print '2.5% percentile', np.percentile(dist, 2.5)
   print '97.5% percentile', np.percentile(dist, 97.5)
   # 2.5% percentile 33750.0
   # 97.5% percentile 42930.0
   ```

   ![image](imgs/profit_hist.png)

##Extra Credit: More Probability Exercises

##Set

1. Out of the students in a class, 60% are geniuses, 70% love chocolate,
   and 40% fall into both categories. Determine the probability that a
   randomly selected student is neither a genius nor a chocolate lover.

   ```
   Draw a Venn diagram

    - Probability of both Genius and Chocolate lover is 40%
    - Probability of Genius but not Chocolate lover is 20%
    - Probability of Chocolate lover and not Genius is 30%.

   100% - (40% + 20% + 30%) = 10%
   ```
##Combinatorics

1. A fair 6-sided die is rolled three times, independently. Which is more likely: a
   sum of 11 or a sum of 12?

   ```
   There are 216 possibilities.  We simply enumerate and count.

    * Sum of 11:
      * (1,4,6), (1,5,5) | (2,3,6), (2,4,5) |  (3,3,5), (3,4,4);
      * 6 + 3 + 6 + 6 + 3 + 3 = 27

    * Sum of 12:
      * (1,5,6) | (2,4,6), (2,5,5) | (3,3,6), (3,4,5) | (4,4,4)
      * 6 + 6 + 3 + 3 + 6 + 1 = 25

    ==> So it is more likely to roll an 11.
    ```


2. 90 students are to be split at random into 3 classes of equal size. Joe and Jane are
   two of the students. What is the probability that they end up in the same
   class?

   ```
   There a few approaches, all equal to 0.3258
    * Approach 1:
       (Think given the class of jane, probability joe is in same class) = 29/89

    * Approach 2:
       88_choose_28 / 89_choose_29

    * Approach 3:
       3 x P(both in class i) = 3 * 88_choose_28 / 90_choose_30
   ```

3. A well-shuffled 52-card deck is dealt to 4 players. Find the probability that
   each of the players gets an ace.

   ```
   (Assume ordering, factor by num. permutations)

    4! * (48_choose_12 x 36_choose_12 x 24_choose_12 /
    (52_choose_13  x  39_choose_13 x 26_choose_13)
   ```

##Random Variable


1. A six-sided die is loaded in a way that each even face is twice as likely as
   each odd face. All even faces are equally likely, as are all odd faces.

   Construct a probabilistic model for a single roll of this die and find the
   probability that the outcome is less than 4.

   ```
   # Set up a probability mass function (PMF) and compute probability

   Sample Space = 1, 2, 3, 4, 5, 6

   P(1) = P(3) = P(5) = k
   P(2)= P(4) = P(6) = 2k

   3k + 6k = 1
   k = 1/9

   P(X < 4)
   = P(1) + P(2) + P(3)
   = 1/9 + 2/9 + 1/9
   = 4/9
   ```

2. `X` is a random variable with PMF `p(k) = k^2/a` if `k = -3, -2, -1, 0, 1, 2, 3`
   and `p(k) = 0` otherwise.

  (a) Find `a` and `E[X]`

  (b) Find the expectation value of the random variable `Z = (X - E[X])^2)`

  (c) Using the result from (b), find the variance of `X`

  ```
  (a) Find `a` and `E[X]`
  (b) Find the PMF of the random variable `Z = (X - E[X])^2)`

      First, notice that E(X) = 0 by symmetry

      Let Z = X^2

      P(Z = 1)
      = P(X = -1) + P(X = 1) = 2/a

      P(Z = 4)
      = P(X = -2) + P(X = 2) = 8/a

      P(Z = 9)
      = P(X = -3) + P(X = 3) = 18/a

      28/a = 1
      a = 28

  (c) Find the variance of `X` (use the result from above)

      Var(X)
      = E[X^2] - E(X)^2
      = E(Z)
      = (2/28) * 1 + (8/28) * 4 + (18/28) * 9
      = 7
  ```


3. A soccer team has 2 games scheduled for one weekend. It has a 0.4 probability
   of not losing first game and 0.7 probability of not losing the second
   game, independent of the first. If the team wins a particular game, the
   team is equally likely to win or tie. The team will receive 2 points for a win,
   1 for a tie, and 0 for a loss.
  (a) Find the PMF of the number of points that the team earns over the
   weekend.
  (b) Find the expected value for the number of points earned.
  (c) Find the variance for the number of points earned.

  ```
   * a. Find the PMF of the number of points that the team earns over the weekend.

    P(lose first) = 0.6 --> P(tie first)=P(win first)=0.2
    P(lose second) = 0.3 --> P(tie second)=P(win second)=0.35


    P(X=0) = 0.18
    P(X=1) = 0.27
    P(X=2) = 0.34 = P(LW or TT or WL) = 0.6 x 0.35 + 0.2 x 0.35 + 0.2 x 0.3
    P(X=3) = 0.14
    P(X=4) = 0.07

  * b. Find the expected value for the number of points earned.
    E(X) = 0.18 x 0 + 0.27 x 1 + ... + 0.07 x 4 = 1.65


  * c. Find the variance for the number of points earned.
    E(X^2) = 0.18 x 0 + 0.27 x 1 + ... + 0.07 x 16 = 4.01
    V(X) = E(X^2) - E(X)^2 = 4.01-1.65^2 = 1.2875
  ```


4. You have 5 keys, one of which is the correct one for the house. Find the PMF
   of the number of trials you will need to open the door, assuming that after
   you try a key that doesn't work you set it aside and you otherwise randomly
   select a key to try next.

   ```
   * P(1) = 1/5
   * P(2) = (4/5) * (1/4) = 1/5
   * P(3) = (4/5) * (3/4) * (1/3) = 1/5
   * P(4) = 1/5
   * P(5) = 1/5

   Another approach, logical reasoning:
   Imagine 5 slots [---]  [---]  [---]  [---]  [---]
   Equal chance in any of those slots.  If check one after another,
   chance it will take exactly 1 time, 2 times, 3 times, 4 times, 5 times are 1/5 each.
   ```

5. You toss independently a fair coin and you count the number of tosses until
   the first tail appears. If this number is `n`, you receive `2^n` dollars.
   What is the expected amount you will receive? How much would you be willing
   to play this game?

   ```
   Let X = # tosses until first tail.
   X ~ Geom(0.5), P(X = k) = 1/2^k, for all k = 1, 2, 3, ...
   W = winnings. P(W = 2^n) = 1/2^n for all n = 1, 2, 3, ...
   E(W) = sum [(2^n) P(W = 2^n)] = infinity
   In reality, no one has an infinite amount of money to play the game forever.
   ```


### Joint Distributions


1. A class of `n` students takes a test consisting of `m` questions. Suppose that
   student `i` submitted answers to the first `m_i,` for `m_i <= m` questions.
  - The grader randomly picks one answer, call it `(I, J)` where `I` is the student
    ID number (values `1,...,n`) and `J` is the question number (values `1,...,m`).
    Assume that all answers are equally likely to be picked. Calculate the joint
    and marginal PMFs of `I` and `J`.
  - Assume that an answer to question `j` if submitted by student `i` is correct
    with probability `p_ij`. Each answer gets `a` points if it is correct and `b`
    points otherwise. Find the expected value of the score of student `i`.

  ```
  * The grader randomly picks one answer,
    call it (I, J) where I is the student ID number (values 1,...,n)
    and J is the question number (values 1,...,m).

  Assume that all answers are equally likely to be picked.
  Calculate the joint and marginal PMFs of I and J.

  P(I=i) = m_i/sum(m_j over all j)
  (Simply number of questions answered by student i over total number of questions answered)

  P(J=j) = sum(Indicator(m_i>=j))/sum(m_j over all j)
  (Probability of selecting a particular question is the total number
   of students who answered at least that many questions,
   divided by total number of questions answered.)

  P(I=i, J=j) = 1/sum(m_j over all j)


  * Assume that an answer to question j if submitted by student i is correct
    with probability p.ij. Each answer gets a points if it is correct and b
    points otherwise. Find the expected value of the score of student i.

    E(score for student i) = sum( a x p_ij + b x (1-p_ij) over all j)
  ```


### Independence

1. Alice and Bob want to choose between the opera and the movies by tossing a
   coin. Unfortunately, the only coin they have is biased, though the bias is
   not known exactly. How can they use the biased coin to make a decision so
   that either option (opera or movies) is equally likely to be chosen?

   ```
   (symmetry)

   Flip the coin in pairs.  If HT comes up first, go to opera.
   If TH comes up first, go to movies.

   Experiment might look like HH, TT, HH, HH, HH HH, HT --> opera
   ```

2. A system consists of `n` identical components, each of which is operational
   with probability `p`, independent of other components. The system is
   operational if at least `k` out of the `n` components are operational. What is
   the probability that the system is operational?

   ```
   Let X ~ Binomial(n,p)

   Then simply plug in P(X >= k) = P(k)+P(k+1)+...+P(n)
   ```

### Covariance and Correlation

1. Suppose that *X* and *Y* are random variables with the same variance. Show
   that *X - Y* and *X + Y* are uncorrelated.

   ```
   cov(X-Y,X+Y) = E[ ((X-Y) - E(X-Y)) ((X+Y) - E(X+Y)) ]

   cov(X-Y, X+Y) = 0

   Correlation is just covariance scaled, so X-Y and X+Y are uncorrelated
   ```

'''
def import_R_module():
    from rpy2.robjects.packages import importr
    pwr = importr('pwr')
    result = pwr.pwr_anova_test(k=5,f=.25,sig_level=.05,power=.8)
    print str(result)


def R_write_to_file():
    #creates file mtcars.R
    %%writefile mtcars.R
    #!/usr/bin/Rscript
    mtcars <- read.csv("mtcars_adj.csv")
    #factor <- mtcars$hp_cat
    #response <- mtcars$mpg
    #run summary of anova_test
    results = summary(aov(mtcars$LpCK ~ mtcars$hp_cat,data=mtcars))
    print(results)
    #print out a plot.  first assign file, then fill it
    png("mtcars_plot.png")
    plot(mtcars$LpCK ~ mtcars$hp_cat,data=mtcars)
    dev.off()

def R_read_file():
    !Rscript mtcars.R

def R_display_plot_in_python():
    from IPython.display import Image
    Image('mtcars_plot.png')
def R_write_csv(dataset, filename):
    #code within R to run to assign data to file
    #write.csv(dataset, "filename.csv")
    write.csv(dataset, filename)

def R_power_analysis_html():
    '''
    Power Analysis in R
    https://www.statmethods.net/stats/power.html
    one-way ANOVA analysis
    http://www.stat.columbia.edu/~martin/W2024/R3.pdf
    '''
def R_script_file_handle_function():
    '''
    #!/usr/bin/Rscript

    ## prep in and out files
    infileName <- "infile.csv"
    infilePath <- paste(getwd(),infileName,sep="/")
    inData <- read.table(infilePath,header=T,sep=",")
    header <- names(inData)

    outfileName <- "outfile.csv"
    newHeader <- paste('group','value','date','time','daysFrom',sep=",")
    cat(newHeader,file=outfileName,append=F,fill=T)

    ## create a function to do something to the file
    dayOne <- as.Date("1979-11-19")
    process_file <- function(x, outfile) {
        group <- x[1]
        value <- x[2]
        date  <- x[3]
        time  <- x[4]
        daysFrom <- as.Date(date) - dayOne
        rowToOutput <- paste(group, value, date, time, daysFrom, sep=",")
        print(paste(group, value, date, time, sep=","))
        cat(rowToOutput,file=outfile,append=T,fill=T)
    }

    apply(inData, 1, process_file,outfile=outfileName)

    ## read it in again just to be sure we made it correctly
    x <- read.csv(outfileName)
    attach(x)
    print(names(x))
    print(mean(x$value))
    '''

def terminal_track_operations():
    '''
    bg      will show start of what is running in the background
    fg      will put you into that process.  you can use ctrl-d or ctrl-c to stop it properly
    ps      lists processes that are running
    kill pid    shuts down process by id
    pkill jupyter-noteboo   shuts down all processes by application
    '''

def gpu_nvidia_example():
    '''
    lspci command shows software
    look for NVIDIA
    '''


def bayesian_pyMC3():
    #from pymc3 import DiscreteUniform
    from pymc3 import Uniform
    with Model() as disaster_model:
    #    switchpoint = DiscreteUniform('switchpoint', lower=0, upper=n_years)
        switchpoint = Uniform('switchpoint', lower=0, upper=n_years)
        from pymc3 import Exponential
        early_mean = Exponential('early_mean', 1)
        late_mean = Exponential('late_mean', 1)
        from pymc3.math import switch
        rate = switch(switchpoint >= np.arange(n_years), early_mean, late_mean)
        from pymc3 import Poisson
        disasters = Poisson('disasters', mu=rate, observed=disasters_data)
        from pymc3 import Deterministic
        rate = Deterministic('rate', switch(switchpoint >= np.arange(n_years), early_mean, late_mean))
        trace = sample(1000)
        traceplot(trace);

def pyMC3():
    '''
    to look at the underlying data
    '''
    trace('switchpoint')
    trace('early_mean')

import numpy as np
import pandas as pd

#%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline, FeatureUnion


def simple_linear_regression(filename):
    import numpy as np
    import pandas as pd

    #%matplotlib inline
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.pipeline import Pipeline, FeatureUnion

    data = pd.read_csv(filename)
    i_reg = LinearRegression()
    x_data = data[['colA','colB','colC']]
    y_data = data['target']
    i_reg.fit(x_data,y_data) #this actually creates the mnodel
    i_reg.intercept_
    i_reg.coef_ #list of coefficients
    data_predictions = i_reg.predict(x_data) #can use against training data or testing dataset


    #plotting regression
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].scatter(data.colA, data.target, s=40)
    axs[0].set_title("Actual Data")
    axs[0].set_xlabel("ColA")
    axs[0].set_ylabel("Actual target data")


    axs[1].scatter(insects.latitude, wing_size_predictions, s=40)
    axs[1].set_title("Predicted Data")
    axs[1].set_xlabel("ColA")
    axs[1].set_ylabel("Predicted target data");

def simple_logistic_regression(filename):
    wells = pd.read_csv('~/galvanize/linear-regression/warm-up/data/wells.dat', sep=' ')
    wells_regression = LogisticRegression()
    # We don't need the i'd column, so drop it.
    X_wells_names = np.array(['arsenic', 'dist', 'assoc', 'educ'])
    X_wells = wells[X_wells_names] #pass only names we care about, not ID
    # The response is already encoded as 0's, and 1's.
    y_wells = wells['switch']
    wells_regression.fit(X_wells, y_wells)
    wells_predictions = wells_regression.predict(X_wells)
    print(wells_predictions[:10])
    #this produces a 2 column array that denotes the probability of [[failure success]]
    wells_probabilities = wells_predictions.predict_proba(x_wells)

    #plotting results
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    axs[0].scatter(wells.dist, wells.switch, s=40, alpha=0.33)
    axs[0].set_title("Actual Data")
    axs[0].set_xlabel("Distance to Safe Well")
    axs[0].set_ylabel("Did Resident Switch?")


    axs[1].scatter(wells.dist, wells_probabilities[:, 1], s=40, alpha=0.33)
    axs[1].set_title("Predicted Data")
    axs[1].set_xlabel("Distance to Safe Well")
    axs[1].set_ylabel("Probability of Switching")

def regression_standard_scaler(x_data):
    # x_standardized = (x-xbar)/sd
    standardizer = StandardScaler()
    standarizer.fit(x_data)
    standardizer.transform(x_data)
    #then plug the standardized data back into regression
    for name, col in zip(X_wells_names, X_wells_standardized.T):
        print("Mean of column {}: {:2.2f}".format(name, col.mean()))
        print("Standard Deviation of column {}: {:2.2f}".format(name, col.std()))

def regression_KBest(x_wells, y_wells, x_names):
    best_3 = SelectKBest(chi2,k=3)
    best_3.fit(x_wells,y_wells)
    x_wells_best_3 = best_3.transform(x_wells)
    #can now use smaller column set in regression
    #get names of Columns
    x_name[best_3.get_support()]

class ColumnSelector(object):
    #selects specific columns by index
    def __init__(self, idxs):
        self.idxs = np.asarray(idxs)

    # Fit here doesn't need to do anything.  We already know the indices of the columns
    # we want to keep.
    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        # Need to treat pandas data frames and numpy arrays slightly differently.
        '''
        There are a few rules we need to follow:

        fit needs to be defined as either fit(self, *args, **kwargs) if we do not need to look at the data to fit the transformer, or fit(self, X, y, *args, **kwargs) if we do need to look at the data.
        fit needs to return self. This is a common oversight, and will case problems when using Pipeline below if forgotten.
        transform needs to be defined as transform(self, X, **transform_params), and returns the transformed data set.

        This process, of implementing certain methods under some constraints, is called coding to an interface. As long as it is done properly, it allows us to seamlessly use our objects inside of code that was designed to work with built in transformer objects.
        '''
        if isinstance(X_wells, pd.DataFrame):
            return X.iloc[:, self.idxs]
        return X[:, self.idxs]

def use_ColumnnSelector():
    #will select only the first column.  in this case just trying to understand how class works and applies
    column_selector = ColumnSelector([0])
    column_selector.fit()
    X_wells_column_selected = column_selector.transform(X_wells)
    print(X_wells_column_selected.shape)
    print(X_wells_names[column_selector.idxs])

def regression_pipeline_example(x_wells, y_wells):
    wells_pipeline = Pipeline([
        ('select_best_3', SelectKBest(chi2, k=3)),
        ('standardize', StandardScaler()),
        ('regression', LogisticRegression())
    ])
    #only need to fit once
    wells_pipeline.fit(x_wells,y_wells)
    #this will do all three processes on the dataset

    # The column means memorized by the pipeling.
    print(wells_pipeline.named_steps['standardize'].mean_)
    # The column standard deviations memorized by the pipeline.
    print(wells_pipeline.named_steps['standardize'].scale_)
    print(wells_pipeline.named_steps['regression'].coef_)

class PolynomialExpansion(object):
    ''' a transformer class that consumes a single column, and creates a matrix with the square, cube, etc of the column.
    '''
    def __init__(self, degree):
        self.degree = degree

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        # Initialize our return value as a matrix of all zeros.
        # We are going to overwrite all of these zeros in the code below.
        X_poly = np.zeros((X.shape[0], self.degree))
        # The first column in our transformed matrix is just the vector we started with.
        X_poly[:, 0] = X.squeeze()
        # Cleverness Alert:
        # We create the subsequent columns by multiplying the most recently created column
        # by X.  This creates the sequence X -> X^2 -> X^3 -> etc...
        for i in range(2, self.degree + 1):
            X_poly[:, i-1] = X_poly[:, i-2] * X.squeeze()
        return X_poly

def use_polynomialExpansion():
    X = np.array([[1], [2], [3], [4]])
    P = PolynomialExpansion(3)
    P.fit(X)
    P.transform(X)

def regression_pipeline_example2():
    p = Pipeline([
        ('latitude_selector', ColumnSelector([1])),
        ('quadratic_expansion', PolynomialExpansion(10)),
        ('quadratic_model', LinearRegression())
    ])
    p.fit(X_insects, y_insects)
    #because LinearRegression has prediction method, its passed on to pipeline
    predictions = p.prediction(X_insects)

def regression_feature_union():
    '''
    What if we want to create a polynomial expansion using two features in our model.

    To accomplish this, we would need to grab two different columns, take a polynomial transformation of them individually, and then re-join the results into a single matrix:

        +--- Select Column 1 --- Polynomial Expansion ---+
    X --+                                                +--- Rejoin --> X transfomed
        +--- Select Column 2 --- Polynomial Expansion ---+

    The splitting and rejoining operation can be accomplished with another sklean feature, the FeatureUnion.

    Here's a simple example:

        +--- Select Column 1 ---+
    X --+                       +--- Rejoin --> X transfomed
        +--- Select Column 2 ---+

    '''
    two_columns = FeatureUnion([
        ('arsenic_selector', ColumnSelector([0])),
        ('distance_selector', ColumnSelector([1]))
    ])
    two_columns.fit(X_wells)
    print(two_columns.transform(X_wells))
    #this does the same thing as calling ColumnSelector([0,1]), but opens up new possibiliteis
    '''
    Let's end by putting together the example I outlined above:

        +--- Select Column 1 --- Polynomial Expansion ---+
    X --+                                                +--- Rejoin --> X transfomed
        +--- Select Column 2 --- Polynomial Expansion ---+

    We will use polynomials of degree 2, and end the pipeline with a LogisticRegression.
    '''
    wells_pipeline = Pipeline([
        ('polynomial_expansions', FeatureUnion([
            ('arsenic_quadratic', Pipeline([
                ('arsenic_selector', ColumnSelector([0])),
                ('quadratic_expansion', PolynomialExpansion(2))
            ])),
            ('distance_quadratic', Pipeline([
                ('distance_selector', ColumnSelector([1])),
                ('quadratic_expansion', PolynomialExpansion(2))
            ]))
        ])),
        ('regression', LogisticRegression())
    ])
    #note that the featureUnion allows you to call the polynomial expansions together.
    #otherwise would leave columns behind.

def args_and_keywordargs():
    '''
    *args takes a list of arguments and passes them to the function.  can be variable number
    **kwargs takes a dictionary of arguments, passed by keyword.
    '''

def print_all(*args):
    for count, item in enumerate(args):
        print('{0}. {1}'.format(count,item))
print(print_all('earth','fire','water'))

def list_all(**kwargs):
    lst = []
    for name, data in kwargs.items:
        lst.append((name, data))

def assumptions_of_linear_models():
    '''
    1. Leanearity
    2.  Full args_and_keywordargs
    3.  Exogeneity
    4.
    5. no Covariance
    6. normally distributed error
    '''

def python_create_installable_module():
    '''
    great presentation on linear regression practical techniques
    ~/galvanize/linear-regression/lecturenotebooks/linear-regression-morning-lecture.ipynb

    I wrote a fair amount of my own code for this lecture, which I've packeged together into a couple of python libraries. You can install these libraries with the commands:

    pip install git+https://github.com/madrury/basis-expansions.git
    pip install git+https://github.com/madrury/regression-tools.git

    The Basis Expansions library contains classes used for creating cubic spline basis expansions. This will allow us to flexibly capture non-linear effects in our regression models.

    The Regression Tools library contains various tools for transforming pandas data frames, and plotting fit regression models.
    '''

def regression_process():
    '''
    from sklearn import LinearRegression
    .fit
    .predict
    .score
    .accuracy
    .number of data points
    .shape of
    .rsqd

    '''
