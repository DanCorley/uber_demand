import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from math import sqrt
import time

def rmse(y_real, y_hat):
    return sqrt(mean_squared_error(y_real, y_hat))

def fit_hw(train, test, season_periods=168, season = 'add',
           alpha = None, beta=None, gamma = None):
    '''
    vanilla model for holt_winters
    '''
    model = ExponentialSmoothing(train,
                  seasonal='add',
                  seasonal_periods=season_periods)
    
    results = model.fit(smoothing_level = alpha,
                        smoothing_slope = beta,
                        smoothing_seasonal = gamma,
                        use_boxcox=False)
    
    y_hat = results.predict(0)
    score = rmse(train, y_hat)
    
    return score


def get_base_score(train, test, num_test, run_test = ['alpha', 'beta', 'gamma']):
    '''
    return best values for smoothing parameters
    '''
    
    scores = {}

    for run in run_test:

        _ = {} 

        for param in np.linspace(0, 1, num_test):
            if run == 'alpha':
                score = fit_hw(train, test, alpha=param)
            elif run == 'beta':
                score = fit_hw(train, test, beta=param)
            else:
                score = fit_hw(train, test, gamma=param)

            _[param] = score

        scores[run] = _

    return scores

def plot_params(params, labels):
    num_param = len(params)
    fig, axes = plt.subplots(nrows=1, ncols=num_param,
                             figsize=(5*num_param, 3))
    for i, param in enumerate(params):
        param = params[param]
        axes[i].plot(pd.Series(param))
        axes[i].set_title(labels[i])
        fig.tight_layout()

def parameter_tune(train, test, params, num_tests, val_range = .15, verbose=False):
    '''
    tune smoothing parameters
    '''
    
    alpha, beta, gamma = [min(params[x], key=params[x].get) for x in params]
    
    best_score = min([min(params[x].values()) for x in params])
    
    # make sure there are no negative values
    a_smooth = np.linspace(alpha-val_range,alpha+val_range, num_tests)
    a_smooth = [x for x in a_smooth if x > 0]
    g_smooth = np.linspace(gamma-val_range,gamma+val_range, num_tests)
    g_smooth = [x for x in g_smooth if x > 0]
    
    # cycle through all results to get lowest rmse scores
    result = []
    train_len = len(train)-1
    for a in a_smooth:
        for g in g_smooth:
            model = ExponentialSmoothing(train,
                             seasonal='add',
                             seasonal_periods=168)
            results = model.fit(smoothing_level=a,
                                smoothing_seasonal=g)
            y_hat = results.predict(0, train_len)
            score = rmse(train, y_hat)
            result.append([a, g, score])
            if verbose:
                if score < best_score:
                    best_score = score
                    print(f'alpha:{a}, gamma:{g}, score:{score}')
                
    if best_score < min([x[2] for x in result]):
        df = pd.DataFrame([[None, None, best_score]], columns=['alpha', 'gamma', 'rmse'])
    else:
        df = pd.DataFrame(result, columns=['alpha', 'gamma', 'rmse'])
    return df


def run_final_model(train, test, results, validate=None):
    '''
    return the best performing model
    '''
    best = results.nsmallest(1, 'rmse')
    a_best = best.alpha.iloc[0]
    g_best = best.gamma.iloc[0]
    
    model = ExponentialSmoothing(train,
                                 seasonal='add',
                                 seasonal_periods=168)

    if type(a_best) == type(None):
        model = model.fit(smoothing_level=None,
                          smoothing_seasonal=None)
    else:
        model = model.fit(smoothing_level=a_best,
                          smoothing_seasonal=g_best)
    results = {}
    
    results['train'] = train
    results['train_pred'] = model.predict(start=0, end=-1)
    if type(validate) == pd.Series:
        predictions = model.forecast(144)
        results['validate'] = test
        results['val_forecast'] = predictions[:72]
        results['test'] = validate
        results['test_forecast'] = predictions[-72:]
        results['val_rmse'] = rmse(results['validate'], results['val_forecast'])
        results['test_rmse'] = rmse(results['test'],results['test_forecast'])
    else:
        results['test'] = test
        results['forecast'] = model.forecast(len(test))
        results['test_rmse'] = rmse(results['test'],results['forecast'])
    
    return results

def run_area_test(df, daterange, area, base = 15, tune= 10, season_periods=168):
    print(f'Starting with area: {area}')
    start = time.time()
    new_df = df[df['pickup_community_area'] == area]
    
    daterange = pd.DataFrame(daterange)

    new_df = daterange.merge(new_df, how='left',
                             left_on=0, right_on='date_time')

    new_df.fillna(1, inplace=True)

    new_df.set_index(0, inplace=True)

    val_split = '2019-10-03'
    test_split = '2019-10-06'
    end_split = '2019-10-09'
    ind = new_df.index

    train = new_df['rides'][ind < val_split].asfreq('h')
    validate = new_df['rides'][(ind >= val_split) & (ind < test_split)].asfreq('h')
    test = new_df['rides'][(ind >= test_split) & (ind < end_split)].asfreq('h')

    score = fit_hw(train, validate, season_periods=season_periods)

    params = get_base_score(train, validate, base,
                               run_test = ['alpha', 'beta', 'gamma'])

#     plot_params(params, labels=['alphas', 'betas', 'gammas'])

    results = parameter_tune(train, validate, params,\
                                num_tests = tune, val_range = .15)

    results = run_final_model(train, validate, results, validate=test)
    results['area'] = area
    results['time'] = time.time() - start
    
    print(f'Done with area: {area}')
    
    return results

def plot_predictions(results, start=-200, end= None, test=False, validate=True, scatter=False, area=False, save=False):
    '''
    show the predicted values for train & test
    '''
    
    results['train'][start:end].plot(figsize=(15,5), label='train data')
#     results['train_pred'][start:end].plot(label='train predictions')

    if validate:
        results['validate'][:24*3].plot(label='testing data')
        forecast = results['val_forecast'][:24*3]
        if scatter:
            
            plt.scatter(forecast.index, forecast, alpha=.8, s=15, label='predictions')
        else:
            forecast.plot(label='predictions')
            
    if test:
        results['test'].plot(label='test data')
        results['forecast'].plot(label='test predictions')
        
    plt.xlabel('')
    plt.ylabel('Total Rides per Hour', fontsize=20)
    plt.legend()
    if area:
        plt.title(f"Area {results['area']} Predictions")
    if save:
        plt.savefig(fname=f'results_{results["area"]}', transparent=True)
    plt.show()

    