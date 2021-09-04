import numpy as np
import scipy
from sklearn.linear_model import *
import statsmodels.api as sm
from lineartree import LinearTreeClassifier, LinearTreeRegressor
                
def changepoint_detector(x,y):
    '''
    ------------------------------------------------------------------------------
    PROGRAM: cru_changepoint_detector.py
    ------------------------------------------------------------------------------
    Version 0.1
    24 August, 2021
    Michael Taylor
    https://patternizer.github.io
    patternizer AT gmail DOT com
    michael DOT a DOT taylor AT uea DOT ac DOT uk
    ------------------------------------------------------------------------------
    Uses linear tree regression fit to CUSUM series to detect changepoints.
    
    y		 	: timeseries
    x		 	: timeseries index (e.g. decimal year dates)
    
    RETURNS:
    
    y_fit 		: linear tree regression fit at dervied optimal tree depth    
    y_fit_diff 	: 1st difference of y_fit
    breakpoints 	: vector of breakpoints (in the same format as x)
    depth 		: optimal tree depth derived from goodness of fit and correlation
    r			: vector of Pearson correlation coefficients as a function of tree depth
    R2adj		: vector of adjusted R^2 as a function of tree depth    
    ------------------------------------------------------------------------------
    CALL SYNTAX: y_fit, y_fit_diff, breakpoints, depth, r, R2adj = changepoint_detector(x, y)
    ------------------------------------------------------------------------------
    '''

    #--------------------------------------------------------------------------
    # METHODS
    #--------------------------------------------------------------------------

    def adjusted_r_squared(x,y):
        
        X = x.reshape(len(x),1)
        model = sm.OLS(y, X).fit()
        R2adj = model.rsquared_adj
        return R2adj

    #--------------------------------------------------------------------------
    # SETTINGS (pre-optimised for monthly timeseries)
    #--------------------------------------------------------------------------

    min_samples_leaf = 100
    max_bins = 24
    max_depth = 12 # in range [1,20]
    max_r_over_rmax = 0.995 

    # FORMAT: mask and reshape data

    mask = np.isfinite(y)
    x_obs = np.arange( len(y) ) / len(y)
    x_obs = x_obs[mask].reshape(-1, 1)
    y_obs = y[mask].reshape(-1, 1)		                

    # DEDUCE: optimal tree depth

    r = []
    r2adj = []
    for depth in range(1,max_depth+1):       
	       
        # FIT: linear tree regression (LTR) model

        lt = LinearTreeRegressor(
        base_estimator = LinearRegression(),
        min_samples_leaf = min_samples_leaf,
        max_bins = max_bins,
        max_depth = depth
        ).fit(x_obs, y_obs)            
        y_fit = lt.predict(x_obs)    	       
        Y = y_obs
        Z = y_fit.reshape(-1,1)
        mask_ols = np.isfinite(Y) & np.isfinite(Z)
        corrcoef = scipy.stats.pearsonr(Y[mask_ols], Z[mask_ols])[0]
        R2adj = adjusted_r_squared(Y,Z)
        r.append(corrcoef)
        r2adj.append(R2adj)    

    r_diff = [0.0] + list(np.diff(r))
    depth = np.arange(1,max_depth+1)[np.array(r/np.max(r)) > max_r_over_rmax][1] - 1

    # FIT: LTR model for optimum depth and extract breakpoints
		
    lt = LinearTreeRegressor(
        base_estimator = LinearRegression(),
        min_samples_leaf = min_samples_leaf,
        max_bins = max_bins,
        max_depth = depth        
    ).fit(x_obs, y_obs)    
    y_fit = lt.predict(x_obs)            
    y_fit_diff = [0.0] + list(np.diff(y_fit))        
    breakpoints = x[mask][ np.abs(y_fit_diff) >= np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)) ][0:]

    return y_fit, y_fit_diff, breakpoints, depth, r, R2adj    
#------------------------------------------------------------------------------



