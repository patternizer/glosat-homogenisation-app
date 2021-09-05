#------------------------------------------------------------------------------
# PROGRAM: adjustments_histogram.py
#------------------------------------------------------------------------------
# Version 0.1
# 5 September, 2021
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

# Numerics and dataframe libraries:
import numpy as np
import numpy.ma as ma
import scipy
import scipy.stats as stats    
import pandas as pd
import xarray as xr
import pickle
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
import matplotlib.colors as c
import seaborn as sns; sns.set()
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
matplotlib.rcParams['text.usetex'] = False

#------------------------------------------------------------------------------
import filter_cru_dft as cru_filter # CRU DFT filter
import cru_changepoint_detector as cru # CRU changepoint detector
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 16

# Seasonal mean parameters

nsmooth = 12                  # 1yr MA monthly
nfft = 10                     # decadal smoothing

df_temp = pd.read_pickle('df_temp_expect.pkl', compression='bz2')

# GENERATE: station code:name list

gb = df_temp.groupby(['stationcode'])['stationname'].unique().reset_index()
stationcodestr = gb['stationcode']
stationnamestr = [ gb['stationname'][i][0] for i in range(len(stationcodestr)) ]

#----------------------------------------------------------------------------------
# METHODS
#----------------------------------------------------------------------------------

def moving_average(x, w):
  """
  Calculate moving average of a vector
  
  Parameters:
    x (vector of float): the vector to be smoothed
    w (int): the number of samples over which to average
    
  Returns:
    (vector of float): smoothed vector, which is shorter than the input vector
  """
  return np.convolve(x, np.ones(w), 'valid') / w

def prepare_dists(lats, lons):
  """
  Prepare distance matrix from vectors of lat/lon in degrees assuming
  spherical earth
  
  Parameters:
    lats (vector of float): latitudes
    lons (vector of float): latitudes
  
  Returns:
    (matrix of float): distance matrix in km
  """
  las = np.radians(lats)
  lns = np.radians(lons)
  dists = np.zeros([las.size,las.size])
  for i in range(lns.size):
    dists[i,:] = 6371.0*np.arccos( np.minimum( (np.sin(las[i])*np.sin(las) + np.cos(las[i])*np.cos(las)*np.cos(lns[i]-lns) ), 1.0 ) )
  return dists

def smooth_fft(x, span):  
    
    y_lo, y_hi, zvarlo, zvarhi, fc, pctl = cru_filter.cru_filter_dft(x, span)    
    x_filtered = y_lo

    return x_filtered

#----------------------------------------------------------------------------------
# CALCULATE: global adjustments
#----------------------------------------------------------------------------------

adjustments = []
for k in range(len(stationcodestr)):
#for k in range(1000):
            
    df = df_temp[ df_temp['stationcode'] == stationcodestr[k] ].sort_values(by='year').reset_index(drop=True).dropna()
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    sd = (df.groupby('year').mean().iloc[:,43:55]).reset_index()        
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel()    
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel()    
    sd_monthly = np.array( sd.groupby('year').mean().iloc[:,0:12]).ravel()                   
    dn_monthly = np.array( dn.groupby('year').mean().iloc[:,0:12]).ravel()                   
    ts_monthly = np.array( moving_average( ts_monthly, nsmooth ) )    
    ex_monthly = np.array( moving_average( ex_monthly, nsmooth ) )    
    sd_monthly = np.array( moving_average( sd_monthly, nsmooth ) )    
    diff_monthly = ts_monthly - ex_monthly
    # Solve Y1677-Y2262 Pandas bug with Xarray:        
    t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')     
    mask = np.isfinite(ex_monthly) & np.isfinite(ts_monthly)    
         
    # CALCULATE: CUSUM
    	
    x = t_monthly[mask]
    y = np.cumsum( diff_monthly[mask] )

    if len(x) < 10:
        
        continue

    # CALL: cru_changepoint_detector

    y_fit, y_fit_diff, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
    
    if len(breakpoints) == 0:
        
        y_means = np.zeros(len(x))

    else:
    
        breakpoints_all = x[mask][ np.abs(y_fit_diff) >= np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)) ]    
        breakpoints_idx = np.arange(len(x[mask]))[ np.abs(y_fit_diff) >= np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)) ][0:]        
           
        # CALCULATE: intra-breakpoint fragment means
        
        y_means = []
        for j in range(len(breakpoints_all)+1):                
            if j == 0:              
                y_means = y_means + list( len( ts_monthly[mask][0:breakpoints_idx[0]] ) * [ -np.nanmean(ts_monthly[mask][0:breakpoints_idx[0]]) + np.nanmean(ex_monthly[mask][0:breakpoints_idx[0]]) ] ) 
            if (j > 0) & (j<len(breakpoints_all)):
                y_means = y_means + list( len( ts_monthly[mask][breakpoints_idx[j-1]:breakpoints_idx[j]] ) * [ -np.nanmean(ts_monthly[mask][breakpoints_idx[j-1]:breakpoints_idx[j]]) + np.nanmean(ex_monthly[mask][breakpoints_idx[j-1]:breakpoints_idx[j]]) ] ) 
            if (j == len(breakpoints_all)):              
                y_means = y_means + list( len( ts_monthly[mask][breakpoints_idx[-1]:] ) * [ -np.nanmean(ts_monthly[mask][breakpoints_idx[-1]:]) + np.nanmean(ex_monthly[mask][breakpoints_idx[-1]:]) ] ) 
    
    adjustments.extend(y_means)

    #----------------------------------------------------------------------------------
    # PLOT: global adjustments histogram every additional 100 stations
    #----------------------------------------------------------------------------------
    
    if (k>0) & (k%100 == 0):
        
        print(k)
    
        adjustments_all = np.array(adjustments).flatten()
        mask = (np.isfinite(adjustments_all)) & (np.abs(adjustments_all)>0.01)
        adjustments_k = adjustments_all[mask]

        xmin = -2.0; xmax = 2.0
        bins = 51
        x = np.linspace(xmin,xmax,bins)
        
        figstr = 'adjustment-histogram' + '-' + str(k).zfill(6) + '.png'
        titlestr = 'Histogram of LEK adjustments'
                         
        fig, ax = plt.subplots(figsize=(15,10))     
        kde = stats.gaussian_kde(adjustments_k)
        plt.hist(adjustments_k, density=False, bins=bins, alpha=1.0, color='lightgrey', label='bin counts')
        plt.xlabel(r'Temperature anomaly (from 1961-1990) adjustment [°C]', fontsize=fontsize)
        plt.ylabel('Count', fontsize=fontsize)
        plt.xlim(xmin,xmax)
        plt.grid(True, which='major')      
        plt.tick_params(labelsize=fontsize)    
        plt.legend(loc='upper right', fontsize=fontsize)
        plt.title(titlestr, fontsize=fontsize)
        plt.savefig(figstr, dpi=300)
        plt.close(fig)

adjustments_all = np.array(adjustments).flatten()
pd.DataFrame({'adjustments_all':adjustments_all}).to_csv('lek_adjustments_global.csv')

mask = (np.isfinite(adjustments_all)) & (np.abs(adjustments_all)>0.02)
adjustments = adjustments_all[mask]

#----------------------------------------------------------------------------------
# PLOT: global adjustments histogram
#----------------------------------------------------------------------------------

xmin = -2.0; xmax = 2.0
bins = 161
x = np.linspace(xmin,xmax,bins)

figstr = 'adjustment-histogram.png'
titlestr = 'Histogram of local expectation Kriging (LEK) adjustments: N(stations)=' + str(k)
                 
fig, ax = plt.subplots(figsize=(15,10))     
kde = stats.gaussian_kde(adjustments)
plt.hist(adjustments, density=False, bins=x, alpha=1.0, color='orange', edgecolor = "black", label='bin width=' + str(np.round(x[1]-x[0],3)))
plt.xlabel(r'Adjustment size [°C]', fontsize=fontsize)
plt.ylabel('Frequency', fontsize=fontsize)
plt.xlim(xmin,xmax)
plt.grid(True, which='major')      
plt.tick_params(labelsize=fontsize)    
plt.legend(loc='upper right', fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr, dpi=300)
plt.close(fig)

print('** END')
    
