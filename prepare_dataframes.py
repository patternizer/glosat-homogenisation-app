#------------------------------------------------------------------------------
# PROGRAM: prepare_dataframes.py
#------------------------------------------------------------------------------
# Version 0.1
# 31 July, 2020
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

# Numerics and dataframe libraries:
import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import pickle

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

reduce_precision = True

#------------------------------------------------------------------------------
# LOAD: LEK output dataframe: df_temp_expect
#------------------------------------------------------------------------------

df_temp = pd.read_pickle('df_temp_expect.pkl', compression='bz2')

if reduce_precision == True:

    # PRECISION: --> int16 and float16
        
    for i in range(1,13):                 
        
        df_temp[str(i)] = df_temp[str(i)].astype('float16')
        df_temp['n'+str(i)] = df_temp['n'+str(i)].astype('float16')
        df_temp['e'+str(i)] = df_temp['e'+str(i)].astype('float16')
        df_temp['s'+str(i)] = df_temp['s'+str(i)].astype('float16')
        
    df_temp['stationlat'] = df_temp['stationlat'].astype('float16')
    df_temp['stationlon'] = df_temp['stationlon'].astype('float16')
    df_temp['stationelevation'] = df_temp['stationelevation'].astype('float16')
    df_temp['year'] = df_temp['year'].astype('int16')
    df_temp['stationfirstyear'] = df_temp['stationfirstyear'].astype('float16')          # NB: int16 fails due to NaN being float
    df_temp['stationlastyear'] = df_temp['stationlastyear'].astype('float16')            # NB: int16 fails due to NaN being float
    df_temp['stationfirstreliable'] = df_temp['stationfirstreliable'].astype('float16')  # NB: int16 fails due to NaN being float
    df_temp['stationsource'] = df_temp['stationsource'].astype('float16')                # NB: int16 fails due to NaN being float
#    del df_temp['stationfirstyear']
#    del df_temp['stationlastyear']
#    del df_temp['stationfirstreliable']
#    del df_temp['stationsource']
    df_temp.to_pickle( "df_temp_expect_reduced.pkl", compression="bz2" )

#------------------------------------------------------------------------------
# CONSTRUCT: anaomlies dataframe --> df_anom (NB: includes metadata)
#------------------------------------------------------------------------------

df_anom = df_temp.copy()
for i in range(1,13):
    
    df_anom[str(i)] = df_temp[str(i)] - df_temp['n'+str(i) ]

df_anom = df_anom.iloc[:,0:19]
df_anom.to_pickle( "df_anom.pkl", compression="bz2" )

#------------------------------------------------------------------------------
# CONSTRUCT: local expectations dataframe --> df_ex (NB: only stationcode metadata)
#------------------------------------------------------------------------------

df_ex = df_temp.copy()
for i in range(1,13):
    
    df_ex[str(i)] = df_temp['e'+str(i)]

df_ex = df_ex.iloc[:,0:14]
df_ex.to_pickle( "df_ex.pkl", compression="bz2" )

#------------------------------------------------------------------------------
# CONSTRUCT: local expectations SD dataframe --> df_sd (NB: only stationcode metadata)
#------------------------------------------------------------------------------

df_sd = df_temp.copy()
for i in range(1,13):
    
    df_sd[str(i)] = df_temp['s'+str(i)]

df_sd = df_sd.iloc[:,0:14]
df_sd.to_pickle( "df_sd.pkl", compression="bz2" )

#------------------------------------------------------------------------------
print('** END')
