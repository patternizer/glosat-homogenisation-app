#------------------------------------------------------------------------------
# PROGRAM: glosat.py
#------------------------------------------------------------------------------
# Version 0.17
# 16 November, 2021
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
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
matplotlib.rcParams['text.usetex'] = False
# Plotly libraries
import plotly.express as px
import plotly.graph_objects as go
# App Deployment Libraries
#import dash
#from dash import dcc
#from dash import html
#import dash_leaflet as dl
#import dash_leaflet.express as dlx
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import Dash
from dash_extensions.javascript import assign

from apps import home, about, glosat
from app import app

#------------------------------------------------------------------------------
import filter_cru_dft as cru_filter # CRU DFT filter
import cru_changepoint_detector as cru # CRU changepoint detector
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 12

# Seasonal mean parameters

nsmooth = 12                  # 1yr MA monthly
nfft = 10                     # decadal smoothing

df_temp = pd.read_pickle('df_temp_expect_reduced.pkl', compression='bz2')

# GENERATE: station code:name list

gb = df_temp.groupby(['stationcode'])['stationname'].unique().reset_index()
stationcodestr = gb['stationcode']
stationnamestr = [ gb['stationname'][i][0] for i in range(len(stationcodestr)) ]
stationstr = stationcodestr + ': ' + stationnamestr
opts = [{'label' : stationstr[i], 'value' : i} for i in range(len(stationstr))]

#==============================================================================
value = np.where(df_temp['stationcode'].unique()=='619930')[0][0] # Pamplemousses
#==============================================================================

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

#------------------------------------------------------------------------------
# GloSAT APP LAYOUT
#------------------------------------------------------------------------------

layout = html.Div([
    dbc.Container([
                
        dbc.Row([
            dbc.Col(html.Div([   
                dcc.Dropdown(
                    id = "station",
                    options = opts,           
                    value = 0,
                    style = {"color": "black", 'padding' : '10px', 'width': '100%', 'display': 'inline-block'},
                ),                                    
            ], className="dash-bootstrap"), 
            width={'size':6},
            ),         
                                
            dbc.Col( html.Div([
                dcc.Graph(id="station-info", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}), 
            ]), 
            width={'size':6}, 
            ),                        
        ]),

        dbc.Row([
            dbc.Col(html.Div([                    
                dcc.Graph(id="plot-timeseries", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                     
            ]), 
            width={'size':6}, 
            ),                        
            dbc.Col(html.Div([                    
                dcc.Graph(id="plot-differences", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                     
            ]), 
            width={'size':6}, 
            ),                        

        ]),

        dbc.Row([
            dbc.Col(html.Div([                    
                dcc.Graph(id="plot-adjustments", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                     
            ]), 
            width={'size':6}, 
            ),                        
            dbc.Col(html.Div([
                dcc.Graph(id="plot-changepoints", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                    
            ]), 
            width={'size':6}, 
            ),           
        ]),

        dbc.Row([
            dbc.Col(html.Div([
                dcc.Graph(id="plot-seasonal", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                    
            ]), 
            width={'size':6}, 
            ),           
            dbc.Col( html.Div([
                dcc.Graph(id="breakpoints", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}), 
            ]), 
            width={'size':6}, 
            ),     
        ]),
        
        dbc.Row([
            dbc.Col(html.Div([     
                html.Br(),
                html.Label(['Status: Experimental']),
                html.Br(),
                html.Label(['Dataset: GloSAT.p03']),
                html.Br(),
                html.Label(['Codebase: ', html.A('Github', href='https://github.com/patternizer/glosat-homogenisation-app')]),                
            ],
            style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),    
            width={'size':6}, 
            ),   
#            dbc.Col(html.Div([
#                dcc.Graph(id="plot-worldmap", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                    
#            ]), 
#            width={'size':6}, 
#            ),                                            
        ]),
            
    ]),
])

# ========================================================================
# Callbacks
# ========================================================================

@app.callback(
    Output(component_id='station-info', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )
    
def update_station_info(value):
    
    """
    Display station info
    """

    lat = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationlat'].iloc[0]
    lon = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationlon'].iloc[0]
    elevation = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationelevation'].iloc[0]
    station = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationname'].iloc[0].upper()
    country = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationcountry'].iloc[0]
                                  
    data = [
        go.Table(
            header=dict(values=['Latitude <br> [°N]','Longitude <br> [°E]','Elevation <br> [m] AMSL','Station <br>','Country <br>'],
                line_color='darkslategray',
                fill_color='lightgrey',
                font = dict(color='Black'),
                align='left'),
            cells=dict(values=[
                    [str(lat)], 
                    [str(lon)],
                    [str(elevation)],
                    [station], 
                    [country], 
                ],
                line_color='slategray',
                fill_color='black',
                font = dict(color='white'),
                align='left')
        ),
    ]
    layout = go.Layout(
       template = "plotly_dark", 
       height=130, width=550, margin=dict(r=10, l=10, b=10, t=10))

    return {'data': data, 'layout':layout} 

@app.callback(
    Output(component_id='plot-worldmap', component_property='figure'),
    [Input(component_id='station', component_property='value')],                   
    )
            
def update_plot_worldmap(value):
    
    """
    Plot station location on world map
    """

    # EXTRACT: Neighbouring stations (lat,lon)

    lats = df_temp.groupby('stationcode').mean()["stationlat"].values
    lons = df_temp.groupby('stationcode').mean()["stationlon"].values
#   dists = prepare_dists( lats, lons )        
    da = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True)    
    lat = [ df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationlat'].iloc[0] ]
    lon = [ df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationlon'].iloc[0] ]
    station = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationcode'].iloc[0]
    
    df = df_temp.groupby('stationcode').mean()
    df['stationlat'] = [ np.round( df['stationlat'][i], 2) for i in range(len(df)) ]
    df['stationlon'] = [ np.round( df['stationlon'][i], 2) for i in range(len(df)) ]
    
    fig = go.Figure(
#	px.set_mapbox_access_token(open(".mapbox_token").read()),
        px.scatter_mapbox(da, lat='stationlat', lon='stationlon', color_discrete_sequence=['rgba(234, 89, 78, 1.0)'], size_max=10, zoom=5, opacity=0.7)
#        px.scatter_mapbox(lat=df['stationlat'], lon=df['stationlon'], text=df.index, color_discrete_sequence=['rgba(234, 89, 78, 1.0)'], zoom=3, opacity=0.7)
#        dl.Map([dl.TileLayer(), cluster], center=(33, 33), zoom=3, id="map", style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block"}),
   )
 
    fig.update_layout(
       template = "plotly_dark",
#      template = None,
        xaxis_title = {'text': 'Longitude, °E'},
        yaxis_title = {'text': 'Latitude, °N'},
        title = {'text': 'STATIONS WITHIN 900 KM', 'x':0.1, 'y':0.95},        
    )
#    fig.update_layout(mapbox_style="carto-positron", mapbox_center_lat=df['stationlat'].iloc[0], mapbox_center_lon=df['stationlon'].iloc[0]) 
    fig.update_layout(mapbox_style="carto-positron", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0]) 
#    fig.update_layout(mapbox_style="stamen-watercolor", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0]) 
#    fig.update_layout(mapbox_style="open-street-map", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0]) 
#    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lat=lat[0], mapbox_center_lon=lon[0])     
    fig.update_layout(height=400, width=550, margin={"r":10,"t":50,"l":10,"b":40})    
    
    return fig

@app.callback(
    Output(component_id='plot-timeseries', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )
    
def update_plot_timeseries(value):
    
    """
    Plot station timeseries
    """

    df_compressed = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()    
    t_yearly = np.arange( df_compressed.iloc[0].year, df_compressed.iloc[-1].year + 1)
    df_yearly = pd.DataFrame({'year':t_yearly})
    df = df_yearly.merge(df_compressed, how='left', on='year')
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,43:55]).reset_index()  
    
    # TRIM: to start of Pandas datetime range
        
    da = da[da.year >= 1678].reset_index(drop=True)
    de = de[de.year >= 1678].reset_index(drop=True)
    ds = ds[ds.year >= 1678].reset_index(drop=True)
          
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)           

    # Solve Y1677-Y2262 Pandas bug with Xarray:        
#   t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')      
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')     
    mask = np.array(len(ts_monthly) * [True])

    # COMPUTE: 12-m MA
    
    a = pd.Series(ts_monthly).rolling(12, center=True).mean().values
    e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
    s = pd.Series(sd_monthly).rolling(12, center=True).mean().values

    t = t_monthly[mask]
    a = a[mask]
    e = e[mask]
    s = s[mask]

    diff_yearly = a - e
    
    # CALCULATE: CUSUM
        	
    c = np.nancumsum( diff_yearly )
    x = ( np.arange(len(c)) / len(c) )
    y = c    
        
    if mask.sum() > 0:

        data = []
        
        trace_error = [
            go.Scatter(x=t, y=e+s, 
                       mode='lines', 
                       fill='none',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.0)'),             # grey                  
                       name='uncertainty',      
                       showlegend=False,
            ),
            go.Scatter(x=t, y=e-s, 
                       mode='lines', 
                       fill='tonexty',
                       fillcolor='rgba(242,242,242,0.2)',                               # grey
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.0)'),             # grey                  
                       name='uncertainty',      
            )]         

        trace_expect = [            
            go.Scatter(x=t, y=e, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(234, 89, 78, 1.0)'),                 # red (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(234, 89, 78, 1.0)'),     # red (colorsafe)
                name='E',
            )]

        trace_obs = [
            go.Scatter(x=t, y=a, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(20,115,175,1.0)'),                  # blue (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(20,115,175,1.0)'),      # blue (colorsafe)
                name='O',
            )]   

    data = data + trace_error + trace_expect + trace_obs
                                      
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis = dict(range=[t[0],t[-1]]),       
        yaxis_title = {'text': 'Anomaly (from 1961-1990), °C'},
        title = {'text': 'OBSERVATIONS (O) Vs LOCAL EXPECTATION (E)', 'x':0.1, 'y':0.95},                
    )

    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t_yearly[np.floor(len(t)/2).astype(int)],
                    y=0,
                    xref="x",
                    yref="y",
                    text="No anomaly timeseries",
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="#ffffff"
                        ),                    
                )
            ]
        )   
         
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor="top",
            y=0.1,
            xanchor="left",
            x=0.05),              
    )        
    fig.update_layout(height=400, width=550, margin={"r":10,"t":50,"l":70,"b":50})    
    
    return fig

@app.callback(
    Output(component_id='plot-differences', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )

def update_plot_differences(value):
    
    """
    Plot station timeseries differences
    """

    df_compressed = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()    
    t_yearly = np.arange( df_compressed.iloc[0].year, df_compressed.iloc[-1].year + 1)
    df_yearly = pd.DataFrame({'year':t_yearly})
    df = df_yearly.merge(df_compressed, how='left', on='year')
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,43:55]).reset_index()    
    
    # TRIM: to start of Pandas datetime range
        
    da = da[da.year >= 1678].reset_index(drop=True)
    de = de[de.year >= 1678].reset_index(drop=True)
    ds = ds[ds.year >= 1678].reset_index(drop=True)
            
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)           

    # Solve Y1677-Y2262 Pandas bug with Xarray:        
#   t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')      
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')     
    mask = np.array(len(ts_monthly) * [True])

    # COMPUTE: 12-m MA
    
    a = pd.Series(ts_monthly).rolling(12, center=True).mean().values
    e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
    s = pd.Series(sd_monthly).rolling(12, center=True).mean().values

    t = t_monthly[mask]
    a = a[mask]
    e = e[mask]
    s = s[mask]

    diff_yearly = a - e
    
    # CALCULATE: CUSUM
        	
    c = np.nancumsum( diff_yearly )
    x = ( np.arange(len(c)) / len(c) )
    y = c    

    # CALL: cru_changepoint_detector

#    y_fit, y_fit_diff, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
    y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
            
    if mask.sum() > 0:

        data = []

        trace_error = [                    
            go.Scatter(x=t, y=s, 
                       mode='lines', 
                       fill='none',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.0)'),             # grey                  
                       name='uncertainty',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
            ),
            go.Scatter(x=t, y=-s, 
                       mode='lines', 
                       fill='tonexty',
                       fillcolor='rgba(242,242,242,0.2)',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.0)'),             # grey                  
                       name='uncertainty',      
#                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
            )]

        trace_diff = [
            go.Scatter(x=t, y=diff_yearly, 
                mode='lines+markers', 
                
#                line=dict(width=1.0, color='rgba(242,242,242,0.2)'),                   # grey                  
#                line=dict(width=1.0, color='rgba(234, 89, 78, 1.0)'),                  # red (colorsafe)                      
#                line=dict(width=1.0, color='rgba(20,115,175,1.0)'),                    # blue (colorsafe)
#                line=dict(width=1.0, color='rgba(137, 195, 239, 1.0)'),                # lightblue (colorsafe)                               
#                line=dict(width=1.0, color='rgba(229, 176, 57, 1.0)'),                 # mustard (colorsafe)

                line=dict(width=1.0, color='rgba(137, 195, 239, 1.0)'),                 # lightblue (colorsafe)   
                marker=dict(size=5, opacity=0.5, color='rgba(137, 195, 239, 1.0)'),     # lightblue (colorsafe)   
                name='O-E',
#               hovertemplate='%{y:.2f}',
            )]   
                                      
        data = data + trace_error + trace_diff
        
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis = dict(range=[t[0],t[-1]]),       
        yaxis_title = {'text': 'Anomaly (from 1961-1990), °C'},
        title = {'text': 'DIFFERENCE (O-E)', 'x':0.1, 'y':0.95},        
    )

    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t_yearly[np.floor(len(t)/2).astype(int)],
                    y=0,
                    xref="x",
                    yref="y",
                    text="No differences",
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="#ffffff"
                        ),                    
                )
            ]
        )    
        
    for k in range(len(breakpoints)):
    
        print(t[breakpoints[k]])    	
        fig.add_shape(type='line',
            yref="y",
            xref="x",
            x0=t[breakpoints[k]],
            y0=np.nanmin([np.nanmin(diff_yearly), np.nanmin(s)]),                        
            x1=t[breakpoints[k]],
            y1=np.nanmax([np.nanmax(diff_yearly), np.nanmax(s)]),    
            line=dict(color='rgba(229, 176, 57, 1)', width=1, dash='dot'))        
        
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor="top",
            y=0.1,
            xanchor="left",
            x=0.05),              
    )        
    fig.update_layout(height=400, width=550, margin={"r":10,"t":50,"l":70,"b":50})    

    return fig

@app.callback(
    Output(component_id='plot-changepoints', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )

def update_plot_changepoints(value):
    
    """
    Plot station timeseries CUSUM changepoints
    """

    df_compressed = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()    
    t_yearly = np.arange( df_compressed.iloc[0].year, df_compressed.iloc[-1].year + 1)
    df_yearly = pd.DataFrame({'year':t_yearly})
    df = df_yearly.merge(df_compressed, how='left', on='year')
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,43:55]).reset_index()      
    
    # TRIM: to start of Pandas datetime range
        
    da = da[da.year >= 1678].reset_index(drop=True)
    de = de[de.year >= 1678].reset_index(drop=True)
    ds = ds[ds.year >= 1678].reset_index(drop=True)
      
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)           

    # Solve Y1677-Y2262 Pandas bug with Xarray:        
#   t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')      
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')     
    mask = np.array(len(ts_monthly) * [True])

    # COMPUTE: 12-m MA
    
    a = pd.Series(ts_monthly).rolling(12, center=True).mean().values
    e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
    s = pd.Series(sd_monthly).rolling(12, center=True).mean().values

    t = t_monthly[mask]
    a = a[mask]
    e = e[mask]
    s = s[mask]

    diff_yearly = a - e
    
    # CALCULATE: CUSUM
        	
    c = np.nancumsum( diff_yearly )
    x = ( np.arange(len(c)) / len(c) )
    y = c    

    # CALL: cru_changepoint_detector

#    y_fit, y_fit_diff, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
    y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)

    print('BE WATER MY FRIEND')

    data = []
    
    trace_ltr_cumsum = [
           
        go.Scatter(x=t, y=y, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(20,115,175,1.0)'),                     # blue (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(20,115,175,1.0)'),         # blue (colorsafe)
                name='CUSUM (O-E)',
        )] 

    trace_ltr = [
           
        go.Scatter(x=t, y=y_fit, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(234, 89, 78, 1.0)'),                   # red (colorsafe)
                marker=dict(size=2, opacity=0.5, color='rgba(234, 89, 78, 1.0)'),       # red (colorsafe)
                name='LTR fit',
        )]
    
    trace_ltr_diff = [
           
        go.Scatter(x=t, y=y_fit_diff, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(229, 176, 57, 1.0)'),                 # mustard (colorsafe)
                marker=dict(size=2, opacity=0.5, color='rgba(229, 176, 57, 1.0)'),     # mustard (colorsafe)
                name='d(LTR)',                
        )]
        
    trace_6_sigma = [                    
    
        go.Scatter(x=t, y=np.tile( np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)), len(t) ), 
                       mode='lines', 
                       fill='none',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(229, 176, 57, 0.0)'),           # mustard (colorsafe)                  
                       name='6 sigma',      
                       showlegend=False,
        ),           
        go.Scatter(x=t, y=np.tile( np.abs(np.nanmean(y_fit_diff)) - 6.0*np.abs(np.nanstd(y_fit_diff)), len(t) ), 
                       mode='lines', 
                       fill='tonexty',
                       fillcolor='rgba(229, 176, 57, 0.2)',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(229, 176, 57, 0.2)'),           # mustard (colorsafe)       
                       name='μ ± 6σ',                       
       )]

    trace_slopes = [
                    
        go.Scatter(x=t, y=slopes, 
                mode='lines', 
                fill='tozeroy',
                fillcolor='rgba(137, 195, 239, 0.2)',                      # lightblue (colorsafe)
                line=dict(width=1.0, color='rgba(137, 195, 239, 0.2)'),    # lightblue (colorsafe)
                connectgaps=True,
                name='CUSUM / decade',                
        )]
                                            	                                          
#    data = data + trace_ltr_cumsum + trace_ltr + trace_ltr_diff + trace_6_sigma
    data = data + trace_ltr_cumsum + trace_ltr + trace_slopes
                                     
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis = dict(range=[t[0],t[-1]]),       
        yaxis_title = {'text': 'CUSUM (O-E)'},
        title = {'text': 'CHANGEPOINTS', 'x':0.1, 'y':0.95},                          
    )

    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t_yearly[np.floor(len(t)/2).astype(int)],
                    y=0,
                    xref="x",
                    yref="y",
                    text="No CUSUM data",
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="#ffffff"
                        ),                    
                )
            ]
        )    

    for k in range(len(breakpoints)):
    
        print(t[breakpoints[k]])    	
        fig.add_shape(type='line',
            yref="y",
            xref="x",
            x0=t[breakpoints[k]],
            y0=np.nanmin([np.nanmin(slopes), np.nanmin(y), np.nanmin(y_fit)]),                        
            x1=t[breakpoints[k]],
            y1=np.nanmax([np.nanmax(slopes), np.nanmax(y), np.nanmax(y_fit)]),                        
            line=dict(color='rgba(229, 176, 57, 1)', width=1, dash='dot'))
    
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor="top",
            y=0.1,
            xanchor="left",
            x=0.05),              
    )        
    fig.update_layout(height=400, width=550, margin={"r":10,"t":50,"l":70,"b":50})    
    
    return fig
       
@app.callback(
    Output(component_id='plot-adjustments', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )
    
def update_plot_adjustments(value):
    
    """
    Plot station timeseries adjustments
    """

    df_compressed = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()    
    t_yearly = np.arange( df_compressed.iloc[0].year, df_compressed.iloc[-1].year + 1)
    df_yearly = pd.DataFrame({'year':t_yearly})
    df = df_yearly.merge(df_compressed, how='left', on='year')
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,43:55]).reset_index()    
    
    # TRIM: to start of Pandas datetime range
        
    da = da[da.year >= 1678].reset_index(drop=True)
    de = de[de.year >= 1678].reset_index(drop=True)
    ds = ds[ds.year >= 1678].reset_index(drop=True)
        
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)           

    # Solve Y1677-Y2262 Pandas bug with Xarray:        
#   t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')      
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')     
    mask = np.array(len(ts_monthly) * [True])

    # COMPUTE: 12-m MA
    
    a = pd.Series(ts_monthly).rolling(12, center=True).mean().values
    e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
    s = pd.Series(sd_monthly).rolling(12, center=True).mean().values

    t = t_monthly[mask]
    a = a[mask]
    e = e[mask]
    s = s[mask]

    diff_yearly = a - e
    
    # CALCULATE: CUSUM
        	
    c = np.nancumsum( diff_yearly )
    x = ( np.arange(len(c)) / len(c) )
    y = c    
    
    # CALL: cru_changepoint_detector

    #y_fit, y_fit_diff, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
    y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
       
    if len(breakpoints) == 0:
    
        mask = np.array(len(t)*[False])
        data = []
        trace_obs = [
            go.Scatter(x=t, y=a, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(20,115,175,0.0)'),                  # blue (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(20,115,175,0.0)'),      # blue (colorsafe)
                name='O',
#               hovertemplate='%{y:.2f}',
            )]       	
        data = data + trace_obs
            
    if mask.sum() > 0:

#        breakpoints_all = x[mask][ np.abs(y_fit_diff) >= np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)) ][0:]    
#        breakpoints_idx = np.arange(len(x[mask]))[ np.abs(y_fit_diff) >= np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)) ][0:]        
        breakpoints_all = breakpoints
        breakpoints_idx = breakpoints
       
        # CALCULATE: intra-breakpoint fragment means
    
        y_means = []
        for j in range(len(breakpoints_all)+1):                
            if j == 0:              
                y_means = y_means + list( len( a[0:breakpoints_idx[0]] ) * [ -np.nanmean(a[0:breakpoints_idx[0]]) + np.nanmean(e[0:breakpoints_idx[0]]) ] ) 
            if (j > 0) & (j<len(breakpoints_all)):
                y_means = y_means + list( len( a[breakpoints_idx[j-1]:breakpoints_idx[j]] ) * [ -np.nanmean(a[breakpoints_idx[j-1]:breakpoints_idx[j]]) + np.nanmean(e[breakpoints_idx[j-1]:breakpoints_idx[j]]) ] ) 
            if (j == len(breakpoints_all)):              
                y_means = y_means + list( len( a[breakpoints_idx[-1]:] ) * [ -np.nanmean(a[breakpoints_idx[-1]:]) + np.nanmean(e[breakpoints_idx[-1]:]) ] ) 

        data = []
                         
        trace_error = [
            go.Scatter(x=t, y=e+s, 
                       mode='lines', 
                       fill='none',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.0)'),             # grey                  
                       name='uncertainty',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
            ),
            go.Scatter(x=t, y=e-s, 
                       mode='lines', 
                       fill='tonexty',
                       fillcolor='rgba(242,242,242,0.2)',                               # grey
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.0)'),             # grey                  
                       name='uncertainty',      
#                       showlegend=True,
#                      hovertemplate='%{y:.2f}',
            )]   
            	
        trace_obs = [
            go.Scatter(x=t, y=a, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(20,115,175,1.0)'),                  # blue (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(20,115,175,1.0)'),      # blue (colorsafe)
                name='O',
#               hovertemplate='%{y:.2f}',
            )]   

        trace_expect = [            
            go.Scatter(x=t, y=e, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(234, 89, 78, 1.0)'),                 # red (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(234, 89, 78, 1.0)'),     # red (colorsafe)
                name='E',
#               hovertemplate='%{y:.2f}',
            )]

        trace_obs_adjusted = [
               
            go.Scatter(x=t, y=a + y_means, 
                    mode='lines+markers', 
                    line=dict(width=1.0, color='rgba(137, 195, 239, 1.0)'),                 # lightblue (colorsafe)
                    marker=dict(size=2, opacity=0.5, color='rgba(137, 195, 239, 1.0)'),     # lightblue (colorsafe)
                    name='O (adjusted)',                
    #               hovertemplate='%{y:.2f}', 
            )]
                    
        trace_adjustments = [
               
            go.Scatter(x=t, y=y_means, 
                    mode='lines+markers', 
                    line=dict(width=1.0, color='rgba(229, 176, 57, 1.0)'),                 # mustard (colorsafe)
                    marker=dict(size=3, opacity=0.5, color='rgba(229, 176, 57, 1.0)'),     # mustard (colorsafe)
                    name='adjustment',                
    #               hovertemplate='%{y:.2f}', 
            )]
                                      	                                          
        data = data + trace_error + trace_expect + trace_obs + trace_obs_adjusted + trace_adjustments
                                     
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis = dict(range=[t[0],t[-1]]),       
        yaxis_title = {'text': 'Anomaly (from 1961-1990), °C'},
        title = {'text': 'ADJUSTMENTS', 'x':0.1, 'y':0.95},                           
    )

    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t[np.floor(len(t)/2).astype('int')],
                    y=0,
                    xref="x",
                    yref="y",
                    text="No adjustments",
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="#ffffff"
                        ),                    
                )
            ]
        )    

    for k in range(len(breakpoints)):

        fig.add_shape(type='line',
            yref="y",
            xref="x",
            x0=t[breakpoints[k]],
            y0=np.nanmin([np.nanmin(a), np.nanmin(e), np.nanmin(e-s)]),
            x1=t[breakpoints[k]],
            y1=np.nanmax([np.nanmax(a), np.nanmax(e), np.nanmax(e+s)]),
            line=dict(color='rgba(229, 176, 57, 1)', width=1, dash='dot'),
       )

    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor="top",
            y=0.1,
            xanchor="left",
            x=0.05),              
    )        
    fig.update_layout(height=400, width=550, margin={"r":10,"t":50,"l":70,"b":50})    
    
    return fig

@app.callback(
    Output(component_id='plot-seasonal', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )

def update_plot_seasonal(value):
    
    """
    Plot seasonal local expectations
    """

#    df = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()
#    dt = df.groupby('year').mean().iloc[:,0:12]
#    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
#    dn = dt.copy()
#    dn.iloc[:,0:] = dn_array
#    da = (dt - dn).reset_index()
#    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
#    sd = (df.groupby('year').mean().iloc[:,43:55]).reset_index()      
#    
#    # TRIM: to start of Pandas datetime range
#    
#    da = da[da.year >= 1678].reset_index(drop=True)
#    de = de[de.year >= 1678].reset_index(drop=True)
#    sd = sd[sd.year >= 1678].reset_index(drop=True)
#          
#    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel()    
#    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel()    
#    sd_monthly = np.array( sd.groupby('year').mean().iloc[:,0:12]).ravel()                   

    df_compressed = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()    
    t_yearly = np.arange( df_compressed.iloc[0].year, df_compressed.iloc[-1].year + 1)
    df_yearly = pd.DataFrame({'year':t_yearly})
    df = df_yearly.merge(df_compressed, how='left', on='year')
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,43:55]).reset_index()    
    
    # TRIM: to start of Pandas datetime range
        
    da = da[da.year >= 1678].reset_index(drop=True)
    de = de[de.year >= 1678].reset_index(drop=True)
    ds = ds[ds.year >= 1678].reset_index(drop=True)
        
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)           

    # Solve Y1677-Y2262 Pandas bug with Xarray:        
#   t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')      
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')     
    mask = np.array(len(ts_monthly) * [True])
    
    # EXTRACT: seasonal components

    trim_months = len(ex_monthly)%12
    df = pd.DataFrame({'Tg':ex_monthly[:-1-trim_months]}, index=t_monthly[:-1-trim_months])     
    
    t = [ pd.to_datetime( str(df.index.year.unique()[i])+'-01-01') for i in range(len(df.index.year.unique())) ][1:] # years
    DJF = ( df[df.index.month==12]['Tg'].values + df[df.index.month==1]['Tg'].values[1:] + df[df.index.month==2]['Tg'].values[1:] ) / 3
    MAM = ( df[df.index.month==3]['Tg'].values[1:] + df[df.index.month==4]['Tg'].values[1:] + df[df.index.month==5]['Tg'].values[1:] ) / 3
    JJA = ( df[df.index.month==6]['Tg'].values[1:] + df[df.index.month==7]['Tg'].values[1:] + df[df.index.month==8]['Tg'].values[1:] ) / 3
    SON = ( df[df.index.month==9]['Tg'].values[1:] + df[df.index.month==10]['Tg'].values[1:] + df[df.index.month==11]['Tg'].values[1:] ) / 3
        
    df_seasonal = pd.DataFrame({'DJF':DJF, 'MAM':MAM, 'JJA':JJA, 'SON':SON}, index = t)     
    df_seasonal_ma = df_seasonal.rolling(10, center=True).mean() # decadal smoothing
    mask = np.isfinite(df_seasonal_ma)

    dates = df_seasonal_ma.index
    df_seasonal_fft = pd.DataFrame(index=dates)
    df_seasonal_fft['DJF'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal_ma['DJF'].values[mask['DJF']], nfft)}, index=df_seasonal_ma['DJF'].index[mask['DJF']])
    df_seasonal_fft['MAM'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal_ma['MAM'].values[mask['MAM']], nfft)}, index=df_seasonal_ma['MAM'].index[mask['MAM']])
    df_seasonal_fft['JJA'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal_ma['JJA'].values[mask['JJA']], nfft)}, index=df_seasonal_ma['JJA'].index[mask['JJA']])
    df_seasonal_fft['SON'] = pd.DataFrame({'DJF':smooth_fft(df_seasonal_ma['SON'].values[mask['SON']], nfft)}, index=df_seasonal_ma['SON'].index[mask['SON']])
                
    mask = np.isfinite(df_seasonal_fft)
    data = []
    trace_winter=[
            go.Scatter(                                  
                x=df_seasonal_fft.index[mask['DJF']], y=df_seasonal_fft['DJF'][mask['DJF']], 
                mode='lines+markers', 
                line=dict(width=1, color='rgba(20,115,175,0.5)'),
                marker=dict(size=7, symbol='square', opacity=0.5, color='rgba(20,115,175,0.5)', line_width=1, line_color='rgba(20,115,175,0.5)'),                       
                name='DJF')
    ]
    trace_spring=[
            go.Scatter(                                  
                x=df_seasonal_fft.index[mask['MAM']], y=df_seasonal_fft['MAM'][mask['MAM']], 
                mode='lines+markers', 
                line=dict(width=1, color='rgba(137, 195, 239, 0.5)'),
                marker=dict(size=7, symbol='square', opacity=0.5, color='rgba(137, 195, 239, 0.5)', line_width=1, line_color='rgba(137, 195, 239, 1.0)'),       
                name='MAM')
    ]
    trace_summer=[
            go.Scatter(                                  
                x=df_seasonal_fft.index[mask['JJA']], y=df_seasonal_fft['JJA'][mask['JJA']], 
                mode='lines+markers', 
                line=dict(width=1, color='rgba(234, 89, 78, 0.5)'),
                marker=dict(size=7, symbol='square', opacity=0.5, color='rgba(234, 89, 78, 0.5)', line_width=1, line_color='rgba(234, 89, 78, 0.5)'),       
                name='JJA')
    ]
    trace_autumn=[
            go.Scatter(                                  
                x=df_seasonal_fft.index[mask['SON']], y=df_seasonal_fft['SON'][mask['SON']], 
                mode='lines+markers', 
                line=dict(width=1, color='rgba(229, 176, 57, 0.5)'),
                marker=dict(size=7, symbol='square', opacity=0.5, color='rgba(229, 176, 57, 0.5)', line_width=1, line_color='rgba(229, 176, 57, 0.5)'),       
                name='SON')
    ]
    
    data = data + trace_winter + trace_spring + trace_summer + trace_autumn
                                          
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis = dict(range=[dates[0],dates[-1]]),       
        xaxis_title = {'text': 'Year'},
        yaxis_title = {'text': 'Anomaly (from 1961-1990), °C'},
        title = {'text': 'SEASONAL DECADAL EXPECTATIONS (E)', 'x':0.1, 'y':0.95},
    )

    if mask.sum().all() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=dates[np.floor(len(dates)/2).astype(int)],
                    y=0,
                    xref="x",
                    yref="y",
                    text="No seasonal extracts",
                    showarrow=False,
                    font=dict(
                        family="Courier New, monospace",
                        size=16,
                        color="#ffffff"
                        ),                    
                )
            ]
        )    

    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor="top",
            y=0.1,
            xanchor="left",
            x=0.05),              
    )        
    fig.update_layout(height=400, width=550, margin={"r":10,"t":50,"l":70,"b":50})    

    return fig

@app.callback(
    Output(component_id='breakpoints', component_property='figure'),
    [Input(component_id='station', component_property='value')],    
    )
    
def update_breakpoints(value):
    
    """
    Display breakpoints
    """

    df_compressed = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()    
    t_yearly = np.arange( df_compressed.iloc[0].year, df_compressed.iloc[-1].year + 1)
    df_yearly = pd.DataFrame({'year':t_yearly})
    df = df_yearly.merge(df_compressed, how='left', on='year')
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    ds = (df.groupby('year').mean().iloc[:,43:55]).reset_index()        
    
    # TRIM: to start of Pandas datetime range
        
    da = da[da.year >= 1678].reset_index(drop=True)
    de = de[de.year >= 1678].reset_index(drop=True)
    ds = ds[ds.year >= 1678].reset_index(drop=True)
    
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)    
    sd_monthly = np.array( ds.groupby('year').mean().iloc[:,0:12]).ravel().astype(float)           

    # Solve Y1677-Y2262 Pandas bug with Xarray:        
#   t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')      
    t_monthly = pd.date_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M')     
    mask = np.array(len(ts_monthly) * [True])

    # COMPUTE: 12-m MA
    
    a = pd.Series(ts_monthly).rolling(12, center=True).mean().values
    e = pd.Series(ex_monthly).rolling(12, center=True).mean().values
    s = pd.Series(sd_monthly).rolling(12, center=True).mean().values

    t = t_monthly[mask]
    a = a[mask]
    e = e[mask]
    s = s[mask]

    diff_yearly = a - e
    
    # CALCULATE: CUSUM
        	
    c = np.nancumsum( diff_yearly )
    x = ( np.arange(len(c)) / len(c) )
    y = c    
    
    # CALL: cru_changepoint_detector

    #y_fit, y_fit_diff, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
    y_fit, y_fit_diff, y_fit_diff2, slopes, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
       
    if len(breakpoints) == 0:
        print('no breakpoints')
                                              
    data = [
        go.Table(
            header=dict(values=['Breakpoint number', 'Date'],
                line_color='darkslategray',
                fill_color='lightgrey',
                font = dict(color='Black'),
                align='left'),
            cells=dict(values=[
                np.arange(1,len(breakpoints)+1),t_monthly[breakpoints]
                ],
                line_color='slategray',
                fill_color='black',
                font = dict(color='white'),
                align='left')
        ),
    ]
    
    layout = go.Layout(
       template = "plotly_dark", 
#      template = None,
       height=400, width=550, margin=dict(r=10, l=10, b=10, t=10))

    return {'data': data, 'layout':layout} 

##################################################################################################
# Run the dash app
##################################################################################################

#if __name__ == "__main__":
#    app.run_server(debug=False)
    
