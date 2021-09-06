#------------------------------------------------------------------------------
# PROGRAM: glosat.py
#------------------------------------------------------------------------------
# Version 0.15
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
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
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

df_temp = pd.read_pickle('df_temp_expect.pkl', compression='bz2')

reduce_precision = False
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
    #df_temp['stationfirstyear'] = df_temp['stationfirstyear'].astype('int16')          # --> fails due to NaN being float
    #df_temp['stationlastyear'] = df_temp['stationlastyear'].astype('int16')            # --> fails due to NaN being float
    #df_temp['stationfirstreliable'] = df_temp['stationfirstreliable'].astype('int16')  # --> fails due to NaN being float
    #df_temp['stationsource'] = df_temp['stationsource'].astype('int16')                # --> fails due to NaN being float
    del df_temp['stationfirstyear']
    del df_temp['stationlastyear']
    del df_temp['stationfirstreliable']
    del df_temp['stationsource']
#    df_temp.to_pickle( "df_temp_expect.pkl", compression="bz2" )

# GENERATE: station code:name list

gb = df_temp.groupby(['stationcode'])['stationname'].unique().reset_index()
stationcodestr = gb['stationcode']
stationnamestr = [ gb['stationname'][i][0] for i in range(len(stationcodestr)) ]
stationstr = stationcodestr + ': ' + stationnamestr
opts = [{'label' : stationstr[i], 'value' : i} for i in range(len(stationstr))]

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

#        dbc.Row([
#            dbc.Col(html.Div([
#                dcc.Graph(id="plot-worldmap", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                    
#            ]), 
#            width={'size':6}, 
#            ),           
#            dbc.Col(html.Div([
#                dcc.Graph(id="plot-seasonal", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                    
#            ]), 
#            width={'size':6}, 
#            ),           
#        ]),
        
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
#               line_color='darkslategray',
#               fill_color='white',
                line_color='slategray',
                fill_color='black',
                font = dict(color='white'),
                align='left')
        ),
    ]
    layout = go.Layout(
       template = "plotly_dark", 
#      template = None,
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
#        px.scatter_mapbox(da, lat='stationlat', lon='stationlon', color_discrete_sequence=['rgba(234, 89, 78, 1.0)'], size_max=10, zoom=10, opacity=0.7)
        px.scatter_mapbox(lat=df['stationlat'], lon=df['stationlon'], text=df.index, color_discrete_sequence=['rgba(234, 89, 78, 1.0)'], zoom=3, opacity=0.7)
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

    df = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()
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
    ts_monthly = np.array( moving_average( ts_monthly, nsmooth ) )    
    ex_monthly = np.array( moving_average( ex_monthly, nsmooth ) )    
    sd_monthly = np.array( moving_average( sd_monthly, nsmooth ) )    
    # Solve Y1677-Y2262 Pandas bug with Xarray:        
    t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')     
    mask = np.isfinite(ex_monthly)
    if mask.sum() > 0:

        data = []
        
        trace_error = [
            go.Scatter(x=t_monthly[mask], y=ex_monthly[mask]+sd_monthly[mask], 
                       mode='lines', 
                       fill='none',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.0)'),             # grey                  
                       name='uncertainty',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
            ),
            go.Scatter(x=t_monthly[mask], y=ex_monthly[mask]-sd_monthly[mask], 
                       mode='lines', 
                       fill='tonexty',
                       fillcolor='rgba(242,242,242,0.2)',                               # grey
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.0)'),             # grey                  
                       name='uncertainty',      
#                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
            )]         

        trace_expect = [            
            go.Scatter(x=t_monthly, y=ex_monthly, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(234, 89, 78, 1.0)'),                 # red (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(234, 89, 78, 1.0)'),     # red (colorsafe)
                name='E',
#               hovertemplate='%{y:.2f}',
            )]

        trace_obs = [
            go.Scatter(x=t_monthly, y=ts_monthly, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(20,115,175,1.0)'),                  # blue (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(20,115,175,1.0)'),      # blue (colorsafe)
                name='O',
#               hovertemplate='%{y:.2f}',
            )]   

    data = data + trace_error + trace_expect + trace_obs
                                      
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis = dict(range=[t_monthly[0],t_monthly[-1]]),       
        yaxis_title = {'text': 'Anomaly (from 1961-1990), °C'},
        title = {'text': 'OBSERVATIONS (O) Vs LOCAL EXPECTATION (E)', 'x':0.1, 'y':0.95},                
    )

    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t_yearly[np.floor(len(t_monthly)/2).astype(int)],
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

    df = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()
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
    ts_monthly = np.array( moving_average( ts_monthly, nsmooth ) )    
    ex_monthly = np.array( moving_average( ex_monthly, nsmooth ) )    
    sd_monthly = np.array( moving_average( sd_monthly, nsmooth ) )    
    diff_monthly = ts_monthly - ex_monthly
    # Solve Y1677-Y2262 Pandas bug with Xarray:        
    t_monthly = xr.cftime_range(start=str(da['year'].iloc[0]), periods=len(ts_monthly), freq='M', calendar='noleap')     
    mask = np.isfinite(ex_monthly) & np.isfinite(ts_monthly)
    if mask.sum() > 0:

        data = []

        trace_error = [                    
            go.Scatter(x=t_monthly, y=sd_monthly, 
                       mode='lines', 
                       fill='none',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.0)'),             # grey                  
                       name='uncertainty',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
            ),
            go.Scatter(x=t_monthly, y=-sd_monthly, 
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
            go.Scatter(x=t_monthly[mask], y=diff_monthly[mask], 
                mode='lines+markers', 
                
#                line=dict(width=1.0, color='rgba(242,242,242,0.2)'),                   # grey                  
#                line=dict(width=1.0, color='rgba(234, 89, 78, 1.0)'),                  # red (colorsafe)                      
#                line=dict(width=1.0, color='rgba(20,115,175,1.0)'),                    # blue (colorsafe)
#                line=dict(width=1.0, color='rgba(137, 195, 239, 1.0)'),                # lightblue (colorsafe)                               
#                line=dict(width=1.0, color='rgba(229, 176, 57, 1.0)'),                 # mustard (colorsafe)

                line=dict(width=1.0, color='rgba(229, 176, 57, 1.0)'),                  # mustard (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(229, 176, 57, 1.0)'),      # mustard (colorsafe)
                name='O-E',
#               hovertemplate='%{y:.2f}',
            )]   
                                      
        data = data + trace_error + trace_diff
        
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis = dict(range=[t_monthly[0],t_monthly[-1]]),       
        yaxis_title = {'text': 'Anomaly (from 1961-1990), °C'},
        title = {'text': 'DIFFERENCE (O-E)', 'x':0.1, 'y':0.95},        
    )

    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t_yearly[np.floor(len(t_monthly)/2).astype(int)],
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

    df = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()
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

    # CALL: cru_changepoint_detector

    y_fit, y_fit_diff, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)

    print('BE WATER MY FRIEND')

    data = []
    
    trace_ltr_cumsum = [
           
        go.Scatter(x=x[mask], y=y[mask], 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(20,115,175,1.0)'),                     # blue (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(20,115,175,1.0)'),         # blue (colorsafe)
                name='CUSUM (O-E)',
#               hovertemplate='%{y:.2f}',
        )] 

    trace_ltr = [
           
        go.Scatter(x=x[mask], y=y_fit, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(234, 89, 78, 1.0)'),                   # red (colorsafe)
                marker=dict(size=2, opacity=0.5, color='rgba(234, 89, 78, 1.0)'),       # red (colorsafe)
                name='LTR fit',
#               hovertemplate='%{y:.2f}',
        )]
    
    trace_ltr_diff = [
           
        go.Scatter(x=x[mask], y=y_fit_diff, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(229, 176, 57, 1.0)'),                 # mustard (colorsafe)
                marker=dict(size=2, opacity=0.5, color='rgba(229, 176, 57, 1.0)'),     # mustard (colorsafe)
                name='d(LTR)',                
#               hovertemplate='%{y:.2f}', 
        )]
        
    trace_6_sigma = [                    
    
        go.Scatter(x=x[mask], y=np.tile( np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)), len(x[mask]) ), 
                       mode='lines', 
                       fill='none',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(229, 176, 57, 0.0)'),           # mustard (colorsafe)                  
                       name='6 sigma',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
        ),           
        go.Scatter(x=x[mask], y=np.tile( np.abs(np.nanmean(y_fit_diff)) - 6.0*np.abs(np.nanstd(y_fit_diff)), len(x[mask]) ), 
                       mode='lines', 
                       fill='tonexty',
                       fillcolor='rgba(229, 176, 57, 0.2)',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(229, 176, 57, 0.2)'),           # mustard (colorsafe)       
                       name='μ ± 6σ',      
#                      showlegend=False,
#                      hovertemplate='%{y:.2f}',                       
       )]
          	                                          
    data = data + trace_ltr_cumsum + trace_ltr + trace_ltr_diff + trace_6_sigma
                                     
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis = dict(range=[x[0],x[-1]]),       
        yaxis_title = {'text': 'CUSUM (O-E)'},
        title = {'text': 'CHANGEPOINTS', 'x':0.1, 'y':0.95},                          
    )

    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t_yearly[np.floor(len(t_monthly)/2).astype(int)],
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

    df = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()
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
    
    # CALL: cru_changepoint_detector

    y_fit, y_fit_diff, breakpoints, depth, r, R2adj = cru.changepoint_detector(x, y)
    
    print(len(breakpoints))
    
    if len(breakpoints) == 0:
    
        mask = np.array(len(x)*[False])
        data = []
        trace_obs = [
            go.Scatter(x=t_monthly, y=ts_monthly, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(20,115,175,0.0)'),                  # blue (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(20,115,175,0.0)'),      # blue (colorsafe)
                name='O',
#               hovertemplate='%{y:.2f}',
            )]       	
        data = data + trace_obs
            
    if mask.sum() > 0:

        breakpoints_all = x[mask][ np.abs(y_fit_diff) >= np.abs(np.nanmean(y_fit_diff)) + 6.0*np.abs(np.nanstd(y_fit_diff)) ][0:]    
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

        data = []
                         
        trace_error = [
            go.Scatter(x=t_monthly[mask], y=ex_monthly[mask]+sd_monthly[mask], 
                       mode='lines', 
                       fill='none',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.0)'),             # grey                  
                       name='uncertainty',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
            ),
            go.Scatter(x=t_monthly[mask], y=ex_monthly[mask]-sd_monthly[mask], 
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
            go.Scatter(x=t_monthly, y=ts_monthly, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(20,115,175,1.0)'),                  # blue (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(20,115,175,1.0)'),      # blue (colorsafe)
                name='O',
#               hovertemplate='%{y:.2f}',
            )]   

        trace_expect = [            
            go.Scatter(x=t_monthly, y=ex_monthly, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(234, 89, 78, 1.0)'),                 # red (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(234, 89, 78, 1.0)'),     # red (colorsafe)
                name='E',
#               hovertemplate='%{y:.2f}',
            )]

        trace_obs_adjusted = [
               
            go.Scatter(x=x[mask], y=ts_monthly + y_means, 
                    mode='lines+markers', 
                    line=dict(width=1.0, color='rgba(137, 195, 239, 1.0)'),                 # lightblue (colorsafe)
                    marker=dict(size=2, opacity=0.5, color='rgba(137, 195, 239, 1.0)'),     # lightblue (colorsafe)
                    name='O (adjusted)',                
    #               hovertemplate='%{y:.2f}', 
            )]
                    
        trace_adjustments = [
               
            go.Scatter(x=x[mask], y=y_means, 
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
        xaxis = dict(range=[x[0],x[-1]]),       
        yaxis_title = {'text': 'Anomaly (from 1961-1990), °C'},
        title = {'text': 'ADJUSTMENTS (E-O)', 'x':0.1, 'y':0.95},                           
    )

    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t_monthly[np.floor(len(t_monthly)/2).astype('int')],
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

    df = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True).dropna()
    dt = df.groupby('year').mean().iloc[:,0:12]
    dn_array = np.array( df.groupby('year').mean().iloc[:,19:31] )
    dn = dt.copy()
    dn.iloc[:,0:] = dn_array
    da = (dt - dn).reset_index()
    de = (df.groupby('year').mean().iloc[:,31:43]).reset_index()
    sd = (df.groupby('year').mean().iloc[:,43:55]).reset_index()      
    
    da = da[da.year >= 1678].reset_index(drop=True)
    de = de[de.year >= 1678].reset_index(drop=True)
    sd = sd[sd.year >= 1678].reset_index(drop=True)
          
    ts_monthly = np.array( da.groupby('year').mean().iloc[:,0:12]).ravel()    
    ex_monthly = np.array( de.groupby('year').mean().iloc[:,0:12]).ravel()    
    sd_monthly = np.array( sd.groupby('year').mean().iloc[:,0:12]).ravel()                   
    ts_monthly = np.array( moving_average( ts_monthly, nsmooth ) )    
    ex_monthly = np.array( moving_average( ex_monthly, nsmooth ) )    
    sd_monthly = np.array( moving_average( sd_monthly, nsmooth ) )    

    trim_months = len(ts_monthly)%12
 
    t_monthly = pd.date_range(start=str(da.year.iloc[0]), periods=len(ts_monthly), freq='MS')    
    df = pd.DataFrame({'Tg':ex_monthly[:-1-trim_months]}, index=t_monthly[:-1-trim_months])     
    
    t = [ pd.to_datetime( str(df.index.year.unique()[i])+'-01-01') for i in range(len(df.index.year.unique())) ][1:] # years
    DJF = ( df[df.index.month==12]['Tg'].values + df[df.index.month==1]['Tg'].values[1:] + df[df.index.month==2]['Tg'].values[1:] ) / 3
    MAM = ( df[df.index.month==3]['Tg'].values[1:] + df[df.index.month==4]['Tg'].values[1:] + df[df.index.month==5]['Tg'].values[1:] ) / 3
    JJA = ( df[df.index.month==6]['Tg'].values[1:] + df[df.index.month==7]['Tg'].values[1:] + df[df.index.month==8]['Tg'].values[1:] ) / 3
    SON = ( df[df.index.month==9]['Tg'].values[1:] + df[df.index.month==10]['Tg'].values[1:] + df[df.index.month==11]['Tg'].values[1:] ) / 3
    df_seasonal = pd.DataFrame({'DJF':DJF, 'MAM':MAM, 'JJA':JJA, 'SON':SON}, index = t)
          
    df_seasonal_ma = df_seasonal.rolling(10, center=True).mean() # decadal smoothing
    mask = np.isfinite(df_seasonal_ma)

#   dates = pd.date_range(start='1678-01-01', end='2021-12-01', freq='MS')
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
                line=dict(width=3, color='black'),
                marker=dict(size=7, symbol='square', opacity=1.0, color='blue', line_width=1, line_color='black'),                       
                name='DJF')
    ]
    trace_spring=[
            go.Scatter(                                  
                x=df_seasonal_fft.index[mask['MAM']], y=df_seasonal_fft['MAM'][mask['MAM']], 
                mode='lines+markers', 
                line=dict(width=3, color='black'),
                marker=dict(size=7, symbol='square', opacity=1.0, color='red', line_width=1, line_color='black'),       
                name='MAM')
    ]
    trace_summer=[
            go.Scatter(                                  
                x=df_seasonal_fft.index[mask['JJA']], y=df_seasonal_fft['JJA'][mask['JJA']], 
                mode='lines+markers', 
                line=dict(width=3, color='black'),
                marker=dict(size=7, symbol='square', opacity=1.0, color='purple', line_width=1, line_color='black'),       
                name='JJA')
    ]
    trace_autumn=[
            go.Scatter(                                  
                x=df_seasonal_fft.index[mask['SON']], y=df_seasonal_fft['SON'][mask['SON']], 
                mode='lines+markers', 
                line=dict(width=3, color='black'),
                marker=dict(size=7, symbol='square', opacity=1.0, color='green', line_width=1, line_color='black'),       
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
    fig.update_layout(legend=dict(
        orientation='v',
        yanchor="top",
        y=0.4,
        xanchor="left",
        x=0.8),
    )    
    fig.update_layout(height=400, width=550, margin={"r":10,"t":50,"l":70,"b":50})    

    return fig

##################################################################################################
# Run the dash app
##################################################################################################

#if __name__ == "__main__":
#    app.run_server(debug=False)
    
