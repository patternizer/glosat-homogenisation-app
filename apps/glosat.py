#------------------------------------------------------------------------------
# PROGRAM: glosat.py
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
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from apps import home, about, glosat
from app import app

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 12
nsmooth = 60
reduce_precision = False

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
#stationnamestr = gb['stationname'].apply(', '.join).str.lower()
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

#----------------------------------------------------------------------------------
import cru_changepoint_detector as cru # CRU changepoint detector
#----------------------------------------------------------------------------------

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
                dcc.Graph(id="plot-worldmap", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                    
            ]), 
            width={'size':6}, 
            ),           
        ]),

        dbc.Row([
            dbc.Col(html.Div([                    
                dcc.Graph(id="plot-differences", style = {'padding' : '10px', 'width': '100%', 'display': 'inline-block'}),                                     
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
                html.Br(),
                html.Label(['Status: Experimental']),
                html.Br(),
                html.Label(['Dataset: GloSAT.p03']),
                html.Br(),
                html.Label(['Codebase: ', html.A('Github', href='https://github.com/patternizer/glosat-homogenisation')]),                
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
    dists = prepare_dists( lats, lons )        
    da = df_temp[ df_temp['stationcode'] == df_temp['stationcode'].unique()[value] ].sort_values(by='year').reset_index(drop=True)    
    lat = [ df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationlat'].iloc[0] ]
    lon = [ df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationlon'].iloc[0] ]
    station = df_temp[df_temp['stationcode']==df_temp['stationcode'].unique()[value]]['stationcode'].iloc[0]
    
    df = df_temp.groupby('stationcode').mean()
    df['stationlat'] = [ np.round( df['stationlat'][i], 2) for i in range(len(df)) ]
    df['stationlon'] = [ np.round( df['stationlon'][i], 2) for i in range(len(df)) ]
    
    fig = go.Figure(
#	px.set_mapbox_access_token(open(".mapbox_token").read()),
#        px.scatter_mapbox(da, lat='stationlat', lon='stationlon', color_discrete_sequence=["red"], size_max=10, zoom=10, opacity=0.7)
#        px.scatter_mapbox(lat=df['stationlat'].round(2), lon=df['stationlon'].round(2), text=df.index, color_discrete_sequence=["red"], zoom=3, opacity=0.7)        
        px.scatter_mapbox(lat=df['stationlat'], lon=df['stationlon'], text=df.index, color_discrete_sequence=["red"], zoom=3, opacity=0.7)        
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

#   value = 5630

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
                       line=dict(width=1.0, color='rgba(242,242,242,0.2)'),             # grey                  
                       name='uncertainty',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
            ),
            go.Scatter(x=t_monthly[mask], y=ex_monthly[mask]-sd_monthly[mask], 
                       mode='lines', 
                       fill='tonexty',
                       fillcolor='rgba(242,242,242,0.1)',                               # grey
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.2)'),             # grey                  
                       name='uncertainty',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
            )]         

        trace_expect = [            
            go.Scatter(x=t_monthly, y=ex_monthly, 
                mode='lines+markers', 
                line=dict(width=1.0, color='rgba(234, 89, 78, 1.0)'),                 # orange (colorsafe)
                marker=dict(size=5, opacity=0.5, color='rgba(234, 89, 78, 1.0)'),     # orange (colorsafe)
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
                    text="No baseline anomaly",
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
                       line=dict(width=1.0, color='rgba(242,242,242,0.2)'),             # grey                  
                       name='uncertainty',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
            ),
            go.Scatter(x=t_monthly, y=-sd_monthly, 
                       mode='lines', 
                       fill='tonexty',
                       fillcolor='rgba(242,242,242,0.1)',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(242,242,242,0.2)'),             # grey                  
                       name='uncertainty',      
                       showlegend=False,
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
        title = {'text': 'DIFFERENCE', 'x':0.1, 'y':0.95},        
    )

    if mask.sum() == 0:
        fig.update_layout(
            annotations=[
                dict(
                    x=t_yearly[np.floor(len(t_monthly)/2).astype(int)],
                    y=0,
                    xref="x",
                    yref="y",
                    text="No baseline anomaly",
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
                       line=dict(width=1.0, color='rgba(229, 176, 57, 0.2)'),           # mustard (colorsafe)                  
                       name='6 sigma',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',
        ),           
        go.Scatter(x=x[mask], y=np.tile( np.abs(np.nanmean(y_fit_diff)) - 6.0*np.abs(np.nanstd(y_fit_diff)), len(x[mask]) ), 
                       mode='lines', 
                       fill='tonexty',
                       fillcolor='rgba(229, 176, 57, 0.1)',
                       connectgaps=True,
                       line=dict(width=1.0, color='rgba(229, 176, 57, 0.2)'),           # mustard (colorsafe)       
                       name='6 sigma',      
                       showlegend=False,
#                      hovertemplate='%{y:.2f}',                       
       )]
                
    data = data + trace_ltr_cumsum + trace_ltr + trace_ltr_diff + trace_6_sigma
                                      
    fig = go.Figure(data)
    fig.update_layout(
        template = "plotly_dark",
#       template = None,
        xaxis = dict(range=[x[0],x[-1]]),       
        yaxis_title = {'text': 'CUSUM'},
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
                    text="No baseline anomaly",
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
       
##################################################################################################
# Run the dash app
##################################################################################################

#if __name__ == "__main__":
#    app.run_server(debug=False)
    
