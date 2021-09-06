#------------------------------------------------------------------------------
# PROGRAM: home.py
#------------------------------------------------------------------------------
# Version 0.1
# 31 July, 2021
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#import dash
#from dash import dcc
#from dash import html
#import dash_leaflet as dl
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

from apps import home, about, glosat
from app import app

image = 'url(/assets/station_745000_full.svg)'

layout = html.Div([
    dbc.Container([

        dbc.Row([
            dbc.Card(
                children=[
                    html.H5(children='Local Expectation Kriging Viewer App', className="text-center"),
                    html.A(dbc.CardImg(src='assets/station_745000_full.svg', top=False, style={'height':'70vh'}), href="/glosat" ), 
                ], 
            body=True, color="dark", outline=True), 
        ], justify="center"),
        
        html.Br(),

        dbc.Row([
            dbc.Col(
                dbc.Card(
                    children=[
                        html.H5(children='GloSAT Dataset', className="text-center"),
                        dbc.Row([
                            dbc.Col(
                                dbc.Button("Station Data", href="http://crudata.uea.ac.uk/cru/data/temperature/crutem4/station-data.htm", color="primary"),
                                className="mt-2", align="center"),
                            dbc.Col(dbc.Button("Temperatures", href="http://crudata.uea.ac.uk/cru/data/temperature/", color="primary"),
                                className="mt-2", align="center"),
                        ], justify="center")
                    ],
                body=True, color="dark", outline=True),
            width=4, className="mb-4"),

            dbc.Col(
                dbc.Card(
                    children=[
                        html.H5(children='Project Website', className="text-center"),
                        dbc.Button("GloSAT", href="https://www.glosat.org", color="primary", className="mt-2"),
                    ],
                body=True, color="dark", outline=True),
            width=4, className="mb-2"),

            dbc.Col(
                dbc.Card(
                    children=[
                        html.H5(children='Plotly Code', className="text-center"),
                        dbc.Button("GitHub", href="https://github.com/patternizer/glosat-homogenisation-app", color="primary", className="mt-2"),
                    ],
                body=True, color="dark", outline=True),
            width=4, className="mb-2"),
            
        ], className="mb-2"),

        html.P(['GloSAT Local Expectation Kriging Viewer is brought to you by ', html.A('Professor Kevin Cowtan', href='https://www.york.ac.uk/chemistry/staff/academic/a-c/kcowtan/'), ' at the University of York together with the ', html.A('Climatic Research Unit', href='http://www.cru.uea.ac.uk'), ' in the School of Environmental Sciences, University of East Anglia']),

    ])
])

