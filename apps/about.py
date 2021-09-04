#------------------------------------------------------------------------------
# PROGRAM: about.py
#------------------------------------------------------------------------------
# Version 0.1
# 31 July, 2021
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_leaflet as dl

from apps import home, about, glosat
from app import app

layout = html.Div([
    dbc.Container([

        dbc.Card(                    
            children=[
                html.P([
                    html.H3(children=dbc.Button("Local Expectation Kriging Viewer App", href="/glosat", color="primary", className="mt-2"), className="text-center"),                              
                ], className="text-center"),
                html.Label(['Kevin Cowtan¹, Michael Taylor², Tim Osborn², Phil Jones²'], className="text-center"),
                html.P([html.A('¹Kevin Cowtan', href='https://www.york.ac.uk/chemistry/staff/academic/a-c/kcowtan/'), ', Department of Chemistry, University of York'], className="text-center"),                                
                html.P([html.A('²Climatic Research Unit', href='http://www.cru.uea.ac.uk'), ', School of Environmental Sciences, University of East Anglia'], className="text-center"),
                html.P(['This research was partly funded by the ', html.A('GloSAT Project', href='https://www.glosat.org/'), '.'], className="text-center"),  
            ], body=True, color="dark", outline=True), 
            
        html.Br(),
              
        dbc.CardDeck(
            [             
            dbc.Card(children=[html.H4(children='Theory', className="text-center"), 
                html.P(html.Br()),                               
                html.P("Coming soon ..."),                               
                html.P("© GloSAT all rights reserved"),
                html.Label(['The results, maps and figures shown on this website are licenced under an ', html.A('Open Government License', href='http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/'), '.']),                                                    
                ], body=True, color="dark", outline=True),                 
            ], className="mb-1"),

    ], className="mb-2"),
])

