#------------------------------------------------------------------------------
# PROGRAM: app.py (multipage)
#------------------------------------------------------------------------------
# Version 0.1
# 31 July, 2021
# Michael Taylor
# https://patternizer.github.io
# michael DOT a DOT taylor AT uea DOT ac DOT uk
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

import dash
import flask
import dash_core_components as dcc
import dash_html_components as html
#from dash import dcc
#from dash import html
import dash_bootstrap_components as dbc
import dash_leaflet as dl

external_stylesheets = [dbc.themes.DARKLY]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, prevent_initial_callbacks=False)
server = app.server
app.config.suppress_callback_exceptions = True

