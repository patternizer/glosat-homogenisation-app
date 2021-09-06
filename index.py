#------------------------------------------------------------------------------
# PROGRAM: index.py
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
#from dash import dcc
#from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
#import dash_leaflet as dl

from app import app, server
from apps import home, about, glosat

# CONSTRUCT: nav bar

dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Home", href="/home"),
        dbc.DropdownMenuItem("About", href="/about"),
        dbc.DropdownMenuItem("App", href="/glosat"),
    ],
    nav = True,
    in_navbar = True,
    label = "Explore",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/logo-glosat-dark.png", height="70px")),
                        dbc.Col(dbc.NavbarBrand("Local Expectation Kriging Viewer", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=False,
                ),
                href="/home",
            ),

#            dbc.NavbarToggler(id="navbar-toggler2"),
#            dbc.Collapse(
#                dbc.Nav(
#                    [dropdown], className="ml-auto", navbar=True
#                ),
#                id="navbar-collapse2",
#                navbar=True,
#            ),

        ]
    ),
    color="#1c1f2b",
    dark=True,
    className="mb-4",
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [3]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# EMBED: nav bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):

    if pathname == '/glosat':
        return glosat.layout
    elif pathname == '/about':
        return about.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True)

