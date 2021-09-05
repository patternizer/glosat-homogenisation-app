![image](https://github.com/patternizer/glosat-homogenisation-app/blob/main/assets/station_745000_full.svg)

# glosat-homogenisation-app

Plotly python reactive dash app for inspecting GloSAT.p03 observations homogenised using [local expectation calculations with hold-out kriging](https://github.com/KCowtan/glosat-homogenisation/tree/main/local_expectation_krig) developed by Professor Kevin Cowtan. Part of ongoing work for the [GloSAT Project](https://www.glosat.org):

[Plotly Python Reactive Dash app](https://glosat-homogenisation-app.herokuapp.com/) for inspecting GloSAT.p03 station local expectation calculations, calculating changepoints and adjustments. 

## Contents

* `index.py` - python app wrapper script
* `app.py` - flask instance
* `assets/` - app imagery
* `apps/` - app page python code

Auxilliary code:

* `adjustments_histogram.py` - global station adjustment histogram accumulator
* `prepare_dataframes.py` - optimisation code to reduce global dataframe size on disk for deployment
* `ml_optimisation.csv` - LUT for DFT smoother code
* `filter_cru_dft.py` - DFT smoother code [repo](https://github.com/patternizer/glosat-dft-filter)
* `cru_changepoint_detector.py` - changepoint detector code [repo](https://github.com/patternizer/glosat-breakpoint-analysis)

The first step is to clone the latest glosat-homogenisation-app code and step into the check out directory: 

    $ git clone https://github.com/patternizer/glosat-homogenisation-app.git
    $ cd glosat-homogenisation-app

### Using Standard Python

The code should run with the [standard CPython](https://www.python.org/downloads/) installation and was tested 
in a conda virtual environment running a 64-bit version of Python 3.7+.

glosat-homogenisation-app scripts can be run from sources directly, once the dependencies in the requirements.txt are resolved and the global local expectation calculation results using hold-out kriging are placed in the DATA/ directory.  

Run and serve on localhost at http://127.0.0.1:8050 if called locally with:

    $ python index.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)



