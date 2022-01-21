<img src="./docs/sphinx/source/_images/logo_horizontal_highres.png" width="500" alt="RdTools logo"/>

Master branch: 
[![Build Status](https://github.com/NREL/rdtools/workflows/pytest/badge.svg?branch=master)](https://github.com/NREL/rdtools/actions?query=branch%3Amaster)  

Development branch: 
[![Build Status](https://github.com/NREL/rdtools/workflows/pytest/badge.svg?branch=development)](https://github.com/NREL/rdtools/actions?query=branch%3Adevelopment)

RdTools is an open-source library to support reproducible technical analysis of
time series data from photovoltaic energy systems. The library aims to provide
best practice analysis routines along with the building blocks for users to
tailor their own analyses.
Current applications include the evaluation of PV production over several years to obtain
rates of performance degradation and soiling loss. RdTools can handle
both high frequency (hourly or better) or low frequency (daily, weekly,
etc.) datasets. Best results are obtained with higher frequency data.

RdTools can be installed automatically into Python from PyPI using the
command line:

```
pip install rdtools
```

For API documentation and full examples, please see the [documentation](https://rdtools.readthedocs.io).

RdTools currently is tested on Python 3.7+.

## Citing RdTools

To cite RdTools, please use the following along with the version number
and the specific DOI coresponding to that version from [Zenodo](https://doi.org/10.5281/zenodo.1210316):

-  Michael G. Deceglie, Ambarish Nag, Adam Shinn, Gregory Kimball,
   Daniel Ruth, Dirk Jordan, Jiyang Yan, Kevin Anderson, Kirsten Perry,
   Mark Mikofski, Matthew Muller, Will Vining, and Chris Deline
   RdTools, version {insert version}, Compuer Software,
   https://github.com/NREL/rdtools. DOI:{insert DOI}

The underlying workflow of RdTools has been published in several places.
If you use RdTools in a published work, you may also wish to cite the following as
appropriate:

-  Dirk Jordan, Chris Deline, Sarah Kurtz, Gregory Kimball, Michael Anderson, "Robust PV
   Degradation Methodology and Application", IEEE Journal of
   Photovoltaics, 8(2) pp. 525-531, 2018, DOI: [10.1109/JPHOTOV.2017.2779779](https://doi.org/10.1109/JPHOTOV.2017.2779779)

-  Michael G. Deceglie, Leonardo Micheli and Matthew Muller, "Quantifying Soiling Loss
   Directly From PV Yield," in IEEE Journal of Photovoltaics, 8(2),
   pp. 547-551, 2018, DOI: [10.1109/JPHOTOV.2017.2784682](https://doi.org/10.1109/JPHOTOV.2017.2784682)

-  Kevin Anderson and Ryan Blumenthal, "Overcoming Communications Outages in
   Inverter Downtime Analysis", 2020 IEEE 47th Photovoltaic Specialists
   Conference (PVSC)" DOI: [10.1109/PVSC45281.2020.9300635](https://doi.org/10.1109/PVSC45281.2020.9300635)

-  Kirsten Perry, Matthew Muller and Kevin Anderson, "Performance Comparison of Clipping
   Detection Techniques in AC Power Time Series," 2021 IEEE 48th Photovoltaic
   Specialists Conference (PVSC), pp. 1638-1643 2021, DOI: [10.1109/PVSC43889.2021.9518733](https://doi.org/10.1109/PVSC43889.2021.9518733).

  
## References
The clear sky temperature calculation, `clearsky_temperature.get_clearsky_tamb()`, uses data
from images created by Jesse Allen, NASA’s Earth Observatory using data courtesy of the MODIS Land Group.  
https://neo.sci.gsfc.nasa.gov/view.php?datasetId=MOD_LSTD_CLIM_M  
https://neo.sci.gsfc.nasa.gov/view.php?datasetId=MOD_LSTN_CLIM_M

Other useful references which may also be consulted for degradation rate methodology include:

  - D. C. Jordan, M. G. Deceglie, S. R. Kurtz, "PV degradation methodology comparison — A basis for a standard", in 43rd IEEE Photovoltaic Specialists Conference, Portland, OR, USA, 2016, DOI: 10.1109/PVSC.2016.7749593.
  - Jordan DC, Kurtz SR, VanSant KT, Newmiller J, Compendium of Photovoltaic Degradation Rates, Progress in Photovoltaics: Research and Application, 2016, 24(7), 978 - 989.
  - D. Jordan, S. Kurtz, PV Degradation Rates – an Analytical Review, Progress in Photovoltaics: Research and Application, 2013, 21(1), 12 - 29.
  - E. Hasselbrink, M. Anderson, Z. Defreitas, M. Mikofski, Y.-C.Shen, S. Caldwell, A. Terao, D. Kavulak, Z. Campeau, D. DeGraaff, "Validation of the PVLife model using 3 million module-years of live site data", 39th IEEE Photovoltaic Specialists Conference, Tampa, FL, USA, 2013, p. 7 – 13, DOI: 10.1109/PVSC.2013.6744087.

## Further Instructions and Updates

Check out the [wiki](https://github.com/NREL/rdtools/wiki) for additional usage documentation, and for information on development goals and framework.

