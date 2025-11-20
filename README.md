# finding_exoplanet
finding_exoplanet


Steps for project:
Using lightkurve python package, get light curves.
1. Get list of stars, some with confirmed exoplanets, some without. For the stars with confirmed exoplanets, also get orbital periods
2. For each of those stars, use lightkurve to get the lightcurves. Tutorial
3. Feed lightcurves into models of our choice
  - Bayesian Model
    - From lecture on detecting periodic signals
  - ML Model
    - Idk what architecture works best, this is something we should choose
