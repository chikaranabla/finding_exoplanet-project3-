# finding_exoplanet
finding_exoplanet


Proposed steps for project:
Using lightkurve python package, get light curves.
1. Get list of stars, some with confirmed exoplanets, some without. For the stars with confirmed exoplanets, also get orbital periods. [Possible Dataset](https://www.kaggle.com/datasets/vijayveersingh/kepler-and-tess-exoplanet-data/data?select=keplerstellar_2025.02.03_04.41.47.csv)
2. For each of those stars, use lightkurve to get the lightcurves. [Tutorial](https://lightkurve.github.io/lightkurve/tutorials/1-getting-started/searching-for-data-products.html#2.-Searching-for-Light-Curves)
3. Feed lightcurves into models of our choice
  - Bayesian Model
    - From lecture on detecting periodic signals
  - ML Model
    - Idk what architecture works best, this is something we should choose
