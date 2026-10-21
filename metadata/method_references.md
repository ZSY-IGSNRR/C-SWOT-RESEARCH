# Methods and interpretation references

These references are transcribed from the v15 bibliography. The release itself
should also be cited using CITATION.cff or citation.bib. Product identifiers are
listed separately in source_products.csv.

- Durand, M., et al. (2023). A framework for estimating global river discharge
  from the Surface Water and Ocean Topography satellite mission. Water Resources
  Research, 59(4), e2021WR031614. https://doi.org/10.1029/2021WR031614
- Durand, M., et al. (2016). An intercomparison of remote sensing river discharge
  estimation algorithms from measurements of river height, width, and slope.
  Water Resources Research, 52(6), 4527–4549. https://doi.org/10.1002/2015WR018434
- Garambois, P.-A., & Monnier, J. (2015). Inference of effective river properties
  from remotely sensed observations of water surface. Advances in Water Resources,
  79, 103–120. https://doi.org/10.1016/j.advwatres.2015.02.007
- Rodríguez, E., Durand, M., & Frasson, R. P. D. M. (2020). Observing rivers with
  varying spatial scales. Water Resources Research, 56(9), e2019WR026476.
  https://doi.org/10.1029/2019WR026476

The fixed-geometry n-only configuration is a simplified local comparator, not
a reproduction of the entire original Durand framework. The SoS comparison
uses existing unconstrained products, including MOMMA, and assesses the local
implementation together with its additional gauge information.

## Current fivefold method citations

These references were checked against the 21 October manuscript. Stone supports cross-validation generally; Klemeš and Roberts et al. support matching validation to the prediction task. They do not prescribe this study's exact WSE assignment. The original objective weights, threshold values, width safeguards and numerical settings are study-specific.

- Andreadis, K. M., Coss, S. P., Durand, M., Gleason, C. J., Simmons, T. T., Tebaldi, N., Bjerklie, D. M., Brinkerhoff, C., Dudley, R. W., Gejadze, I., Larnier, K., Malaterre, P.-O., Oubanas, H., Allen, G. H., Bates, P. D., David, C. H., Domeneghetti, A., Elmi, O., Fenoglio Marc, L., … Yadav, B. (2025). A first look at river discharge estimation from SWOT satellite observations. Geophysical Research Letters, 52(9), Article e2024GL114185. https://doi.org/10.1029/2024GL114185

- Chai, T., & Draxler, R. R. (2014). Root mean square error (RMSE) or mean absolute error (MAE)? – Arguments against avoiding RMSE in the literature. Geoscientific Model Development, 7(3), 1247–1250. https://doi.org/10.5194/gmd-7-1247-2014

- Klemeš, V. (1986). Operational testing of hydrological simulation models. Hydrological Sciences Journal, 31(1), 13–24. https://doi.org/10.1080/02626668609491024

- Rousseeuw, P. J., & Croux, C. (1993). Alternatives to the median absolute deviation. Journal of the American Statistical Association, 88(424), 1273–1283. https://doi.org/10.1080/01621459.1993.10476408

- Spearman, C. (1904). The proof and measurement of association between two things. The American Journal of Psychology, 15(1), 72–101. https://doi.org/10.2307/1412159

- Stone, M. (1974). Cross-validatory choice and assessment of statistical predictions. Journal of the Royal Statistical Society: Series B (Methodological), 36(2), 111–133. https://doi.org/10.1111/j.2517-6161.1974.tb00994.x

- Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W., Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Larson, E., … SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2


- de Leeuw, J., Hornik, K., & Mair, P. (2009). Isotone optimization in R: Pool-adjacent-violators algorithm (PAVA) and active set methods. Journal of Statistical Software, 32(5), 1–24. https://doi.org/10.18637/jss.v032.i05

- Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling. Journal of Hydrology, 377(1–2), 80–91. https://doi.org/10.1016/j.jhydrol.2009.08.003

- Le Coz, J., Renard, B., Bonnifait, L., Branger, F., & Le Boursicaud, R. (2014). Combining hydraulic knowledge and uncertain gaugings in the estimation of hydrometric rating curves: A Bayesian approach. Journal of Hydrology, 509, 573–587. https://doi.org/10.1016/j.jhydrol.2013.11.016

- Nash, J. E., & Sutcliffe, J. V. (1970). River flow forecasting through conceptual models part I—A discussion of principles. Journal of Hydrology, 10(3), 282–290. https://doi.org/10.1016/0022-1694(70)90255-6

- Pearson, R. K., Neuvo, Y., Astola, J., & Gabbouj, M. (2016). Generalized Hampel filters. EURASIP Journal on Advances in Signal Processing, 2016, Article 87. https://doi.org/10.1186/s13634-016-0383-6

- Powell, M. J. D. (1964). An efficient method for finding the minimum of a function of several variables without calculating derivatives. The Computer Journal, 7(2), 155–162. https://doi.org/10.1093/comjnl/7.2.155

- Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., & Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913–929. https://doi.org/10.1111/ecog.02881

- Storn, R., & Price, K. (1997). Differential evolution – A simple and efficient heuristic for global optimization over continuous spaces. Journal of Global Optimization, 11(4), 341–359. https://doi.org/10.1023/a:1008202821328
