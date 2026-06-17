Datasets:
JMulTi http://www.jmulti.de/data_atse.html  Applied Time Series Econometrics – Helmut Lutkepohl
German GNP series: the break date is known to be the third quarter of 1990
U.S. investment and German interest rate series. In no case can the stationarity null hypothesis be rejected at the 5% level for the U.S. investment series, whereas it is rejected for the German interest rate. This result corresponds quite nicely to rejecting the unit root for the U.S. investment series and not rejecting it for the interest rate with the ADF and Schmidt–Phillips tests. Notice, however, that using lq = 12 for the investment series results in a test value that is significant at the 10% level. Still, taking all the evidence together, we find that the series is better viewed as I(0) than as I(1).

seasonally adjusted, quarterly German consumption data for the period 1960Q1 − 1982Q4. The series is the consumption series given in Table E.1 of Lutkepohl (1991). They reveal that constructing a model for the logs is likely to be advantageous because the changes in the log series display a more stable variance than the changes
in the original series. At the 5% level the unit root cannot be rejected. The corresponding KPSS test, again with a constant and a time trend, confirms the result. It clearly rejects stationarity at the 5% level. This outcome results with a lag truncation parameter of three, but a rejection of stationarity is also obtained with other lag orders. Overall, the test results support one unit root in the log consumption series; thus, specifying a stationary model for the first differences seems appropriate.

The second example series consists of the logarithms of a seasonally unadjusted quarterly Polish productivity series for the period 1970Q1–1998Q4. Although the series is not seasonally adjusted, it does not have a very clear seasonal pattern. the presence of a structural shift in 1990Q1 for Polish productivity. case the unit root is clearly rejected even at the 1% level. Hence, we continue the analysis with the first differences of the series and include an impulse dummy as a deterministic term in addition to a constant.

https://www.time-series.net/data_sets  - Applied Econometric Time Series – Walter Enders



Wei: W1 series: stationary

W2 series stationary in mean but may not in variance
W3: series stationary in mean but may not in variance
Blow-fly data: typo at t=20
W4: non stat in mean 

W5: non stat in mean with increasing trend
W6: non stat in mean and variance
W7: log series is stat
W10 -  seasonal quarterly – seasonal unit root

Monthly O3 readings: January 1960 break


Shumway & Stoffer: Time Series Analysis and Its Applications With R Examples – 3rd edition
Twenty-four month forecasts for the Recruitment series. The actual data shown are from about January 1980 to September 1987 – stationary
In this example, we consider the analysis of quarterly U.S. GNP from 1947(1) to 2002(3), n = 223 observations. The data are real U.S. gross national product in billions of chained 1996 dollars and have been seasonally adjusted. The data were obtained from the Federal Reserve Bank of St. Louis (http://research.stlouisfed.org/). the trend has been removed we are able to notice that the variability in the second half of the data is larger than in the first half of the data. Also, it appears as though a trend is still present after differencing. ARIMA(1; 1; 0) for log GNP.

the Monthly Federal Reserve Board Production Index and Unemployment (1948-1978, n = 372 months). ARIMA(0; 1; 1)  (0; 1; 1)12
Periodogram of SOI and Recruitment, n = 453 (n0 = 480), where the frequency axis is labeled in multiples of  = 1=12. one cycle per year (12 months), one cycle every four years (48 months).

Brockwell and Davis Time Series: Theory and Methods 1991 Brockwell & Davis Data Files http://users.stat.umn.edu/~kb/classes/5932/BDFiles.html
Population of the U.S.A. at ten-year intervals, 1 790- 1980 - the second to a roughly exponentially increasing graph, 
the third to a graph which fluctuates erratically about a nearly constant or slowly rising level, Strikes in the U.S.A., 1 95 1 - 1 980 (Bureau of Labor Statistics, U.S. Labor Department).
the fourth to an erratic series of minus ones and ones. All-star baseball games, 1933 - 1 980.
The fifth graph appears to have a strong cyclic component with period about 11 years The Wolfer sunspot numbers, 1770- 1869.
and the last has a pronounced seasonal component with period 12. Monthly accidental deaths in the U.S.A., 1 973 - 1 978 (National Safety Council). The Small Trend
International airline passengers; monthly totals in tlrousands of passengers { U, t = I , . . . , 144} from January 1949 to December 1960 (Box and Jenkins ( 1970)). Regular and seasonal stochastic trend

Peter J. Brockwell Richard A. Davis – Introduction to Time Series and Forecasting Second Edition
Monthly accidental deaths regular and seasonal unit root



Bai-Perron: Bai, J. and Perron, P. (2003), Computation and analysis of multiple structural change models. J. Appl. Econ., 18: 1-22. https://doi.org/10.1002/jae.659
US ex-post real interest rate 1961:1–1986:3 The break dates are estimated at 1966:4, 1972:3 and 1980:3.
POST-war UK inflation rate 1947–1987 The first break date is the same as in the one-break model, namely 1967, which is linked to the end of the Bretton Woods system. The second break is located in 1975.
Eric Zivot and Donald W. K. Andrews, Further Evidence on the Great Crash, the Oil-Price Shock, and the Unit-Root Hypothesis, Journal of Business & Economic Statistics , Jul., 1992, Vol. 10, No. 3 (Jul., 1992),pp. 251-270
jstor.org/stable/pdf/1391541.pdf?refreqid=fastly-default%3A12d20c0e8bcfe309f5aa55679e859d51&ab_segments=&initiator=&acceptTC=1



Chow test: https://web.archive.org/web/20191228155733/http://pdfs.semanticscholar.org/0f70/219160c8ad2f9db02e226d3f7d7320e729b8.pdf
RegressionsGregory C. Chow, Tests of Equality Between Sets of Coefficients in Two Linear, Econometrica, Vol. 28, No. 3. (Jul., 1960), pp. 591-605.



https://arxiv.org/pdf/2601.02957v3  

The datasets represent different types of realworld changepoints: Nile (annual river flow, 1871–1970) shows a mean shift when the Aswan Low Dam construction began in 1898 (Cobb, 1978); Seatbelts (monthly UK road casualties, 1976– 1984) exhibits a sudden drop following the 1983
compulsory seatbelt law (Harvey and Durbin, 1986); LGA (monthly LaGuardia Airport passengers, 1977–2015) captures the immediate and sustained impact of the September 11 attacks on air travel (Ito and Lee, 2005); Ireland Debt (annual debt-to-GDP ratio, 2000–2020) shows the dramatic surge following the 2008 banking crisis and subsequent bailout (Lane, 2011); Ozone (annual Antarctic  zone measurements, 1961–2014) marks the reversal point when Montreal Protocol effects began (Solomon et al., 2016); Robocalls (monthly US call volume, 2015–2019) increases sharply after a 2018 federal court ruling loosened FCC restrictions; and Japan Nuclear (annual nuclear share of electricity, 1985–2024) drops precipitously after the 2011 Fukushima disaster led to reactor shutdowns (Hayashi and Hughes, 2013). Most datasets are sourced from the Turing Change Point Dataset (van den Burg and Williams, 2020). https://github.com/alan-turing-institute/TCPD/tree/master/datasets 




