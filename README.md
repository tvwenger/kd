# Kinematic Distance Utilities
Utilities to calculate kinematic distances and kinematic distance uncertainties. [See Wenger et al. (2018)](http://adsabs.harvard.edu/abs/2018ApJ...856...52W). An on-line tool which uses this code to compute kinematic distances is available here (http://www.treywenger.com/kd/). If you use this work, please reference both the code and the paper:
1. [Wenger et al. (2018)](http://ui.adsabs.harvard.edu/abs/2018ApJ...856...52W)
2. https://ascl.net/1712.001
3. https://zenodo.org/record/1166001

## New in Version 2.0
0. Re-added multiprocessing from `pdf_kd` and `pdf_parallax`
1. Added support for [Reid et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019ApJ...885..131R/abstract) rotation curve.
2. Added parameter covariences for Reid et al. (2019) rotation curve.
3. Fixed minor bug in Reid et al. (2014) rotation curve.
4. Updated distance uncertainties to use minimum width Bayesian credible interval (BCI).
5. General improvements in readability and performance.

## Requirements
The following packages are required for this module to work "out-of-the-box"
1. `numpy`
2. `scipy`
3. `matplotlib`
4. `pyqt_fit` (N.B. The currently available version on PyPI, as of January 2020, has a dependency issue. I have [forked the repository](https://github.com/tvwenger/pyqt-fit) and corrected the issue.)

The easiest way to install this package is
```bash
pip install git+https://github.com/tvwenger/pyqt-fit.git
pip install git+https://github.com/tvwenger/kd.git
```

Alternatively, download the code and install:
```bash
git clone https://github.com/tvwenger/pyqt-fit.git; cd pyqt-fit; python setup.py install; cd ..
git clone https://github.com/tvwenger/kd.git; cd kd; python setup.py install; cd ..
```

## General Utilities
The script `kd_utils.py` includes several functions relevant to computing kinematic distances:
1. `calc_Rgal(glong, dist)` computes the Galactocentric radius for a given Galactic longitude, `glong`, and distance, `dist`
2. `calc_az(glong, dist)` computes the Galactocentric azimuth for a given Galactic longitude, `glong`, and distance, `dist`
3. `calc_dist(az, Rgal)` computes the distance of a given Galactocentric radius, `Rgal`, and azimuth, `az`
4. `calc_glong(az, Rgal)` computes the Galactic longitude of a given Galactocentric radius, `Rgal`, and azimuth, `az`
5. `correct_vlsr(glong, glat, vlsr)` computes the corrected LSR velocity given updated solar motion parameters.
6. `calc_anderson2012_uncertainty(glong, vlsr)` returns the Anderson et al. (2012) kinematic distance uncertainties.
7. `calc_hpd(samples)` returns the kernel density estimator (KDE) fit to some samples, as well as the mode (most likely value) and highest posterior density (HPD), which is the minimum width Bayesian credible interval (BCI).

Each of these functions may be invoked to compute the value at a given position or multiple positions at once:
```python
import numpy as np
from kd import kd_utils
glong = np.array([30.,130.])
dist = np.array([3.,5.])
kd_utils.calc_Rgal(glong,dist) # array([  5.93461783,  12.17226711]) (kpc)
```

## Rotation Curves
This module includes three rotation curves: 
1. Brand, J., & Blitz, L. 1993, A&A, 275, 67 (`brand_rotcurve.py`)
2. Reid, M. J., Menten, K. M., Brunthaler, A., et al. 2014, ApJ, 783, 130 (`reid14_rotcurve.py`)
3. Reid, M. J., Menten, K. M., Brunthaler, A., et al. 2019, ApJ, 885, 131 (`reid19_rotcurve.py`)

These rotation curve scripts, and any new rotation curve scripts you wish to add, *must* include the following four functions:
1. `params_nominal()` which returns the nominal rotation curve parameters as a dictionary
2. `params_resample(size)` which returns resampled rotation curve parameters as a dictionary
3. `calc_theta(R)` which returns the circular orbital speed, `theta`, at a given Galactocentric radius, `R`
4. `calc_vlsr(glong, dist)` which returns the LSR velocity at a given Galactic longitude, `glong`, and distance, `dist`

These scripts may be invoked to compute the circular orbit speed or LSR velocity for a single position or multiple positions at once:

```python
import numpy as np
from kd import reid14_rotcurve
R = np.array([4.,6.])
reid14_rotcurve.calc_theta(R) # array([ 223.40500419,  238.51262935]) (km/s)
glong = np.array([30.,130.])
dist = np.array([3.,5.])
reid14_rotcurve.calc_vlsr(glong,dist) # (array([ 46.94466602, -58.85105356]), (km/s)
```

## Traditional Kinematic Distance
The traditional kinematic distance is derived by finding the minimum difference between the rotation curve LSR velocity and the measured LSR velocity of an object. The script `rotcurve_kd.py` computes this traditional kinematic distance. The syntax is
```python
from kd import rotcurve_kd
glong = 30. # Galactic longitude, degrees
velo = 20. # measured LSR velocity, km/s
velo_tol = 0.1 # tolerance to determine a "match" between rotation curve and measured LSR velocity (km/s)
rotcurve = 'reid14_rotcurve' # the name of the script containing the rotation curve
rotcurve_kd.rotcurve_kd(glong,velo,velo_tol=velo_tol,rotcurve=rotcurve)
# {'Rgal': 7.1412985771972739,        Galactocentric radius (kpc)
# 'Rtan': 4.1700008432135309,         Galactocentric radius of the tangent point (kpc)
# 'far': 13.02,                       Far kinematic distance (kpc)
# 'near': 1.4199999999999999,         Near kinematic distance (kpc)
# 'tangent': 7.2199999999999998,      Distance of the tangent point (kpc)
# 'vlsr_tangent': 105.15666704280204} LSR velocity of the tangent point (km/s)
```

## Monte Carlo Kinematic Distance
The Monte Carlo kinematic distance is derived by resampling the measured LSR velocity and rotation curve parameters within their uncertainties. The script `pdf_kd.py` computes the Monte Carlo kinematic distance. The syntax is
```python
from kd import pdf_kd
glong = 30. # Galactic longitude, degrees
velo = 20. # measured LSR velocity, km/s
velo_err = 5. # measured LSR velocity uncertainty, km/s
rotcurve = 'reid14_rotcurve' # the name of the script containing the rotation curve
num_samples = 10000 # number of re-samples
pdf_kd.pdf_kd(glong,velo,velo_err=velo_err,rotcurve=rotcurve,num_samples=num_samples)
# {'Rgal': 7.1124726106671012,                        Galactocentric radius (kpc)
# 'Rgal_err_neg': 0.27321259042889601,                Galactocentric radius uncertainty toward the Galactic Center (kpc)
# 'Rgal_err_pos': 0.34151573803611868,                Galactocentric radius uncertainty away from the Galactic Center (kpc)
# 'Rgal_kde': <pyqt_fit.kde.KDE1D at 0x7fc41ee2fa90>, Kernel Density Estimator (KDE) fit to the Rgal probability distribution function (PDF)
# 'Rtan': 4.1813677581628541,                         Galactocentric radius of the tangent point (kpc)
# 'Rtan_err_neg': 0.090269992563094092,               Uncertainty toward the Galactic Center (kpc)
# 'Rtan_err_pos': 0.067702494422319681,               Uncertainty away from the Galactic Center (kpc)
# 'Rtan_kde': <pyqt_fit.kde.KDE1D at 0x7fc41ee2fa58>, KDE fit to the Rtan PDF
# 'far': 12.977444444444433,                          Far kinematic distance (kpc)
# 'far_err_neg': 0.3764848484848482,                  Uncertainty toward the Sun (kpc)
# 'far_err_pos': 0.47060606060606069,                 Uncertainty away from the Sun (kpc)
# 'far_kde': <pyqt_fit.kde.KDE1D at 0x7fc41ee2f898>,  KDE fit to the far PDF
# 'near': 1.4793636363636349,                         Near kinematic distance (kpc)
# 'near_err_neg': 0.40581818181818141,                Uncertainty toward the Sun (kpc)
# 'near_err_pos': 0.30436363636363617,                Uncertainty away from the Sun (kpc)
# 'near_kde': <pyqt_fit.kde.KDE1D at 0x7fc41ee2fcc0>, KDE fit to the near PDF
# 'tangent': 7.2519999999999936,                      Distance of the tangent point (kpc)
# 'tangent_err_neg': 0.16622222222222227,             Uncertainty toward the Sun (kpc)
# 'tangent_err_pos': 0.10755555555555496,             Uncertainty away from the Sun (kpc)
# 'tangent_kde': <pyqt_fit.kde.KDE1D at 0x7fc41ee2f6d8>, KDE fit to the tangent PDF
# 'vlsr_tangent': 103.2468359294412,                  LSR velocity of the tangent point (km/s)
# 'vlsr_tangent_err_neg': 7.8548626921567717,         Uncertainty toward negative LSR velocity (km/s)
# 'vlsr_tangent_err_pos': 12.084404141779657,         Uncertainty toward positive LSR velocity (km/s)
# 'vlsr_tangent_kde': <scipy.stats.kde.gaussian_kde at 0x7fc41ee2f828>} KDE fit to vlsr_tangent PDF
```
