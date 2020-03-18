#!/usr/bin/env python
"""
kd_utils.py

Utility functions for rotcurve_kd.py, pdf_kd.py, and rotation curves.

Copyright(C) 2017-2020 by
Trey V. Wenger; tvwenger@gmail.com

GNU General Public License v3 (GNU GPLv3)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

2017-04-12 Trey V. Wenger
2018-02-10 Trey V. Wenger added correct_vlsr
2019-01-18 Trey V. Wenger removed pool_wait
                          added function to compute Anderson+2012
                          kinematic distance uncertainties
2020-02-19 Trey V. Wenger updates for v2.0
"""

import os
import numpy as np
from scipy.io import readsav
from scipy.stats.kde import gaussian_kde
from pyqt_fit import kde as pyqt_kde
from pyqt_fit import kde_methods

# IAU-defined solar motion parameters (km/s)
__Ustd = 10.27
__Vstd = 15.32
__Wstd = 7.74

# Reid+2019 Galactocentric radius and solar motion parameters
__R0 = 8.15 # kpc
__Usun = 10.6 # km/s
__Vsun = 10.7 # km/s
__Wsun = 7.6 # km/s

def calc_Rgal(glong, dist, R0=__R0):
    """
    Return the Galactocentric radius of an object with a given
    Galacitic longitude and distance.

    Parameters:
      glong :: scalar or array of scalars
        Galactic longitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      R0 :: scalar (optional)
        Galactocentric radius of the Sun.

    Returns: R
      Rgal :: scalar or array of scalars
        Galactocentric radius (kpc).
    """
    #
    # law of cosines
    #
    Rgal2 = R0**2. + dist**2.
    Rgal2 = Rgal2 - 2.*R0*dist*np.cos(np.deg2rad(glong))
    Rgal = np.sqrt(Rgal2)
    return Rgal

def calc_az(glong, dist, R0=__R0):
    """
    Return the Galactocentric azimuth of an object with a given
    Galacitic longitude and distance. Galactocentric azimuth is
    defined as zero in the direction of the Sun and increasing
    in the direction of the Solar orbit direction.

    Parameters:
      glong :: scalar or array of scalars
        Galactic longitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      R0 :: scalar (optional)
        Galactocentric radius of the Sun.

    Returns: az
      az :: scalar or array of scalars
        Galactocentric azimuth (degs).
    """
    input_scalar = np.isscalar(glong) and np.isscalar(dist)
    glong, dist = np.atleast_1d(glong, dist)
    # ensure longitude range [0,360) degrees
    glong = glong % 360.
    #
    # Compute Rgal
    #
    Rgal = calc_Rgal(glong, dist, R0=R0)
    #
    # law of cosines
    #
    cos_az = (R0**2. + Rgal**2. - dist**2.)/(2.*Rgal*R0)
    #
    # Catch fringe cases
    #
    cos_az[cos_az > 1.] = 1.
    cos_az[cos_az < -1.] = -1.
    az = np.rad2deg(np.arccos(cos_az))
    #
    # Correct azimuth in 3rd and 4th quadrants
    #
    az[glong > 180.] = 360. - az[glong > 180.]
    if input_scalar:
        return az[0]
    return az

def calc_dist(az, Rgal, R0=__R0):
    """
    Return the line-of-sight distance of an object with a given
    Galactocentric azimuth and radius.

    Parameters:
      az :: scalar or array of scalars
        Galactocentric azimuth (deg).

      Rgal :: scalar or array of scalars
        Galactocentric radius (kpc).

      R0 :: scalar (optional)
        Galactocentric radius of the Sun.

    Returns: dist
      dist :: scalar or array of scalars
        Line-of-sight distance (kpc).
    """
    #
    # law of cosines
    #
    dist2 = R0**2. +Rgal**2.
    dist2 = dist2 - 2.*R0*Rgal*np.cos(np.deg2rad(az))
    dist = np.sqrt(dist2)
    return dist

def calc_glong(az, Rgal, R0=__R0):
    """
    Return the Galactic longitude of an object with a given
    Galacitocentric azimuth and radius. Galactic longitude is
    defined as zero in the direction of the Galactic Center and
    increasing in the direction of the Solar orbit direction.

    Parameters:
      az :: scalar or array of scalars
        Galactocentric azimuth (deg).

      Rgal :: scalar or array of scalars
        Galactocentric radius (kpc).

      R0 :: scalar (optional)
        Galactocentric radius of the Sun.

    Returns: glong
      glong :: scalar or array of scalars
        Galactic longitude (degs).
    """
    input_scalar = np.isscalar(az) and np.isscalar(Rgal)
    az, Rgal = np.atleast_1d(az, Rgal)
    # ensure azimuth range [0,360) degrees
    az = az % 360.
    #
    # Compute line of sight distance
    #
    dist = calc_dist(az, Rgal, R0=R0)
    #
    # law of cosines
    #
    cos_glong = (R0**2. + dist**2. - Rgal**2.)/(2.*dist*R0)
    #
    # Catch fringe cases
    #
    cos_glong[cos_glong > 1.] = 1.
    cos_glong[cos_glong < -1.] = -1.
    glong = np.rad2deg(np.arccos(cos_glong))
    #
    # Correct longitude in 3rd and 4th quadrants
    #
    glong[az > 180.] = 360. - glong[az > 180.]
    if input_scalar:
        return glong[0]
    return glong

def correct_vlsr(glong, glat, vlsr,
                 Ustd=__Ustd, Vstd=__Vstd, Wstd=__Wstd,
                 Usun=__Usun, Vsun=__Vsun, Wsun=__Wsun):
    """
    Return the "corrected" LSR velocity by updating the IAU-defined
    solar motion components.

    Parameters:
      glong :: scalar or array of scalars
        Galactic longitude (deg).

      glat :: scalar or array of scalars
        Galactic latitude (deg).

      vlsr :: scalar or array of scalars
             Measured LSR velocity (km/s).

      Ustd, Vstd, Wstd :: scalars (optional)
        IAU-defined solar motion parameters (km/s).

      Usun, Vsun, Wsun : scalars (optional)
        Updated solar motion parameters (km/s).

    Returns: corr_vlsr
      corr_vlsr :: scalar or array of scalars
        Corrected LSR velocity
    """
    #
    # Useful values
    #
    cos_glong = np.cos(np.deg2rad(glong))
    sin_glong = np.sin(np.deg2rad(glong))
    cos_glat = np.cos(np.deg2rad(glat))
    sin_glat = np.sin(np.deg2rad(glat))
    #
    # Compute heliocentric velocity by subtracting IAU defined solar
    # motion components
    #
    U_part = Ustd*cos_glong
    V_part = Vstd*sin_glong
    W_part = Wstd*sin_glat
    UV_part = (U_part+V_part)*cos_glat
    v_helio = vlsr - UV_part - W_part
    #
    # Compute corrected VLSR
    #
    U_part = Usun*cos_glong
    V_part = Vsun*sin_glong
    W_part = Wsun*sin_glat
    UV_part = (U_part+V_part)*cos_glat
    corr_vlsr = v_helio + UV_part + W_part
    return corr_vlsr

def calc_anderson2012_uncertainty(glong, vlsr):
    """
    Return the Anderson+2012 kinematic distance uncertainties.

    Parameters:
      glong :: scalar or array of scalars
        Galactic longitude (deg).

      vlsr :: scalar or array of scalars
        Measured LSR velocity (km/s).

    Returns: near_err, far_err, tangent_err
      near_err :: scalar or array of scalars
        Anderson+2012 near distance uncertainty

      far_err :: scalar or array of scalars
        Anderson+2012 far distance uncertainty

      tangent_err :: scalar or array of scalars
        Anderson+2012 tangent distance uncertainty

    Raises:
      ValueError : if glong and vlsr are not 1-D; or
                   if glong and vlsr are arrays and 
                   not the same size
    """
    input_scalar = np.isscalar(glong)
    glong, vlsr = np.atleast_1d(glong, vlsr)
    if np.shape(glong) != np.shape(vlsr):
        raise ValueError("glong and vlsr must have same shape")
    #
    # Read Anderson+2012 uncertainty data
    #
    a12file = os.path.join(os.path.dirname(__file__),'curve_data_wise_small.sav')
    a12data = readsav(a12file,python_dict=True)
    a12data = a12data['curve_data_wise_small'][0]
    a12_near_err = a12data['big_percentages_near']/100.
    a12_far_err = a12data['big_percentages_far']/100.
    a12_glongs = a12data['glong']
    a12_vlsrs = a12data['velbinning']
    #
    # find matching longitudes and velocities
    #
    best_glong = np.array([np.nanargmin(np.abs(gl-a12_glongs))
                           for gl in glong])
    best_vlsr = np.array([np.nanargmin(np.abs(vl-a12_vlsrs))
                          for vl in vlsr])
    #
    # Get distance uncertainties
    #
    near_err = a12_near_err[best_glong, best_vlsr]
    far_err = a12_far_err[best_glong, best_vlsr]
    tangent_err = np.nanmax(np.vstack((near_err, far_err)),axis=0)
    if input_scalar:
        return (near_err[0], far_err[0], tangent_err[0])
    return (near_err, far_err, tangent_err)

def calc_hpd(samples, kdetype, alpha=0.683, pdf_bins=1000):
    """
    Fit a kernel density estimator (KDE) to the posterior given
    by a collection of samples. Return the mode (posterior peak)
    and the highest posterior density (HPD) determined by the minimum
    width Bayesian credible interval (BCI) containing a fraction of
    the posterior samples. The posterior should be well described by a
    single-modal distribution.

    Parameters:
      samples :: 1-D array of scalars
        The samples being fit with a KDE

      kdetype :: string
        Which KDE method to use
          'pyqt' uses pyqt_fit with boundary at 0
          'scipy' uses gaussian_kde with no boundary

      alpha :: scalar (optional)
        The fraction of samples included in the BCI.

      pdf_bins :: integer (optional)
        Number of bins used in calculating the PDF

    Returns: kde, mode, lower, upper
      kde :: scipy.gaussian_kde or pyqt_fit.1DKDE object
        The KDE calculated for this kinematic distance

      mode :: scalar
        The mode of the posterior

      lower :: scalar
        The lower bound of the BCI

      upper :: scalar
        The upper bound of the BCI
    """
    # check inputs
    if (alpha <= 0.) or (alpha >= 1.):
        raise ValueError("alpha should be between 0 and 1.")
    #
    # Fit KDE
    #
    nans = np.isnan(samples)
    if np.sum(~nans) < 2:
        # skip if fewer than two non-nans
        return (None, np.nan, np.nan, np.nan)
    try:
        if kdetype == 'scipy':
            kde = gaussian_kde(samples[~nans])
        elif kdetype == 'pyqt':
            kde = pyqt_kde.KDE1D(
                samples[~nans], lower=0,
                method=kde_methods.linear_combination)
        else:
            raise ValueError("Invalid KDE method: {0}".format(kdetype))
    except np.linalg.LinAlgError:
        # catch singular matricies (i.e. all values are the same)
        return (None, np.nan, np.nan, np.nan)
    #
    # Compute PDF
    #
    xdata = np.linspace(
        np.nanmin(samples), np.nanmax(samples), pdf_bins)
    pdf = kde(xdata)
    #
    # Get the location of the mode
    #
    mode = xdata[np.argmax(pdf)]
    if np.isnan(mode):
        return (None, np.nan, np.nan, np.nan)
    #
    # Reverse sort the PDF and xdata and find the BCI
    #
    sort_pdf = sorted(
        zip(xdata, pdf/np.sum(pdf)), key=lambda x: x[1], reverse=True)
    cum_prob = 0.
    bci_xdata = np.empty(len(xdata), dtype=float)*np.nan
    for i, dat in enumerate(sort_pdf):
        cum_prob += dat[1]
        bci_xdata[i] = dat[0]
        if cum_prob >= alpha:
            break
    lower = np.nanmin(bci_xdata)
    upper = np.nanmax(bci_xdata)
    return kde, mode, lower, upper
