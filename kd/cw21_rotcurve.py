#!/usr/bin/env python
"""
cw21_rotcurve.py

Utilities involving the Universal Rotation Curve (Persic 1996) from
Cheng & Wenger (2021), henceforth CW21, with re-done analysis by CW21.
Including HMSFR peculiar motion.

Copyright(C) 2017-2021 by
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

2021-03-03 Trey V. Wenger new in V2.0
"""

import os
import dill
import numpy as np
from kriging import kriging  # Requires https://github.com/tvwenger/kriging
from kd import kd_utils

# Need the following lines because VScode...
import sys
from pathlib import Path
_SCRIPT_DIR = str(Path.cwd() / Path(__file__).parent.parent)
# Add to $PATH
sys.path.append(_SCRIPT_DIR)

#
# CW21 A5 rotation model parameters
#
__R0 = 8.181
__Usun = 10.407
__Vsun = 10.213
__Wsun = 8.078
__Upec = 4.430
__Vpec = -4.823
__a2 = 0.971
__a3 = 1.625
__Zsun = 5.583
__roll = 0.010

#
# IAU defined LSR
#
__Ustd = 10.27
__Vstd = 15.32
__Wstd = 7.74


def nominal_params():
    """
    Return a dictionary containing the nominal rotation curve
    parameters.

    Parameters: Nothing

    Returns: params
      params :: dictionary
        params['a1'], etc. : scalar
          The nominal rotation curve parameter
    """
    params = {
        "R0": __R0,
        "Zsun": __Zsun,
        "Usun": __Usun,
        "Vsun": __Vsun,
        "Wsun": __Wsun,
        "Upec": __Upec,
        "Vpec": __Vpec,
        "roll": __roll,
        "a2": __a2,
        "a3": __a3,
    }
    return params


def resample_params(size=None):
    """
    Resample the rotation curve parameters within their
    uncertainties using the Wenger+2020 kernel density estimator
    to include parameter covariances.

    Parameters:
      size :: integer
        The number of random samples to generate. If None, generate
        only one sample and return a scalar.

    Returns: params
      params :: dictionary
        params['a1'], etc. : scalar or array of scalars
                             The re-sampled parameters
    """
    # kdefile contains full KDE + KDEs of each component (e.g. "R0") + kriging function
    kdefile = os.path.join(os.path.dirname(__file__), "cw21_kde_krige.pkl")
    with open(kdefile, "rb") as f:
        kde = dill.load(f)["full"]
    if size is None:
        samples = kde.resample(1)
        params = {
            "R0": samples[0][0],
            "Zsun": samples[1][0],
            "Usun": samples[2][0],
            "Vsun": samples[3][0],
            "Wsun": samples[4][0],
            "Upec": samples[5][0],
            "Vpec": samples[6][0],
            "roll": samples[7][0],
            "a2": samples[8][0],
            "a3": samples[9][0],
        }
    else:
        samples = kde.resample(size)
        params = {
            "R0": samples[0],
            "Zsun": samples[1],
            "Usun": samples[2],
            "Vsun": samples[3],
            "Wsun": samples[4],
            "Upec": samples[5],
            "Vpec": samples[6],
            "roll": samples[7],
            "a2": samples[8],
            "a3": samples[9],
        }
    return params


def krige_Upec(x, y):
    """
    Estimates peculiar radial velocity at a position
    (positive toward GC) using kriging function.
    Requires tvw's kriging program https://github.com/tvwenger/kriging

    Inputs:
      x, y :: scalars or numpy array of scalars (kpc)
        Galactocentric Cartesian positions

    Returns: Upec, Upec_var
      Upec :: scalar or numpy array of scalars (km/s)
        Peculiar radial velocity toward galactic center

      Upec_var :: scalar or numpy array of scalars (km^2/s^2)
        Variance of Upec
    """
    # krigefile contains full KDE + KDEs of each component (e.g. "R0") + kriging function
    krigefile = os.path.join(os.path.dirname(__file__), "cw21_kde_krige.pkl")
    with open(krigefile, "rb") as f:
        krige = dill.load(f)["krige"]

    Upec, Upec_var = krige(x, y)[0:2]

    return Upec, Upec_var


def krige_Vpec(x, y):
    """
    Estimates peculiar tangential velocity at a position (positive in
    direction of galactic rotation) using kriging function.
    Requires tvw's kriging program https://github.com/tvwenger/kriging

    Inputs:
      x, y :: scalars or numpy array of scalars (kpc)
        Galactocentric Cartesian positions

    Returns: Vpec, Vpec_var
      Vpec :: scalar or numpy array of scalars (km/s)
        Peculiar tangential velocity in direction of galactic rotation

      Vpec_var :: scalar or numpy array of scalars (km^2/s^2)
        Variance of Vpec
    """
    # krigefile contains full KDE + KDEs of each component (e.g. "R0") + kriging function
    krigefile = os.path.join(os.path.dirname(__file__), "cw21_kde_krige.pkl")
    with open(krigefile, "rb") as f:
        krige = dill.load(f)["krige"]

    Vpec, Vpec_var = krige(x, y)[2:4]

    return Vpec, Vpec_var


def krige_UpecVpec(x, y):
    """
    Estimates peculiar radial velocity (positive towarrd GC) and
    tangential velocity at a position (positive in direction of
    galactic rotation) using kriging function.
    Requires tvw's kriging program https://github.com/tvwenger/kriging

    Inputs:
      x, y :: scalars or numpy array of scalars (kpc)
        Galactocentric Cartesian positions

    Returns: Upec, Upec_var, Vpec, Vpec_var
      Upec :: scalar or numpy array of scalars (km/s)
        Peculiar radial velocity toward galactic center

      Upec_var :: scalar or numpy array of scalars (km^2/s^2)
        Variance of Upec

      Vpec :: scalar or numpy array of scalars (km/s)
        Peculiar tangential velocity in direction of galactic rotation

      Vpec_var :: scalar or numpy array of scalars (km^2/s^2)
        Variance of Vpec
    """
    # krigefile contains full KDE + KDEs of each component (e.g. "R0") + kriging function
    krigefile = os.path.join(os.path.dirname(__file__), "cw21_kde_krige.pkl")
    with open(krigefile, "rb") as f:
        krige = dill.load(f)["krige"]

    Upec, Upec_var, Vpec, Vpec_var = krige(x, y)

    return Upec, Upec_var, Vpec, Vpec_var


def calc_theta(R, a2=__a2, a3=__a3, R0=__R0):
    """
    Return circular orbit speed at a given Galactocentric radius.

    Parameters:
      R :: scalar or array of scalars
        Galactocentric radius (kpc)

      a2, a3 :: scalars (optional)
        CW21 rotation curve parameters

      R0 :: scalar (optional)
        Solar Galactocentric radius (kpc)

    Returns: theta
      theta :: scalar or array of scalars
        circular orbit speed at R (km/s)
    """
    input_scalar = np.isscalar(R)
    R = np.atleast_1d(R)
    #
    # Re-production of Reid+2019 FORTRAN code
    #
    rho = R / (a2 * R0)
    lam = (a3 / 1.5) ** 5.0
    loglam = np.log10(lam)
    term1 = 200.0 * lam ** 0.41
    term2 = np.sqrt(
        0.8 + 0.49 * loglam + 0.75 * np.exp(-0.4 * lam) / (0.47 + 2.25 * lam ** 0.4)
    )
    term3 = (0.72 + 0.44 * loglam) * 1.97 * rho ** 1.22 / (rho ** 2.0 + 0.61) ** 1.43
    term4 = 1.6 * np.exp(-0.4 * lam) * rho ** 2.0 / (rho ** 2.0 + 2.25 * lam ** 0.4)
    #
    # Catch non-physical case where term3 + term4 < 0
    #
    term = term3 + term4
    term[term < 0.0] = np.nan
    #
    # Circular velocity
    #
    theta = term1 / term2 * np.sqrt(term)
    if input_scalar:
        return theta[0]
    return theta


def calc_vlsr(
    glong,
    glat,
    dist,
    R0=__R0,
    Usun=__Usun,
    Vsun=__Vsun,
    Wsun=__Wsun,
    Upec=__Upec,
    Vpec=__Vpec,
    a2=__a2,
    a3=__a3,
    Zsun=__Zsun,
    roll=__roll,
    peculiar=False,
):
    """
    Return the IAU-LSR velocity at a given Galactic longitude and
    line-of-sight distance.

    Parameters:
      glong, glat :: scalars or arrays of scalars
        Galactic longitude and latitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      R0 :: scalar (optional)
        Solar Galactocentric radius (kpc)

      Usun, Vsun, Wsun, Upec, Vpec, a2, a3 :: scalars (optional)
        CW21 rotation curve parameters

      Zsun :: scalar (optional)
        Height of sun above Galactic midplane (pc)

      roll :: scalar (optional)
        Roll of Galactic midplane relative to b=0 (deg)

      peculiar :: boolean (optional)
        If True, include HMSFR peculiar motion component

    Returns: vlsr
      vlsr :: scalar or array of scalars
        LSR velocity (km/s).
    """
    input_scalar = np.isscalar(glong) and np.isscalar(glat) and np.isscalar(dist)
    glong, glat, dist = np.atleast_1d(glong, glat, dist)
    cos_glong = np.cos(np.deg2rad(glong))
    sin_glong = np.sin(np.deg2rad(glong))
    cos_glat = np.cos(np.deg2rad(glat))
    sin_glat = np.sin(np.deg2rad(glat))
    #
    # Convert distance to Galactocentric, catch small Rgal
    #
    Rgal = kd_utils.calc_Rgal(glong, glat, dist, R0=R0)
    Rgal[Rgal < 1.0e-6] = 1.0e-6
    az = kd_utils.calc_az(glong, glat, dist, R0=R0)
    cos_az = np.cos(np.deg2rad(az))
    sin_az = np.sin(np.deg2rad(az))
    #
    # Rotation curve circular velocity
    #
    theta = calc_theta(Rgal, a2=a2, a3=a3, R0=R0)
    theta0 = calc_theta(R0, a2=a2, a3=a3, R0=R0)
    #
    # Add HMSFR peculiar motion
    #
    if peculiar:
        vR = -Upec
        vAz = theta + Vpec
        vZ = 0.0
    else:
        vR = 0.0
        vAz = theta
        vZ = 0.0
    vXg = -vR * cos_az + vAz * sin_az
    vYg = vR * sin_az + vAz * cos_az
    vZg = vZ
    #
    # Convert to barycentric
    #
    X = dist * cos_glat * cos_glong
    Y = dist * cos_glat * sin_glong
    Z = dist * sin_glat
    # useful constants
    sin_tilt = Zsun / 1000.0 / R0
    cos_tilt = np.cos(np.arcsin(sin_tilt))
    sin_roll = np.sin(np.deg2rad(roll))
    cos_roll = np.cos(np.deg2rad(roll))
    # solar peculiar motion
    vXg = vXg - Usun
    vYg = vYg - theta0 - Vsun
    vZg = vZg - Wsun
    # correct tilt and roll of Galactic midplane
    vXg1 = vXg * cos_tilt - vZg * sin_tilt
    vYg1 = vYg
    vZg1 = vXg * sin_tilt + vZg * cos_tilt
    vXh = vXg1
    vYh = vYg1 * cos_roll + vZg1 * sin_roll
    vZh = -vYg1 * sin_roll + vZg1 * cos_roll
    vbary = (X * vXh + Y * vYh + Z * vZh) / dist
    #
    # Convert to IAU-LSR
    #
    vlsr = (
        vbary + (__Ustd * cos_glong + __Vstd * sin_glong) * cos_glat + __Wsun * sin_glat
    )
    if input_scalar:
        return vlsr[0]
    return vlsr

# print(krige_Upec(np.array([5, 10]), np.array([8, 5])))
# print(krige_Upec(np.array([5]), np.array([8])))
# print(krige_Upec(5, 8))
# print(krige_Vpec(5, 8))
# print(krige_UpecVpec(5, 8))