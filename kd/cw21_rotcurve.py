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
# CW21 A6 rotation model parameters
#
__R0 = 8.1746
__Usun = 10.879
__Vsun = 10.697
__Wsun = 8.088
__Upec = 4.907
__Upec_var = 16.9  # * UPDATE. variance of __Upec
__Vpec = -4.522
__Vpec_var = 34.3  # * UPDATE. variance of __Vpec
__a2 = 0.977
__a3 = 1.626
__Zsun = 5.399
__roll = -0.011

#
# IAU defined LSR
#
__Ustd = 10.27
__Vstd = 15.32
__Wstd = 7.74

# Open statement outside function does not help reduce memory usage
# and I have to call it inside the function anyway
# # krigefile contains: full KDE + KDEs of each component (e.g. "R0", "Zsun", etc.)
# #                     + kriging function + kriging thresholds
# krigefile = os.path.join(os.path.dirname(__file__), "cw21_kde_krige.pkl")
# with open(krigefile, "rb") as f:
#     file = dill.load(f)
#     krige = file["krige"]
#     Upec_var_threshold = file["Upec_var_threshold"]
#     Vpec_var_threshold = file["Vpec_var_threshold"]


def calc_gcen_coords(glong, glat, dist, R0=__R0):
    """
    Calculate galactocentric Cartesian coordinates from
    galactic longitudes, latitudes, and distances from the Sun.

    Parameters:
      glong, glat :: scalar or array of scalars
        Galactic longitude and latitude (deg)

      dist :: scalar or array of scalars
        line-of-sight distance (kpc)

      R0 :: scalar or array of scalars
        Galactocentric radius of the Sun

      Returns: x, y
        x, y :: scalar or array of scalars
          Galactocentric Cartesian x- and y-coordinates
    """
    # glong, glat, dist = np.copy(glong), np.copy(glat), np.copy(dist)
    # print("glong1:", np.shape(glong), np.shape(glat), np.shape(dist), np.shape(R0))
    glong, glat, dist = np.atleast_1d(glong, glat, dist)
    # print("glong2:", np.shape(glong), np.shape(glat), np.shape(dist), np.shape(R0))
    Rgal = kd_utils.calc_Rgal(glong, glat, dist, R0=R0)
    # print("Rgal:", np.shape(Rgal))
    Rgal[Rgal < 1.0e-6] = 1.0e-6  # Catch small Rgal
    az = kd_utils.calc_az(glong, glat, dist, R0=R0)
    cos_az = np.cos(np.deg2rad(az))
    sin_az = np.sin(np.deg2rad(az))

    x = Rgal * -cos_az
    y = Rgal * sin_az

    return x, y, Rgal, cos_az, sin_az


def krige_UpecVpec(
    x, y,
    Upec_avg=__Upec, Vpec_avg=__Vpec,
    var_Upec_avg=__Upec_var, var_Vpec_avg=__Vpec_var):
    """
    Estimates peculiar radial velocity (positive towarrd GC) and
    tangential velocity at a position (positive in direction of
    galactic rotation) using kriging function.
    Requires tvw's kriging program https://github.com/tvwenger/kriging

    Parameters:
      x, y :: scalars or numpy array of scalars
        Galactocentric Cartesian positions (kpc)

      Upec_avg :: scalar (optional)
        Average peculiar motion of HMSFRs toward galactic center (km/s)

      Vpec_avg :: scalar (optional)
        Average peculiar motion of HMSFRs in
        direction of galactic rotation (km/s)

      var_Upec_avg, var_Vpec_avg :: scalar (optional)
        Variance of Upec_avg and Vpec_avg (km^2/s^2)

    Returns: Upec, Upec_var, Vpec, Vpec_var
      Upec :: scalar or numpy array of scalars
        Peculiar radial velocity of source toward galactic center (km/s)

      Upec_var :: scalar or numpy array of scalars
        Variance of Upec (km^2/s^2)

      Vpec :: scalar or numpy array of scalars
        Peculiar tangential velocity of source in
        direction of galactic rotation (km/s)

      Vpec_var :: scalar or numpy array of scalars
        Variance of Vpec (km^2/s^2)
    """
    # krigefile contains: full KDE + KDEs of each component (e.g. "R0", "Zsun", etc.)
    #                     + kriging function + kriging thresholds
    krigefile = os.path.join(os.path.dirname(__file__), "cw21_kde_krige.pkl")
    with open(krigefile, "rb") as f:
        file = dill.load(f)
        krige = file["krige"]
        Upec_var_threshold = file["Upec_var_threshold"]
        Vpec_var_threshold = file["Vpec_var_threshold"]
        file = None

    # Switch to convention used in kriging map
    # (Rotate 90 deg CW, Sun is on +y-axis)
    x, y = y, -x

    # Calculate expected Upec and Vpec at source location(s)
    Upec, Upec_var, Vpec, Vpec_var = krige(x, y)
    # print("kriging results:", Upec, Upec_var, Vpec, Vpec_var)
    # print("krige_UpecVpec, Upec before reshape:", np.shape(Upec))
    Upec = Upec.reshape(np.shape(x))
    Upec_var = Upec_var.reshape(np.shape(x))
    Vpec = Vpec.reshape(np.shape(x))
    Vpec_var = Vpec_var.reshape(np.shape(x))
    # print("x, y", np.shape(x), np.shape(y))
    # print("Upec_var", np.shape(Upec_var))
    # print("Upec_avg", np.shape(Upec_avg))
    # print("Upec", np.shape(Upec), type(Upec), Upec)
    # print("var_Upec_avg", np.shape(var_Upec_avg))
    # print("Upec_var_threshold:", type(Upec_var_threshold), Upec_var_threshold)
    # print(Upec[Upec_var > 121])

    # Use average value if component is outside well-constrained area
    if np.isscalar(Upec):
        Upec = Upec_avg if Upec_var > Upec_var_threshold else Upec
        Vpec = Vpec_avg if Vpec_var > Vpec_var_threshold else Vpec
        Upec_var = var_Upec_avg if Upec_var > Upec_var_threshold else Upec_var
        Vpec_var = var_Vpec_avg if Vpec_var > Vpec_var_threshold else Vpec_var
    else:
        # print("Upec_avg shape:", np.shape(Upec_avg), np.shape(var_Upec_avg))
        # if np.shape(Upec_avg) != np.shape(Upec):
        #     # print("full_like Upec_avg and Vpec_avg")
        #     Upec_avg = np.full_like(Upec, Upec_avg, float)
        #     Vpec_avg = np.full_like(Vpec, Vpec_avg, float)
        # if np.shape(var_Upec_avg) != np.shape(Upec):
        #     # print("full_like var_Upec_avg and var_Vpec_avg")
        #     var_Upec_avg = np.full_like(Upec, var_Upec_avg, float)
        #     var_Vpec_avg = np.full_like(Vpec, var_Vpec_avg, float)
        # print(Upec[:, 0:10])
        # print(Upec_avg[:, 0:10])
        # print("krige_UpecVpec, Upec, Upec_avg after reshape", np.shape(Upec), np.shape(Upec_avg))
        Upec_mask = Upec_var > Upec_var_threshold
        Vpec_mask = Vpec_var > Vpec_var_threshold
        # Upec[Upec_mask] = Upec_avg[Upec_mask]
        # Vpec[Vpec_mask] = Vpec_avg[Vpec_mask]
        # Upec_var[Upec_mask] = var_Upec_avg[Upec_mask]
        # Vpec_var[Vpec_mask] = var_Vpec_avg[Vpec_mask]
        Upec[Upec_mask] = Upec_avg
        Vpec[Vpec_mask] = Vpec_avg
        Upec_var[Upec_mask] = var_Upec_avg
        Vpec_var[Vpec_mask] = var_Vpec_avg

    # print("Final kriging results:", Upec, Upec_var, Vpec, Vpec_var)
    krige = Upec_var_threshold = Vpec_var_threshold = None

    return Upec, Upec_var, Vpec, Vpec_var


def nominal_params(glong=None, glat=None, dist=None, use_kriging=False):
    """
    Return a dictionary containing the nominal rotation curve
    parameters.

    Parameters:
      glong, glat :: scalars or arrays of scalars
        Galactic longitude and latitude (deg)

      dist :: scalar or array of scalars
        Line-of-sight distance (kpc)

      use_kriging :: boolean (optional)
        If True, estimate individual Upec & Vpec from kriging program
        If False, use average Upec & Vpec

    Returns: params, Rgal, cos_az, sin_az
      params :: dictionary
        params['R0'], etc. : scalar
          The nominal rotation curve parameter

      Rgal :: scalar or array of scalars
        Galactocentric cylindrical radius (kpc)

      cos_az, sin_az :: scalar or array of scalars
        Cosine and sine of Galactocentric azimuth (rad)
    """
    print("In nominal params:", np.shape(glong), np.shape(glat), np.shape(dist))
    if use_kriging and glong is not None and glat is not None and dist is not None:
        print("Using kriging in nominal params")
        # Calculate galactocentric positions
        x, y, Rgal, cos_az, sin_az = calc_gcen_coords(glong, glat, dist, R0=__R0)
        # Calculate individual Upec and Vpec at source location(s)
        Upec, Upec_var, Vpec, Vpec_var = krige_UpecVpec(
            x, y, Upec_avg=__Upec, Vpec_avg=__Vpec,
            var_Upec_avg=__Upec_var, var_Vpec_avg=__Vpec_var)
    else:
        # Use average Upec and Vpec
        Upec = __Upec
        Vpec = __Vpec
        # Upec_var = __Upec_var
        # Vpec_var = __Vpec_var
        Rgal = cos_az = sin_az = None
    print("Exiting nominal params")

    params = {
        "R0": __R0,
        "Zsun": __Zsun,
        "Usun": __Usun,
        "Vsun": __Vsun,
        "Wsun": __Wsun,
        "Upec": Upec,
        # "Upec_var": Upec_var,
        "Vpec": Vpec,
        # "Vpec_var": Vpec_var,
        "roll": __roll,
        "a2": __a2,
        "a3": __a3,
    }
    # print(np.shape(params["Upec"]))
    return params, Rgal, cos_az, sin_az


# def printit(dict, **args):
#     for arg in args:
#         print(arg, dict[arg])

def resample_params(size=None, glong=None, glat=None, dist=None, use_kriging=False):
    """
    Resample the rotation curve parameters within their
    uncertainties using the CW21 kernel density estimator
    to include parameter covariances.

    Parameters:
      size :: integer
        The number of random samples to generate (per source, if use_kriging).
        If None, generate only one sample and return a scalar

      glong, glat :: scalars or arrays of scalars
        Galactic longitude and latitude (deg)

      dist :: scalar or array of scalars
        Line-of-sight distance (kpc)

      use_kriging :: boolean (optional)
        If True, estimate individual Upec & Vpec from kriging program
        If False, use average Upec & Vpec

    Returns: params, Rgal, cos_az, sin_az
      params :: dictionary
        params['R0'], etc. : scalar or array of scalars
                             The re-sampled parameters
        If use_kriging :
            Each shape of 'Upec', 'Vpec', 'Upec_var', and 'Vpec_var' is
            = (# sources, size) if # sources > 1
              i.e., columns are the same source,
              rows are all n=size samples of one source
            = (size) if # sources == 1

      Rgal :: scalar or array of scalars
        Galactocentric cylindrical radius (kpc)

      cos_az, sin_az :: scalar or array of scalars
        Cosine and sine of Galactocentric azimuth (rad)
    """
    # kdefile contains: full KDE + KDEs of each component (e.g. "R0")
    #                   + kriging function + kriging thresholds
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
            # "Upec_var": __Upec_var,
            "Vpec": samples[6][0],
            # "Vpec_var": __Vpec_var,
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
            # "Upec_var": np.array([__Upec_var,] * size),
            "Vpec": samples[6],
            # "Vpec_var": np.array([__Vpec_var,] * size),
            "roll": samples[7],
            "a2": samples[8],
            "a3": samples[9],
        }
    kde = None

    if use_kriging and glong is not None and glat is not None and dist is not None:
        # print("In resample params:", np.shape(glong), np.shape(glat), np.shape(dist))
        # print(np.shape(params["Upec"]))

        # if np.size(glong) == 1 and np.size(glat) == 1 and np.size(dist) == 1:
        # if np.size(glong) == 1 and np.size(glat) == 1:
        #     Upec_avg = params["Upec"]
        #     Vpec_avg = params["Vpec"]
        #     var_Upec_avg = params["Upec_var"]
        #     var_Vpec_avg = params["Vpec_var"]
        # else:
        #     Upec_avg = np.array([params["Upec"],] * len(glong))
        #     Vpec_avg = np.array([params["Vpec"],] * len(glong))
        #     var_Upec_avg = np.array([params["Upec_var"],] * len(glong))
        #     var_Vpec_avg = np.array([params["Vpec_var"],] * len(glong))

        Upec_avg = params["Upec"]
        Vpec_avg = params["Vpec"]
        # var_Upec_avg = params["Upec_var"]
        # var_Vpec_avg = params["Vpec_var"]
        # if size is not None:
        #     # Shape = (# sources, size) if # sources > 1; else: shape = (size)
        #     # i.e., columns are the same source, rows are all the samples of one source
        #     glong = np.array([glong,] * size).T
        #     glat = np.array([glat,] * size).T
        #     # dist = np.array([dist,] * size).T

        # if np.shape(glong) != np.shape(Upec_avg):
        #     raise ValueError("Please ensure glong, glat, and dist are 1D arrays" + \
        #                      f"\nglong shape: {np.shape(glong)} " + \
        #                      f"vs. Upec_avg shape: {np.shape(Upec_avg)}")

        # Calculate galactocentric positions
        x, y, Rgal, cos_az, sin_az = calc_gcen_coords(
            glong, glat, dist, R0=params["R0"])
        # Calculate individual Upec and Vpec at source location(s)
        # Upec, Upec_var, Vpec, Vpec_var = krige_UpecVpec(
        #     x, y, Upec_avg=Upec_avg, Vpec_avg=Vpec_avg,
        #     var_Upec_avg=var_Upec_avg, var_Vpec_avg=var_Vpec_avg)
        Upec, Upec_var, Vpec, Vpec_var = krige_UpecVpec(
            x, y, Upec_avg=Upec_avg, Vpec_avg=Vpec_avg,
            var_Upec_avg=__Upec_var, var_Vpec_avg=__Vpec_var)
        # Sample Upec and Vpec
        Upec = np.random.normal(loc=Upec, scale=np.sqrt(Upec_var))
        Vpec = np.random.normal(loc=Vpec, scale=np.sqrt(Vpec_var))
        # Save in dictionary
        params_orig = params
        params = {
            "R0": params_orig["R0"],
            "Zsun": params_orig["Zsun"],
            "Usun": params_orig["Usun"],
            "Vsun": params_orig["Vsun"],
            "Wsun": params_orig["Wsun"],
            "Upec": Upec,
            # "Upec_var": Upec_var,
            "Vpec": Vpec,
            # "Vpec_var": Vpec_var,
            "roll": params_orig["roll"],
            "a2": params_orig["a2"],
            "a3": params_orig["a3"],
        }
        # print("Upec_avg", Upec_avg)
        # print("Upec", params["Upec"])
        # print(params["a3"])
        # print(params["Vpec"])
        # print(np.sqrt(params["Vpec_var"]))
        # printit(params, **params)
    else:
        Rgal = cos_az = sin_az = None

    # print("Resample_params final Upec shape:", np.shape(params["Upec"]))
    return params, Rgal, cos_az, sin_az


# * --- TEST STUFF ---
# resample_params(size=None, glong=np.array([19.36]), glat=np.array([-0.03]),
#                 dist=np.array([2.84]), use_kriging=True)
# resample_params(size=4, glong=19.36, glat=-0.03, dist=2.84, use_kriging=True)
# resample_params(
#     size=None,
#     glong=np.array([19.36, 36.12]),
#     glat=np.array([-0.03, 0.55]),
#     dist=np.array([2.84, 4.07]),
#     use_kriging=True,
# )
# resample_params(
#     size=4,
#     glong=np.array([19.36, 12.12]),
#     glat=np.array([-0.03, 111.55]),
#     dist=np.array([2.84, 4.07]),
#     use_kriging=True,
# )
# # The following should fail... (?)
# resample_params(
#     size=None,
#     glong=np.array([[19.36, 36.12],[19.36, 36.12]]),
#     glat=np.array([[-0.03, 0.55], [-0.03, 0.55]]),
#     dist=np.array([[2.84, 4.07], [2.84, 4.07]]),
#     use_kriging=True,
# )
# TODO: check each of the above cases. YES!
# TODO: fix all calls in rotcurve_kd.py. I think it is done...
# * --- TEST STUFF ---
# def nominal_params():
#     """
#     Return a dictionary containing the nominal rotation curve
#     parameters.

#     Parameters: Nothing

#     Returns: params
#       params :: dictionary
#         params['a1'], etc. : scalar
#           The nominal rotation curve parameter
#     """
#     params = {
#         "R0": __R0,
#         "Zsun": __Zsun,
#         "Usun": __Usun,
#         "Vsun": __Vsun,
#         "Wsun": __Wsun,
#         "Upec": __Upec,
#         "Upec_var": __Upec_var,
#         "Vpec": __Vpec,
#         "Vpec_var": __Vpec_var,
#         "roll": __roll,
#         "a2": __a2,
#         "a3": __a3,
#     }
#     return params


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


def calc_vlsr(glong, glat, dist, Rgal=None, cos_az=None, sin_az=None,
              R0=__R0, Usun=__Usun, Vsun=__Vsun,
              Wsun=__Wsun, Upec=__Upec, Upec_var=__Upec_var,
              Vpec=__Vpec, Vpec_var=__Vpec_var, a2=__a2, a3=__a3,
              Zsun=__Zsun, roll=__roll, peculiar=False, use_kriging=False):
    """
    Return the IAU-LSR velocity at a given Galactic longitude and
    line-of-sight distance.

    Parameters:
      glong, glat :: scalars or arrays of scalars
        Galactic longitude and latitude (deg).

      dist :: scalar or array of scalars
        line-of-sight distance (kpc).

      Rgal :: scalar or array of scalars (optional)
        Galactocentric cylindrical radius (kpc).

      cos_az, sin_az :: scalar or array of scalars (optional)
        Cosine and sine of Galactocentric azimuth (rad).

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
    # print("glong shape", np.shape(glong))
    #
    if Rgal is None or cos_az is None or sin_az is None:
        # print("Calculating Rgal in calc_vlsr")
        # Convert distance to Galactocentric, catch small Rgal
        Rgal = kd_utils.calc_Rgal(glong, glat, dist, R0=R0)
        # print("Rgal shape", np.shape(Rgal))
        # print("glong shape after Rgal:", np.shape(glong))
        Rgal[Rgal < 1.0e-6] = 1.0e-6
        az = kd_utils.calc_az(glong, glat, dist, R0=R0)
        # print("az shape", np.shape(az))
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
    if use_kriging and np.shape(vlsr)[1] > 1:
        # vlsr = np.median(vlsr, axis=1)[:, np.newaxis]
        # print("vlsr", np.shape(vlsr))
        # print(vlsr[0:10, 0:10])
        vlsr = vlsr[0]
        vlsr = vlsr[:, np.newaxis]
    # print("vlsr", np.shape(vlsr))
    # print("vlsr[0]", vlsr[0])
    # print("vlsr[1:10]", vlsr[1:10])
    if input_scalar:
        return vlsr[0]
    return vlsr
