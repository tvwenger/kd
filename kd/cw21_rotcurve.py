#!/usr/bin/env python
"""
cw21_rotcurve_old.py

OLD FILE THAT HAS MEMORY LEAK ISSUE.

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

import numpy as np
from kd import kd_utils

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

      R0 :: scalar or array of scalars (optional)
        Galactocentric radius of the Sun

      Returns: x, y
        x, y :: scalar or array of scalars
          Galactocentric Cartesian x- and y-coordinates
    """
    glong, glat, dist = np.atleast_1d(glong, glat, dist)
    if np.shape(glong) != np.shape(dist):
        glong = np.array([glong, ] * len(dist))
    Rgal = kd_utils.calc_Rgal(glong, glat, dist, R0=R0)
    Rgal[Rgal < 1.0e-6] = 1.0e-6  # Catch small Rgal
    az = kd_utils.calc_az(glong, glat, dist, R0=R0)
    cos_az = np.cos(np.deg2rad(az))
    sin_az = np.sin(np.deg2rad(az))

    x = Rgal * -cos_az
    y = Rgal * sin_az

    return x, y, Rgal, cos_az, sin_az


def krige_UpecVpec(x, y, krige, Upec_var_threshold, Vpec_var_threshold,
                   Upec_avg=__Upec, Vpec_avg=__Vpec,
                   var_Upec_avg=__Upec_var, var_Vpec_avg=__Vpec_var):
    """
    Estimates the peculiar radial velocity (positive toward GC) and
    tangential velocity (positive in direction of galactic rotation)
    at a given (x, y) position using kriging.
    Requires tvw's kriging program https://github.com/tvwenger/kriging

    Parameters:
      x, y :: scalars or arrays of scalars
        Galactocentric Cartesian positions (kpc)

      krige :: function
        Kriging function that takes in parameters (x, y) to evaluate peculiar
        motions at the given coordinate(s) (i.e. tvw's kriging program)

      Upec_var_threshold, Vpec_var_threshold :: scalars
        Maximum variance of Upec and Vpec allowed by kriging.
        If the Upec and Vpec from kriging have variances larger than this,
        this function will use the average Upec and Vpec instead

      Upec_avg :: scalar (optional)
        Average peculiar motion of HMSFRs toward galactic center (km/s)

      Vpec_avg :: scalar (optional)
        Average peculiar motion of HMSFRs in
        direction of galactic rotation (km/s)

      var_Upec_avg, var_Vpec_avg :: scalar (optional)
        Variance of Upec_avg and Vpec_avg (km^2/s^2)

    Returns: Upec, Upec_var, Vpec, Vpec_var
      Upec :: array of scalars
        Peculiar radial velocity of source toward galactic center (km/s)

      Upec_var :: array of scalars
        Variance of Upec (km^2/s^2)

      Vpec :: array of scalars
        Peculiar tangential velocity of source in
        direction of galactic rotation (km/s)

      Vpec_var :: array of scalars
        Variance of Vpec (km^2/s^2)
    """
    # Switch to convention used in kriging map
    # (Rotate 90 deg CW, Sun is on +y-axis)
    x, y = y, -x

    # Calculate expected Upec and Vpec at source location(s)
    Upec, Upec_var, Vpec, Vpec_var = krige(x, y)
    Upec = Upec.reshape(np.shape(x))
    Upec_var = Upec_var.reshape(np.shape(x))
    Vpec = Vpec.reshape(np.shape(x))
    Vpec_var = Vpec_var.reshape(np.shape(x))

    # Use average value if component is outside well-constrained area
    Upec_mask = Upec_var > Upec_var_threshold
    Vpec_mask = Vpec_var > Vpec_var_threshold
    Upec[Upec_mask] = Upec_avg
    Vpec[Vpec_mask] = Vpec_avg
    Upec_var[Upec_mask] = var_Upec_avg
    Vpec_var[Vpec_mask] = var_Vpec_avg

    # Free up resources
    krige = Upec_var_threshold = Vpec_var_threshold = None

    return Upec, Upec_var, Vpec, Vpec_var


def nominal_params(glong=None, glat=None, dist=None,
                   krige=None, Upec_var_threshold=None, Vpec_var_threshold=None,
                   use_kriging=False):
    """
    Return a dictionary containing the nominal rotation curve
    parameters.

    Parameters:
      glong, glat :: scalars or arrays of scalars (optional, required for kriging)
        Galactic longitude and latitude (deg)

      dist :: scalar or array of scalars (optional, required for kriging)
        Line-of-sight distance (kpc)

      krige :: function (optional, required for kriging)
        Kriging function that takes in parameters (x, y) to evaluate peculiar
        motions at the given coordinate(s) (i.e. tvw's kriging program)

      Upec_var_threshold, Vpec_var_threshold :: scalars (optional,
                                                         required for kriging)
        Maximum variance of Upec and Vpec allowed by kriging.
        If the Upec and Vpec from kriging have variances larger than this,
        the program will use the average Upec and Vpec instead

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
    if use_kriging and glong is not None and glat is not None and dist is not None:
        # Calculate galactocentric positions
        x, y, Rgal, cos_az, sin_az = calc_gcen_coords(glong, glat, dist, R0=__R0)
        # Calculate individual Upec and Vpec at source location(s)
        Upec, Upec_var, Vpec, Vpec_var = krige_UpecVpec(
            x, y, krige, Upec_var_threshold, Vpec_var_threshold,
            Upec_avg=__Upec, Vpec_avg=__Vpec,
            var_Upec_avg=__Upec_var, var_Vpec_avg=__Vpec_var)
    else:
        # Use average Upec and Vpec
        Upec = __Upec
        Vpec = __Vpec
        Upec_var = __Upec_var
        Vpec_var = __Vpec_var
        Rgal = cos_az = sin_az = None

    params = {
        "R0": __R0,
        "Zsun": __Zsun,
        "Usun": __Usun,
        "Vsun": __Vsun,
        "Wsun": __Wsun,
        "Upec": Upec,
        "Upec_var": Upec_var,
        "Vpec": Vpec,
        "Vpec_var": Vpec_var,
        "roll": __roll,
        "a2": __a2,
        "a3": __a3,
    }
    return params, Rgal, cos_az, sin_az


def resample_params(kde, size=None, nom_params=None, use_kriging=False):
    """
    Resample the rotation curve parameters within their
    uncertainties using the CW21 kernel density estimator
    to include parameter covariances.

    Parameters:
      kde :: kernel density estimator class (e.g. scipy.stats.gaussian_kde)
        Kernel density estimator containing all the rotation model parameters

      size :: integer
        The number of random samples to generate (per source, if use_kriging).
        If None, generate only one sample and return a scalar

      nom_params :: dictionary (optional, required for kriging)
        Dictionary containing the kriging values of Upec, Vpec,
        and their associated variances (Upec_var, Vpec_var)

      use_kriging :: boolean (optional)
        If True, estimate individual Upec & Vpec from kriging program
        If False, use average Upec & Vpec

    Returns: params
      params :: dictionary
        params['R0'], etc. : scalar or array of scalars
                             The re-sampled parameters
    """
    if size is None:
        samples = kde.resample(1)
        params = {
            "R0": samples[0][0],
            "Zsun": samples[1][0],
            "Usun": samples[2][0],
            "Vsun": samples[3][0],
            "Wsun": samples[4][0],
            "Upec": samples[5][0],
            "Upec_var": __Upec_var,
            "Vpec": samples[6][0],
            "Vpec_var": __Vpec_var,
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
            "Upec_var": __Upec_var,
            "Vpec": samples[6],
            "Vpec_var": __Vpec_var,
            "roll": samples[7],
            "a2": samples[8],
            "a3": samples[9],
        }
    kde = None  # free up resources

    if use_kriging and nom_params is not None:
        Upec = nom_params["Upec"]
        Vpec = nom_params["Vpec"]
        Upec_var = nom_params["Upec_var"]
        Vpec_var = nom_params["Vpec_var"]
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
            "Upec_var": Upec_var,
            "Vpec": Vpec,
            "Vpec_var": Vpec_var,
            "roll": params_orig["roll"],
            "a2": params_orig["a2"],
            "a3": params_orig["a3"],
        }
    return params


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
              Zsun=__Zsun, roll=__roll, peculiar=False):
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

      Upec_var, Vpec_var :: scalars (optional)
        Variance in Upec/Vpec values
        Not used but must be stored in the params dictionary,
        hence they are parameters here

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
    # print("glong, glat, dist in calc_vlsr", np.shape(glong), np.shape(glat), np.shape(dist))
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
        Rgal[Rgal < 1.0e-6] = 1.0e-6  # Catch small Rgal
        az = kd_utils.calc_az(glong, glat, dist, R0=R0)
        cos_az = np.cos(np.deg2rad(az))
        sin_az = np.sin(np.deg2rad(az))
    # print("Rgal, cos_az, sin_az in calc_vlsr", np.shape(Rgal), np.shape(cos_az), np.shape(sin_az))
    #
    # Rotation curve circular velocity
    #
    theta = calc_theta(Rgal, a2=a2, a3=a3, R0=R0)
    theta0 = calc_theta(R0, a2=a2, a3=a3, R0=R0)
    # print("Theta, theta0 in calc_vlsr", np.shape(theta), np.shape(theta0))
    # print("Upec, Vpec in calc_vlsr", np.shape(Upec), np.shape(Vpec))
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
    # print("1st vXg, vYg, vZg in calc_vlsr", np.shape(vXg), np.shape(vYg), np.shape(vZg))
    #
    # Convert to barycentric
    #
    X = dist * cos_glat * cos_glong
    Y = dist * cos_glat * sin_glong
    Z = dist * sin_glat
    # print("X, Y, Z in calc_vlsr", np.shape(X), np.shape(Y), np.shape(Z))
    # useful constants
    sin_tilt = Zsun / 1000.0 / R0
    cos_tilt = np.cos(np.arcsin(sin_tilt))
    sin_roll = np.sin(np.deg2rad(roll))
    cos_roll = np.cos(np.deg2rad(roll))
    # solar peculiar motion
    vXg = vXg - Usun
    vYg = vYg - theta0 - Vsun
    vZg = vZg - Wsun
    # print("2nd vXg, vYg, vZg in calc_vlsr", np.shape(vXg), np.shape(vYg), np.shape(vZg))
    # correct tilt and roll of Galactic midplane
    vXg1 = vXg * cos_tilt - vZg * sin_tilt
    vYg1 = vYg
    vZg1 = vXg * sin_tilt + vZg * cos_tilt
    vXh = vXg1
    vYh = vYg1 * cos_roll + vZg1 * sin_roll
    vZh = -vYg1 * sin_roll + vZg1 * cos_roll
    vbary = (X * vXh + Y * vYh + Z * vZh) / dist
    # print("vbary in calc_vlsr", np.shape(vbary))
    #
    # Convert to IAU-LSR
    #
    vlsr = (
        vbary + (__Ustd * cos_glong + __Vstd * sin_glong) * cos_glat + __Wsun * sin_glat
    )
    if input_scalar:
        return vlsr[0]
    # print("final vlsr shape", np.shape(vlsr))
    return vlsr
