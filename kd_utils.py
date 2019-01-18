#!/usr/bin/env python
"""
kd_utils.py

Utility functions for rotcurve_kd.py, pdf_kd.py, and rotation curves.

2017-04-12 Trey V. Wenger
2018-02-10 Trey V. Wenger added correct_vlsr
2019-01-18 Trey V. Wenger removed pool_wait
                          added function to compute Anderson+2012
                          kinematic distance uncertainties
"""

import time
import numpy as np
from scipy.io import readsav

def calc_Rgal(glong, dist, R0=8.34):
    """
    Return the Galactocentric radius of an object with a given
    Galacitic longitude and distance.

    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as dist.
      dist : scalar or 1-D array
             line-of-sight distance (kpc). If it is an array, it
             must have the same size as glong.
      R0 : scalar (optional)
           Galactocentric radius of the Sun.

    Returns: R
      Rgal : scalar or 1-D array
             Galactocentric radius (kpc). If glong and dist are
             scalars, it is a scalar. Otherwise, it has shape
             (dist.size).

    Raises:
      ValueError : if glong or dist are not 1-D; or
                   if glong and dist are arrays and not the same size
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, dist_inp = np.atleast_1d(glong, dist)
    # check shape of inputs
    if glong_inp.ndim != 1 or dist_inp.ndim != 1:
        raise ValueError("glong and dist must be 1-D")
    if glong_inp.size != 1 and glong_inp.size != dist_inp.size:
        raise ValueError("glong and dist must have same size")
    #
    # law of cosines
    #
    Rgal2 = R0**2. + dist_inp**2.
    Rgal2 = Rgal2 - 2.*R0*dist_inp*np.cos(np.deg2rad(glong_inp))
    Rgal = np.sqrt(Rgal2)
    #
    # Convert back to scalar if necessary
    #
    if dist_inp.size == 1:
        return Rgal[0]
    else:
        return Rgal

def calc_az(glong, dist, R0=8.34):
    """
    Return the Galactocentric azimuth of an object with a given
    Galacitic longitude and distance. Galactocentric azimuth is
    defined as zero in the direction of the Sun and increasing
    in the direction of the Solar orbit direction.

    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as dist.
      dist : scalar or 1-D array
             line-of-sight distance (kpc). If it is an array, it
             must have the same size as glong.
      R0 : scalar (optional)
           Galactocentric radius of the Sun.

    Returns: az
      az : scalar or 1-D array
           Galactocentric azimuth (degs). If glong and dist are
           scalars, it is a scalar. Otherwise, it has shape
           (dist.size).

    Raises:
      ValueError : if glong or dist are not 1-D; or
                   if glong and dist are arrays and not the same size
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, dist_inp = np.atleast_1d(glong, dist)
    # check shape of inputs
    if glong_inp.ndim != 1 or dist_inp.ndim != 1:
        raise ValueError("glong and dist must be 1-D")
    if glong_inp.size != 1 and glong_inp.size != dist_inp.size:
        raise ValueError("glong and dist must have same size")
    # ensure range [0,360) degrees
    fix_glong = glong_inp % 360.
    #
    # Compute Rgal
    #
    Rgal = calc_Rgal(fix_glong,dist_inp,R0=R0)
    Rgal_inp = np.atleast_1d(Rgal)
    #
    # law of cosines
    #
    cos_az = (R0**2. + Rgal_inp**2. - dist_inp**2.)/(2.*Rgal_inp*R0)
    #
    # Catch fringe cases
    #
    cos_az[cos_az > 1.] = 1.
    cos_az[cos_az < -1.] = -1.
    az = np.rad2deg(np.arccos(cos_az))
    #
    # Correct azimuth in 3rd and 4th quadrants
    #
    az[fix_glong > 180.] = 360. - az[fix_glong > 180.]
    #
    # Convert back to scalar if necessary
    #
    if dist_inp.size == 1:
        return az[0]
    else:
        return az

def calc_dist(az, Rgal, R0=8.34):
    """
    Return the line-of-sight distance of an object with a given
    Galacitocentric azimuth and radius.

    Parameters:
      az : scalar or 1-D array
           Galactocentric azimuth (deg). If it is an array, it must
           have the same size as dist.
      Rgal : scalar or 1-D array
             Galactocentric radius (kpc). If it is an array, it
             must have the same size as glong.
      R0 : scalar (optional)
           Galactocentric radius of the Sun.

    Returns: dist
      dist : scalar or 1-D array
             Line-of-sight distance (kpc). If az and Rgal are
             scalars, it is a scalar. Otherwise, it has shape
             (Rgal.size).

    Raises:
      ValueError : if az or Rgal are not 1-D; or
                   if az and Rgal are arrays and not the same size
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    az_inp, Rgal_inp = np.atleast_1d(az, Rgal)
    # check shape of inputs
    if az_inp.ndim != 1 or Rgal_inp.ndim != 1:
        raise ValueError("az and Rgal must be 1-D")
    if az_inp.size != 1 and az_inp.size != Rgal_inp.size:
        raise ValueError("az and Rgal must have same size")
    #
    # law of cosines
    #
    dist2 = R0**2. +Rgal_inp**2.
    dist2 = dist2 - 2.*R0*Rgal_inp*np.cos(np.deg2rad(az_inp))
    dist = np.sqrt(dist2)
    #
    # Convert back to scalar if necessary
    #
    if Rgal_inp.size == 1:
        return dist[0]
    else:
        return dist

def calc_glong(az, Rgal, R0=8.34):
    """
    Return the Galactic longitude of an object with a given
    Galacitocentric azimuth and radius. Galactic longitude is
    defined as zero in the direction of the Galactic Center and
    increasing in the direction of the Solar orbit direction.

    Parameters:
      az : scalar or 1-D array
           Galactocentric azimuth (deg). If it is an array, it must
           have the same size as dist.
      Rgal : scalar or 1-D array
             Galactocentric radius (kpc). If it is an array, it
             must have the same size as glong.
      R0 : scalar (optional)
           Galactocentric radius of the Sun.

    Returns: glong
      glong : scalar or 1-D array
              Galactic longitude (degs). If az and Rgal are
              scalars, it is a scalar. Otherwise, it has shape
              (Rgal.size).

    Raises:
      ValueError : if az or Rgal are not 1-D; or
                   if az and Rgal are arrays and not the same size
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    az_inp, Rgal_inp = np.atleast_1d(az, Rgal)
    # check shape of inputs
    if az_inp.ndim != 1 or Rgal_inp.ndim != 1:
        raise ValueError("az and Rgal must be 1-D")
    if az_inp.size != 1 and az_inp.size != Rgal_inp.size:
        raise ValueError("az and Rgal must have same size")
    # ensure range [0,360) degrees
    fix_az = az_inp % 360.
    #
    # Compute line of sight distance
    #
    dist = calc_dist(fix_az,Rgal_inp,R0=R0)
    dist_inp = np.atleast_1d(dist)
    #
    # law of cosines
    #
    cos_glong = (R0**2. + dist_inp**2. - Rgal_inp**2.)/(2.*dist_inp*R0)
    #
    # Catch fringe cases
    #
    cos_glong[cos_glong > 1.] = 1.
    cos_glong[cos_glong < -1.] = -1.
    glong = np.rad2deg(np.arccos(cos_glong))
    #
    # Correct longitude in 3rd and 4th quadrants
    #
    glong[fix_az > 180.] = 360. - glong[fix_az > 180.]
    #
    # Convert back to scalar if necessary
    #
    if Rgal_inp.size == 1:
        return glong[0]
    else:
        return glong

def correct_vlsr(glong, glat, vlsr, e_vlsr,
                 Ustd=10.27, Vstd=15.32, Wstd=7.74,
                 Usun=10.5, e_Usun=1.7, Vsun=14.4, e_Vsun=6.8,
                 Wsun=8.9, e_Wsun=0.9):
    """
    Return the "corrected" LSR velocity by updating the IAU-defined
    solar motion components (Ustd,Vstd,Wstd) to newly-measured
    values (Usun,Vsun,Wsun).
    Also computes the new LSR velocity uncertainty including the
    uncertainties in the newly-measured values (e_Usun,e_Vsun,e_Wsun). 

    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as glat, vlsr, and e_vlsr.
      glat : scalar or 1-D array
             Galactic latitude (deg). If it is an array, it
             must have the same size as glong, vlsr, and e_vlsr.
      vlsr : scalar or 1-D array
             Measured LSR velocity (km/s). If it is an array, it
             must have the same size as glong, glat, and e_vlsr.
      e_vlsr : scalar or 1-D array
               Uncertainty on measured LSR velocity (km/s). If it is 
               an array, it must have the same size as glong, glat, 
               and vlsr.
      Ustd,Vstd,Wstd : scalar (optional)
                       IAU-defined solar motion parameters (km/s).
      Usun,Vsun,Wsun : scalar (optional)
                       Newly measured solar motion parameters (km/s).
                       Defaults are from Reid et al. (2014)
      e_Usun,e_Vsun,e_Wsun : scalar (optional)
                       Newly measured solar motion parameter
                       uncertainties (km/s).
                       Defaults are from Reid et al. (2014)

    Returns: (new_vlsr,e_new_vlsr)
      new_vlsr : scalar or 1-D array
                 Re-computed LSR velocity. Same shape as vlsr.
      e_vlsr : scalar or 1-D array
               Re-computed LSR velocity uncertainty. Same shape as 
               e_vlsr.

    Raises:
      ValueError : if glong, glat, vlsr, and e_vlsr are not 1-D; or
                   if glong, glat, vlsr, and e_vlsr are arrays and 
                   not the same size
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, glat_inp, vlsr_inp, e_vlsr_inp = \
      np.atleast_1d(glong, glat, vlsr, e_vlsr)
    # check shape of inputs
    if (glong_inp.ndim != 1 or glat_inp.ndim != 1 or
        vlsr_inp.ndim != 1 or e_vlsr_inp.ndim != 1):
        raise ValueError("glong, glat, vlsr, and e_vlsr must be 1-D")
    if glong_inp.size != 1 and (glong_inp.size != glat_inp.size or
                                glong_inp.size != vlsr_inp.size or
                                glong_inp.size != e_vlsr_inp.size):
        raise ValueError("glong, glat, vlsr, and e_vlsr must have same size")
    #
    # Useful values
    #
    cos_glong = np.cos(np.deg2rad(glong_inp))
    sin_glong = np.sin(np.deg2rad(glong_inp))
    cos_glat = np.cos(np.deg2rad(glat_inp))
    sin_glat = np.sin(np.deg2rad(glat_inp))
    #
    # Compute heliocentric velocity by subtracting IAU defined solar
    # motion components
    #
    U_part = Ustd*cos_glong
    V_part = Vstd*sin_glong
    W_part = Wstd*sin_glat
    UV_part = (U_part+V_part)*cos_glat
    v_helio = vlsr_inp - UV_part - W_part
    #
    # Compute corrected VLSR
    #
    U_part = Usun*cos_glong
    V_part = Vsun*sin_glong
    W_part = Wsun*sin_glat
    UV_part = (U_part+V_part)*cos_glat
    new_vlsr = v_helio + UV_part + W_part
    #
    # Compute corrected LSR velocity uncertainty
    #
    U_part = (e_Usun*cos_glong*cos_glat)**2.
    V_part = (e_Vsun*sin_glong*cos_glat)**2.
    W_part = (e_Wsun*sin_glat)**2.
    e_new_vlsr = np.sqrt(e_vlsr_inp**2.+U_part+V_part+W_part)
    #
    # Convert back to scalar if necessary
    #
    if glong_inp.size == 1:
        return new_vlsr[0],e_new_vlsr[0]
    else:
        return new_vlsr,e_new_vlsr

def calc_anderson2012_uncertainty(glong, vlsr):
    """
    Return the Anderson+2012 kinematic distance uncertainties.

    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as vlsr
      vlsr : scalar or 1-D array
             Measured LSR velocity (km/s). If it is an array, it
             must have the same size as glong, glat, and e_vlsr.

    Returns: (near_err, far_err, tangent_err)
      near_err : scalar or 1-D array
                 Anderson+2012 near distance uncertainty
      far_err : scalar or 1-D array
                Anderson+2012 tangent distance uncertainty
      tangent_err : scalar or 1-D array
                    Anderson+2012 far distance uncertainty

    Raises:
      ValueError : if glong and vlsr are not 1-D; or
                   if glong and vlsr are arrays and 
                   not the same size
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, vlsr_inp =  np.atleast_1d(glong, vlsr)
    # check shape of inputs
    if (glong_inp.ndim != 1 or vlsr_inp.ndim != 1):
        raise ValueError("glong and vlsr must be 1-D")
    if glong_inp.size != 1 and glong_inp.size != vlsr_inp.size:
        raise ValueError("glong and vlsr must have same size")
    #
    # Read Anderson+2012 uncertainty data
    #
    a12data = readsav("data/curve_data_wise_small.sav",python_dict=True)
    a12data = a12data['curve_data_wise_small'][0]
    a12_near_err = a12data['big_percentages_near']/100.
    a12_far_err = a12data['big_percentages_far']/100.
    a12_glongs = a12data['glong']
    a12_vlsrs = a12data['velbinning']
    #
    # find matching longitudes and velocities
    #
    best_glong = np.array([np.nanargmin(np.abs(gl-a12_glongs))
                           for gl in glong_inp])
    best_vlsr = np.array([np.nanargmin(np.abs(vl-a12_vlsrs))
                          for vl in vlsr_inp])
    #
    # Get distance uncertainties
    #
    near_err = np.array(a12_near_err[best_glong,best_vlsr])
    far_err = np.array(a12_far_err[best_glong,best_vlsr])
    tangent_err = np.nanmax(np.vstack((near_err,far_err)),axis=0)
    #
    # Convert back to scalar if necessary
    #
    if glong_inp.size == 1:
        return near_err[0],far_err[0],tangent_err[0]
    else:
        return near_err,far_err,tangent_err
