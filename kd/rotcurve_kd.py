#!/usr/bin/env python
"""
rotcurve_kd.py

Utilities to calculate kinematic distances using the traditional
rotation curve methods.

2017-04-12 Trey V. Wenger
"""

import importlib
import numpy as np

from kd import kd_utils

def rotcurve_kd(glong,velo,velo_tol=1.e-1,
                rotcurve='reid14_rotcurve',
                dist_res=1.e-2,dist_min=0.01,dist_max=30.,
                resample=False):
    """
    Return the kinematic near, far, and tanget distance for a
    given Galactic longitude and LSR velocity assuming
    a given rotation curve.

    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as velo.
      velo : scalar or 1-D array
             LSR velocity (km/s). If it is an array, it must
             have the same size as glong.
      velo_tol : scalar (optional)
                 LSR velocity tolerance to consider a match between
                 velo and rotation curve velocity
      rotcurve : string (optional)
                 rotation curve model
      dist_res : scalar (optional)
                 line-of-sight distance resolution when calculating
                 kinematic distance (kpc)
      dist_min : scalar (optional)
                 minimum line-of-sight distance when calculating
                 kinematic distance (kpc)
      dist_max : scalar (optional)
                 maximum line-of-sight distance when calculating
                 kinematic distance (kpc)
      resample : bool (optional)
                 if True, resample rotation curve parameters within
                 uncertainties
    
    Returns: output
      output["Rgal"] : scalar or 1-D array
                       Galactocentric radius (kpc).
      output["Rtan"] : scalar or 1-D array
                       Galactocentric radius of tangent point (kpc).
      output["near"] : scalar or 1-D array
                       kinematic near distance (kpc)
      output["far"] : scalar or 1-D array
                      kinematic far distance (kpc)
      output["tangent"] : scalar or 1-D array
                          kinematic tangent distance (kpc)
      output["vlsr_tangent"] : scalar or 1-D array
                               LSR velocity of tangent point (km/s)
      If glong and velo are scalars, each of these is a scalar.
      Otherwise they have shape (velo.size).

    Raises:
      ValueError : if glong and velo are not 1-D; or
                   if glong and velo are arrays and not the same size
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, velo_inp = np.atleast_1d(glong, velo)
    # check shape of inputs
    if glong_inp.ndim != 1 or velo_inp.ndim != 1:
        raise ValueError("glong and velo must be 1-D")
    if glong_inp.size != velo_inp.size:
        raise ValueError("glong and velo must be same size")
    # ensure range [0,360) degrees
    fix_glong = glong_inp % 360.
    #
    # Create array of distances
    #
    dists = np.arange(dist_min,dist_max,dist_res)   
    #
    # Calculate LSR velocity at each (glong,distance) point using
    # rotation curve
    #
    rotcurve_module = importlib.import_module('kd.'+rotcurve)
    vlsrs = np.zeros((fix_glong.size,dists.size))
    params = [None]*fix_glong.size
    for ind,l in enumerate(fix_glong):
        vlsr,param = \
          rotcurve_module.calc_vlsr(l,dists,resample=resample)
        vlsrs[ind] = vlsr
        params[ind] = param
    #
    # Storage for kinematic distance indicies
    #
    near_ind = np.ma.masked_all(velo_inp.size,dtype=np.int)
    far_ind = np.ma.masked_all(velo_inp.size,dtype=np.int)
    tan_ind = np.ma.masked_all(fix_glong.size,dtype=np.int)
    #
    # Find kinematic distance indicies
    #
    for i,(l,v) in enumerate(zip(fix_glong,velo_inp)):
        #
        # 2nd or 3rd quadrants
        #
        if (90. <= l <= 270.):
            #
            # far distance indicies
            #
            velo_diff = np.min(np.abs(vlsrs[i] - v))
            best_ind = np.argmin(np.abs(vlsrs[i] - v))
            if velo_diff < velo_tol:
                far_ind[i] = best_ind
        #
        # 1st or 4th quadrants
        #
        else:
            #
            # tangent distance indicies
            #
            if l <= 90.: tan_ind[i] = np.argmax(vlsrs[i])
            if l >= 270.: tan_ind[i] = np.argmin(vlsrs[i])
            # mask if tangent distance is zero
            if tan_ind[i] == 0:
                tan_ind.mask[i] = True
                continue
            #
            # near distance indicies
            #
            velo_diff = np.min(np.abs(vlsrs[i,0:tan_ind[i]]-v))
            best_ind = np.argmin(np.abs(vlsrs[i,0:tan_ind[i]]-v))
            if velo_diff < velo_tol:
                near_ind[i] = best_ind
            #
            # far distance indicies
            #
            velo_diff = np.min(np.abs(vlsrs[i,tan_ind[i]:]-v))
            best_ind = np.argmin(np.abs(vlsrs[i,tan_ind[i]:]-v))
            best_ind += tan_ind[i]
            if velo_diff < velo_tol:
                far_ind[i] = best_ind
    #
    # Assign distances from indicies, mask where appropriate
    #
    near_dist = np.array([dists[ind] if ind is not np.ma.masked
                          else np.nan for ind in near_ind])
    far_dist = np.array([dists[ind] if ind is not np.ma.masked
                         else np.nan for ind in far_ind])
    Rgal = np.array([kd_utils.calc_Rgal(l,d,R0=params[ind]["R0"])
                     for ind,(l,d) in
                     enumerate(zip(fix_glong,far_dist))])
    tan_dist = np.array([dists[ind] if ind is not np.ma.masked
                         else np.nan for ind in tan_ind])
    Rtan = np.array([kd_utils.calc_Rgal(l,d,R0=params[ind]["R0"])
                     for ind,(l,d) in
                     enumerate(zip(fix_glong,tan_dist))])
    #
    # Assign tangent point velocities
    #
    vlsr_tan = np.array([vlsrs[i][t] if t is not np.ma.masked
                         else np.nan for i,t in enumerate(tan_ind)])
    #
    # Convert back to scalars if necessary
    #
    if len(fix_glong) == 1:
        return {"Rgal":Rgal[0], "Rtan":Rtan[0], "near":near_dist[0],
                "far":far_dist[0], "tangent":tan_dist[0],
                "vlsr_tangent":vlsr_tan[0]}
    else:
        return {"Rgal":Rgal, "Rtan":Rtan, "near":near_dist,
                "far":far_dist, "tangent":tan_dist,
                "vlsr_tangent":vlsr_tan}
