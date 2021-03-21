#!/usr/bin/env python
"""
rotcurve_kd.py

Utilities to calculate kinematic distances using the traditional
rotation curve methods.

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
2020-02-20 Trey V. Wenger updates for v2.0
"""

import os
import importlib
import numpy as np
import pathos.multiprocessing as mp
import dill

from kd import kd_utils

class Worker:
    """
    Multiprocessing wrapper class
    """
    def __init__(self, glong, glat, velo, velo_err, dists, glong_grid,
                 dist_grid, velo_tol, rotcurve, resample, size,
                 peculiar, use_kriging):
        self.glong = glong
        self.glat = glat
        self.velo = velo
        self.velo_err = velo_err
        self.dists = dists
        self.glong_grid = glong_grid
        self.dist_grid = dist_grid
        self.velo_tol = velo_tol
        self.resample = resample
        self.size = size
        self.rotcurve = rotcurve
        self.peculiar = peculiar
        self.use_kriging = use_kriging
        #
        # Get rotation curve module
        #
        self.rotcurve_module = importlib.import_module('kd.'+rotcurve)
        #
        # Get nominal rotation curve parameters
        #
        if rotcurve == "cw21_rotcurve":
            # Load pickle file
            infile = os.path.join(os.path.dirname(__file__), "cw21_kde_krige.pkl")
            # infile contains: full KDE + KDEs of each component (e.g. "R0")
            #                  + kriging function + kriging thresholds
            with open(infile, "rb") as f:
                self.kde = dill.load(f)["full"]
            (self.nominal_params, self.Rgal,
             self.cos_az, self.sin_az) = self.rotcurve_module.nominal_params(
                glong=glong, glat=glat, dist=dist_grid, use_kriging=use_kriging)
        elif not use_kriging:
            self.nominal_params = self.rotcurve_module.nominal_params()
            self.Rgal = self.cos_az = self.sin_az = None
        else:
            raise ValueError("kriging is only supported for 'cw21_rotcurve'")

    def work(self, snum):
        #
        # Random seed at each iteration
        #
        np.random.seed()
        #
        # Resample velocity and rotation curve parameters
        #
        if self.resample:
            if self.rotcurve == "cw21_rotcurve":
                params = self.rotcurve_module.resample_params(
                    self.kde, size=len(self.glong),
                    nom_params=self.nominal_params,
                    use_kriging=self.use_kriging)
                # Allow calculation of Rgal & az with new params
                self.Rgal = self.cos_az = self.sin_az = None
            else:
                params = self.rotcurve_module.resample_params(
                    size=len(self.glong))
            velo_sample = np.random.normal(
                loc=self.velo, scale=self.velo_err)
        else:
            params = self.nominal_params
            velo_sample = self.velo
        #
        # Calculate LSR velocity at each (glong, distance) point
        #
        if self.rotcurve == "cw21_rotcurve":
            grid_vlsrs = self.rotcurve_module.calc_vlsr(
                self.glong_grid, self.glat, self.dist_grid,
                Rgal=self.Rgal, cos_az=self.cos_az, sin_az=self.sin_az,
                peculiar=self.peculiar, **params)
        elif self.rotcurve == "reid19_rotcurve":
            grid_vlsrs = self.rotcurve_module.calc_vlsr(
                self.glong_grid, self.glat, self.dist_grid,
                peculiar=self.peculiar, **params)
        else:
            grid_vlsrs = self.rotcurve_module.calc_vlsr(
                self.glong_grid, self.glat, self.dist_grid, **params)
        #
        # Get index of tangent point along each direction
        #
        tan_idxs = np.array([
            np.argmax(vlsr) if l < 90. else np.argmin(vlsr) if l > 270.
            else -1 for l, vlsr in zip(self.glong, grid_vlsrs.T)])
        #
        # Get index of near and far distances along each direction
        #
        near_idxs = np.array([
            np.argmin(np.abs(vlsr[:tan_idx]-v)) if
            (tan_idx > 0 and np.min(np.abs(vlsr[:tan_idx]-v)) < self.velo_tol)
            else -1 for v, vlsr, tan_idx in zip(velo_sample, grid_vlsrs.T, tan_idxs)])
        far_idxs = np.array([
            tan_idx+np.argmin(np.abs(vlsr[tan_idx:]-v)) if
            (tan_idx > 0 and np.min(np.abs(vlsr[tan_idx:]-v)) < self.velo_tol)
            else np.argmin(np.abs(vlsr-v)) if
            (tan_idx == -1 and np.min(np.abs(vlsr-v)) < self.velo_tol)
            else -1 for v, vlsr, tan_idx in zip(velo_sample, grid_vlsrs.T, tan_idxs)])
        #
        # Get VLSR of tangent point
        #
        vlsr_tan = np.array([
            vlsr[tan_idx] if tan_idx > 0 else np.nan
            for vlsr, tan_idx in zip(grid_vlsrs.T, tan_idxs)])
        #
        # Get distances
        #
        tan_dists = self.dists[tan_idxs]
        tan_dists[tan_idxs == -1] = np.nan
        near_dists = self.dists[near_idxs]
        near_dists[near_idxs == -1] = np.nan
        far_dists = self.dists[far_idxs]
        far_dists[far_idxs == -1] = np.nan
        if self.rotcurve == "cw21_rotcurve":
            # Include (potentially resampled) Zsun and roll values
            Rgal = kd_utils.calc_Rgal(
                self.glong, self.glat, far_dists.T, R0=params['R0'],
                Zsun=params['Zsun'], roll=params['roll'], use_Zsunroll=True).T
            Rtan = kd_utils.calc_Rgal(
                self.glong, self.glat, tan_dists.T, R0=params['R0'],
                Zsun=params['Zsun'], roll=params['roll'], use_Zsunroll=True).T
        else:
            Rgal = kd_utils.calc_Rgal(
                self.glong, self.glat, far_dists.T, R0=params['R0']).T
            Rtan = kd_utils.calc_Rgal(
                self.glong, self.glat, tan_dists.T, R0=params['R0']).T
        return (tan_dists, near_dists, far_dists, vlsr_tan, Rgal, Rtan)

def rotcurve_kd(glong, glat, velo, velo_err=None, velo_tol=0.1,
                rotcurve='cw21_rotcurve',
                dist_res=0.001, dist_min=0.001, dist_max=30.,
                resample=False, size=1, processes=None,
                peculiar=False, use_kriging=False):
    """
    Return the kinematic near, far, and tanget distance for a
    given Galactic longitude and LSR velocity assuming
    a given rotation curve.

    Parameters:
      glong, glat :: scalar or array of scalars
        Galactic longitude and latitude (deg). If it is an array, it must have the
        same shape as velo.

      velo :: scalar or array of scalars
        LSR velocity (km/s). If it is an array, it must have the same
        shape as glong.

      velo_err :: scalar or array of scalars (optional)
        LSR velocity uncertainty (km/s). If it is an array, it must
        have the same shape as velo. Otherwise, this scalar
        uncertainty is applied to all velos.

      velo_tol :: scalar (optional)
        LSR velocity tolerance to consider a match between velo and
        rotation curve velocity

      rotcurve :: string (optional)
        rotation curve model

      dist_res :: scalar (optional)
        line-of-sight distance resolution when calculating kinematic
        distance (kpc)

      dist_min :: scalar (optional)
        minimum line-of-sight distance when calculating kinematic
        distance (kpc)

      dist_max :: scalar (optional)
        maximum line-of-sight distance when calculating kinematic
        distance (kpc)

      resample :: bool (optional)
        if True, use resampled rotation curve parameters and LSR
        velocities.

      size :: integer (optional)
        if resample is True, generate this many samples

      processes :: integer (optional)
        number of simultaneous workers to use
        if None, automatically assign workers based on system's core count

      peculiar :: boolean (optional)
        only supported for "cw21_rotcurve" and "reid19_rotcurve"
        if True, include HMSFR peculiar motion component

      use_kriging :: boolean (optional)
        only supported for rotcurve = "cw21_rotcurve"
        if True, estimate individual Upec & Vpec from kriging program
        if False, use average Upec & Vpec

    Returns: output
      output["Rgal"] :: scalar or array of scalars
        Galactocentric radius (kpc).

      output["Rtan"] :: scalar or array of scalars
        Galactocentric radius of tangent point (kpc).

      output["near"] :: scalar or array of scalars
        kinematic near distance (kpc)

      output["far"] :: scalar or array of scalars
        kinematic far distance (kpc)

      output["tangent"] :: scalar or array of scalars
        kinematic tangent distance (kpc)

      output["vlsr_tangent"] :: scalar or array of scalars
        LSR velocity of tangent point (km/s)

      If glong and velo are scalars, each of these is a scalar.
      Otherwise they have the same shape as input glong and velo.

      If resample is True and size > 1, the samples are stored along
      the last axis of each element.

    Raises:
      ValueError : if glong and velo are not the same shape
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    input_scalar = np.isscalar(glong)
    glong, glat, velo = np.atleast_1d(glong, glat, velo)
    inp_shape = glong.shape
    glong = glong.flatten()
    glat = glat.flatten()
    velo = velo.flatten()
    # check shape of inputs
    if glong.shape != velo.shape or glong.shape != glat.shape:
        raise ValueError("glong, glat, and velo must have same shape")
    if (velo_err is not None and not np.isscalar(velo_err)):
        velo_err = velo_err.flatten()
        if velo_err.shape != velo.shape:
            raise ValueError("velo_err must be scalar or have same shape as velo")
    #
    # Default velo_err to 0, sample size to 1
    #
    elif velo_err is None:
        velo_err = 0.
    if not resample:
        size = 1
    if size < 1:
        raise ValueError("size must be >= 1")
    # ensure range [0,360) degrees
    glong = glong % 360.
    #
    # Create array of distances, then grid of (dists, glongs)
    #
    dists = np.arange(dist_min, dist_max+dist_res, dist_res)
    dist_grid, glong_grid = np.meshgrid(dists, glong, indexing='ij')
    #
    # Initialize worker
    #
    worker = Worker(glong, glat, velo, velo_err, dists, glong_grid, dist_grid,
                    velo_tol, rotcurve, resample, size, peculiar, use_kriging)
    # if processes is None:
    #     with mp.ProcessPool() as pool:
    #         print("Number of rotcurve_kd nodes:", pool.nodes)
    #         results = pool.map(worker.work, range(size))
    # else:
    #     with mp.ProcessPool(nodes=processes) as pool:
    #         print("Number of rotcurve_kd nodes:", pool.nodes)
    #         results = pool.map(worker.work, range(size))
    with mp.Pool(processes=processes) as pool:
        print("Number of rotcurve_kd nodes:", pool._processes)
        results = pool.map(worker.work, range(size))
    # Free memory even though pool should be closed already
    print("Closing pool in rotcurve_kd")
    pool.close()
    pool.join()
    #
    # Store results
    #
    tan_samples = np.ones((len(glong), size), dtype=float)*np.nan
    near_samples = np.ones((len(glong), size), dtype=float)*np.nan
    far_samples = np.ones((len(glong), size), dtype=float)*np.nan
    vlsr_tan_samples = np.ones((len(glong), size), dtype=float)*np.nan
    Rgal_samples = np.ones((len(glong), size), dtype=float)*np.nan
    Rtan_samples = np.ones((len(glong), size), dtype=float)*np.nan
    for snum, result in enumerate(results):
        tan_samples[:, snum] = result[0]
        near_samples[:, snum] = result[1]
        far_samples[:, snum] = result[2]
        vlsr_tan_samples[:, snum] = result[3]
        Rgal_samples[:, snum] = result[4]
        Rtan_samples[:, snum] = result[5]
    #
    # Convert back to scalars if necessary
    #
    output = {"Rgal": Rgal_samples, "Rtan": Rtan_samples,
              "near": near_samples, "far": far_samples,
              "tangent": tan_samples,
              "vlsr_tangent": vlsr_tan_samples}
    if size == 1:
        for key in output:
            output[key] = np.squeeze(output[key], axis=-1)
    if input_scalar:
        for key in output:
            output[key] = output[key][0]
    else:
        for key in output:
            if size == 1:
                output[key] = output[key].reshape(inp_shape)
            else:
                output[key] = output[key].reshape(inp_shape+(size,))
    return output
