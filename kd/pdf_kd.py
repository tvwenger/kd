#!/usr/bin/env python
"""
pdf_kd.py

Utilities to calculate kinematic distances and kinematic distance
uncertainties using the Monte Carlo method.

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
2018-01-17 Trey V. Wenger removed multiprocessing. The overhead
  of multiprocessing dominated the computation time.
2020-02-20 Trey V. Wenger updates for v2.0
"""

import numpy as np
import matplotlib.pyplot as plt
import pathos.multiprocessing as mp

from kd.rotcurve_kd import rotcurve_kd
from kd.kd_utils import calc_hpd

def calc_hpd_wrapper(args):
    """
    Multiprocessing wrapper from KDE results.
    """
    return calc_hpd(args[0], args[1], alpha=args[2], pdf_bins=args[3])

def pdf_kd(glong, glat, velo, velo_err=None, rotcurve='cw21_rotcurve',
           rotcurve_dist_res=0.001, rotcurve_dist_max=30.,
           pdf_bins=1000, num_samples=10000,
           plot_pdf=False, plot_prefix='pdf_',
           peculiar=False, use_kriging=False):
    """
    Return the kinematic near, far, and tanget distance and distance
    uncertainties for a given Galactic longitude and LSR velocity
    assuming a given rotation curve. Generate distance posteriors by
    resampling within rotation curve parameter and velocity
    uncertainties. Peak of posterior and 68.3% minimum width Bayesian
    credible interval (BCI) are returned.

    Parameters:
      glong, glat :: scalar or array of scalars
        Galactic longitude and latitude (deg).

      velo :: scalar or array of scalars
        LSR velocity (km/s).

      velo_err :: scalar or array of scalars (optional)
        LSR velocity uncertainty (km/s). If it is an array, it must
        have the same size as velo. Otherwise, this scalar
        uncertainty is applied to all velos.

      rotcurve :: string (optional)
        rotation curve model

      rotcurve_dist_res :: scalar (optional)
        line-of-sight distance resolution when calculating kinematic
        distance with rotcurve_kd (kpc)

      rotcurve_dist_max :: scalar (optional)
        maximum line-of-sight distance when calculating kinematic
        distance with rotcurve_kd (kpc)

      pdf_bins :: integer (optional)
        number of bins used to calculate PDF

      num_samples :: integer (optional)
        Number of MC samples to use when generating PDF

      plot_pdf :: bool (optional)
        If True, plot each PDF. Filenames are:
        plot_prefix+"{glong}_{velo}.pdf".

      plot_prefix :: string (optional)
        The prefix for the plot filenames.

      peculiar :: boolean (optional)
        Only supported for "cw21_rotcurve" and "reid19_rotcurve"
        If True, include HMSFR peculiar motion component

      use_kriging :: boolean (optional)
        Only supported for rotcurve = "cw21_rotcurve"
        If True, estimate individual Upec & Vpec from kriging program
        If False, use average Upec & Vpec

    Returns: output
      output["Rgal"] :: scalar or array of scalars
        Galactocentric radius (kpc).

      output["Rtan"] :: scalar or array of scalars
        Galactocentric radius of tangent point (kpc).

      output["near"] :: scalar or array of scalars
        kinematic near distance (kpc)

      output["far"] :: scalar or array of scalars
        kinematic far distance (kpc)

      output["distance"] :: scalar or array of scalars
        kinematic distance (near and far combined) (kpc)

      output["tangent"] :: scalar or array of scalars
        kinematic tangent distance (kpc)

      output["vlsr_tangent"] :: scalar or array of scalars
        LSR velocity of tangent point (km/s)

      Each of these values is the mode of the posterior distribution.
      Also included in the dictionary for each of these parameters
      are param+"_err_neg" and param+"_err_pos", which define the
      68.3% Bayesian credible interval around the mode, and
      param+"_kde", which is the kernel density estimator fit to the
      posterior samples.

    Raises:
      ValueError : if glong and velo are not the same shape
                   if velo_err is an array and not the same shape as
                   glong and velo
    """
    #
    # check inputs
    #
    # check shape of inputs
    input_scalar = np.isscalar(glong)
    glong, glat, velo = np.atleast_1d(glong, glat, velo)
    if glong.shape != velo.shape:
        raise ValueError("glong and velo must have same shape")
    if (velo_err is not None and not np.isscalar(velo_err) and
        velo_err.shape != velo.shape):
        raise ValueError("velo_err must be scalar or have same shape as velo")
    #
    # Storage for final PDF kinematic distance results
    #
    results = {"Rgal": np.zeros(glong.shape),
               "Rgal_kde": np.empty(shape=glong.shape,
                                    dtype=object),
               "Rgal_err_neg": np.zeros(glong.shape),
               "Rgal_err_pos": np.zeros(glong.shape),
               "Rtan": np.zeros(glong.shape),
               "Rtan_kde": np.empty(shape=glong.shape,
                                    dtype=object),
               "Rtan_err_neg": np.zeros(glong.shape),
               "Rtan_err_pos": np.zeros(glong.shape),
               "near": np.zeros(glong.shape),
               "near_kde": np.empty(shape=glong.shape,
                                    dtype=object),
               "near_err_neg": np.zeros(glong.shape),
               "near_err_pos": np.zeros(glong.shape),
               "far": np.zeros(glong.shape),
               "far_kde": np.empty(shape=glong.shape,
                                   dtype=object),
               "far_err_neg": np.zeros(glong.shape),
               "far_err_pos": np.zeros(glong.shape),
               "distance": np.zeros(glong.shape),
               "distance_kde": np.empty(shape=glong.shape,
                                       dtype=object),
               "distance_err_neg": np.zeros(glong.shape),
               "distance_err_pos": np.zeros(glong.shape),
               "tangent": np.zeros(glong.shape),
               "tangent_kde": np.empty(shape=glong.shape,
                                       dtype=object),
               "tangent_err_neg": np.zeros(glong.shape),
               "tangent_err_pos": np.zeros(glong.shape),
               "vlsr_tangent": np.zeros(glong.shape),
               "vlsr_tangent_kde": np.empty(shape=glong.shape,
                                            dtype=object),
               "vlsr_tangent_err_neg": np.zeros(glong.shape),
               "vlsr_tangent_err_pos": np.zeros(glong.shape)}
    #
    # Calculate rotcurve kinematic distance
    #
    kd_out = rotcurve_kd(
        glong, glat, velo, velo_err=velo_err, velo_tol=0.1,
        rotcurve=rotcurve, dist_res=rotcurve_dist_res, dist_min=0.01,
        dist_max=rotcurve_dist_max, resample=True, size=num_samples,
        peculiar=peculiar, use_kriging=use_kriging)
    #
    # Set up multiprocessing for fitting KDEs
    #
    kdtypes = ["Rgal", "Rtan", "near", "far", "tangent", "vlsr_tangent"]
    kdetypes = ["pyqt", "pyqt", "pyqt", "pyqt", "pyqt", "scipy"]
    args = []
    for kdtype, kdetype in zip(kdtypes, kdetypes):
        for i in np.ndindex(glong.shape):
            args.append((
                kd_out[kdtype][i], kdetype, 0.683, pdf_bins))
    #
    # Also, distance (near + far)
    # check if both are nan -> use tangent distance
    #
    kdtypes += ["distance"]
    kdetypes += ["pyqt"]
    for i in np.ndindex(glong.shape):
        is_tangent = np.isnan(kd_out['near'][i])*np.isnan(kd_out['far'][i])
        samples = kd_out['tangent'][i][is_tangent]
        samples = np.concatenate((
            samples, kd_out['near'][i][~is_tangent],
            kd_out['far'][i][~is_tangent]))
        args.append((samples, 'pyqt', 0.683, pdf_bins))
    #
    # Get results
    #
    nresult = 0
    with mp.Pool() as pool:
        kde_results = pool.map(calc_hpd_wrapper, args)
    for kdtype, kdetype in zip(kdtypes, kdetypes):
        for i in np.ndindex(glong.shape):
            kde, mode, lower, upper = kde_results[nresult]
            results[kdtype][i] = mode
            results[kdtype+"_kde"][i] = kde
            results[kdtype+"_err_neg"][i] = mode-lower
            results[kdtype+"_err_pos"][i] = upper-mode
            nresult += 1
    #
    # Plot PDFs
    #
    if plot_pdf:
        for i in np.ndindex(glong.shape):
            #
            # Set-up figure
            #
            fig, axes = plt.subplots(6, figsize=(8.5, 11))
            axes[0].set_title(
                r"PDFs for ($\ell$, $v$) = ("
                "{0:.1f}".format(glong[i])+r"$^\circ$, "
                "{0:.1f}".format(velo[i])+r"km s$^{-1}$)")
            #
            # Compute "traditional" kinematic distances
            #
            rot_kd = rotcurve_kd(
                glong[i], glat[i], velo[i], rotcurve=rotcurve,
                dist_res=rotcurve_dist_res,
                dist_max=rotcurve_dist_max)
            kdtypes = ["Rgal", "Rtan", "near", "far", "distance", "tangent"]
            labels = [r"$R$ (kpc)", r"$R_{\rm tan}$ (kpc)",
                      r"$d_{\rm near}$ (kpc)", r"$d_{\rm far}$ (kpc)",
                      r"$d$ (kpc)", r"$d_{\rm tan}$ (kpc)"]
            for ax, kdtype, label in zip(axes, kdtypes, labels):
                if kdtype == 'distance':
                    is_tangent = np.isnan(kd_out['near'][i])*np.isnan(kd_out['far'][i])
                    out = kd_out['tangent'][i][is_tangent]
                    out = np.concatenate((
                        out, kd_out['near'][i][~is_tangent],
                        kd_out['far'][i][~is_tangent]))
                else:
                    out = kd_out[kdtype][i]
                peak = results[kdtype][i]
                kde = results[kdtype+"_kde"][i]
                err_neg = results[kdtype+"_err_neg"][i]
                err_pos = results[kdtype+"_err_pos"][i]
                # find bad data
                out = out[~np.isnan(out)]
                # skip if kde failed (all data is bad)
                if kde is None:
                    continue
                # set-up bins
                binwidth = (np.max(out)-np.min(out))/20.
                bins = np.arange(
                    np.min(out), np.max(out)+binwidth, binwidth)
                distwidth = (np.max(out)-np.min(out))/200.
                dists = np.arange(
                    np.min(out), np.max(out)+distwidth, distwidth)
                pdf = kde(dists)
                ax.hist(
                    out, bins=bins, density=True, facecolor='white',
                    edgecolor='black', lw=2, zorder=1)
                ax.plot(dists, pdf, 'k-', zorder=3)
                err_dists = np.arange(
                    peak-err_neg, peak+err_pos, distwidth)
                err_pdf = kde(err_dists)
                ax.fill_between(
                    err_dists, 0, err_pdf, color='gray', alpha=0.5,
                    zorder=2)
                ax.axvline(
                    peak, linestyle='solid',
                    color='k', zorder=3)
                if kdtype == 'distance':
                    ax.axvline(rot_kd['near'], linestyle='dashed',
                               color='k', zorder=3)
                    ax.axvline(rot_kd['far'], linestyle='dashed',
                               color='k', zorder=3)
                else:
                    ax.axvline(
                        rot_kd[kdtype], linestyle='dashed',
                        color='k', zorder=3)
                ax.set_xlabel(label)
                ax.set_ylabel("Normalized PDF")
                ax.set_xlim(np.min(out), np.max(out))
                # turn off grid
                ax.grid(False)
            plt.tight_layout()
            fname = "{0}{1}_{2}.pdf".format(
                plot_prefix, glong[i], velo[i])
            plt.savefig(fname)
            plt.close(fig)
    if input_scalar:
        for key in results:
            results[key] = results[key][0]
    return results
