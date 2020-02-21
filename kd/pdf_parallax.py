#!/usr/bin/env python
"""
pdf_parallax.py

Utility to calculate PDF parallax distances.

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

2019-01-19 Trey V. Wenger
2020-02-20 Trey V. Wener updates for v2.0
"""

import numpy as np
import matplotlib.pyplot as plt

from kd.parallax import parallax
from kd.kd_utils import calc_hpd

# Solar Galactocentric radius and error from Reid+2019
__R0 = 8.15 # kpc
__R0_err = 0.15 # kpc

def pdf_parallax(glong, plx, plx_err=None, dist_max=30., R0=__R0,
                 R0_err=__R0_err, pdf_bins=100, num_samples=1000,
                 plot_pdf=False, plot_prefix='pdf_'):
    """
    Return the parallax Galactocentric radius and distance given a
    parallax and parallax uncertainty. Generate distance posterior by
    resampling parallax and R0 within uncertainties. Peak of posterior
    and 68.3% minimum width Bayesian credible interval (BCI) are
    returned.

    Parameters:
      glong :: scalar or array of scalars
        Galactic longitude (deg).

      plx :: scalar or array of scalars
        Parallax in milli-arcseconds

      plx_err :: scalar or 1-D (optional)
        Parallax uncertainty in milli-arcseconds. If it is an array,
        it must have the same size as plx. Otherwise, this scalar
        uncertainty is applied to all plxs.

      dist_max :: scalar (optional)
        The maximum parallax distance to compute (kpc)

      R0, R0_err :: scalar (optional)
        Solar Galactocentric radius and uncertainty (kpc)

      pdf_bins :: integer (optional)
        number of bins used to calculate PDF

      num_samples :: integer (optional)
        Number of MC samples to use when generating PDF

      plot_pdf :: bool (optional)
        If True, plot each PDF. Filenames are
        plot_prefix+"{plx}_{err}.pdf".

      plot_prefix :: string (optional)
        The prefix for the plot filenames.

    Returns: output
      output["Rgal"] :: scalar or array of scalars
        Galactocentric radius (kpc).

      output["distance"] : scalar or array of scalars
        parallax distance (kpc)

      Each of these values is the mode of the posterior distribution.
      Also included in the dictionary for each of these parameters
      are param+"_err_neg" and param+"_err_pos", which define the
      68.3% Bayesian credible interval around the mode, and
      param+"_kde", which is the kernel density estimator fit to the
      posterior samples.

    Raises:
      ValueError : if glong and plx are not 1-D; or
                   if glong and plx are arrays and not the same size

    """
    #
    # check inputs
    #
    # check shape of inputs
    glong, plx = np.atleast_1d(glong, plx)
    if glong.shape != plx.shape:
        raise ValueError("glong and plx must have same size")
    if (plx_err is not None and not np.isscalar(plx_err) and
        plx_err.shape != plx.shape):
        raise ValueError("plx_err must be scalar or have same shape as plx")
    #
    # Storage for final PDF parallax distance results
    #
    results = {"Rgal": np.zeros(plx.shape),
               "Rgal_kde": np.empty(shape=plx.shape, dtype=object),
               "Rgal_err_neg": np.zeros(plx.shape),
               "Rgal_err_pos": np.zeros(plx.shape),
               "distance": np.zeros(plx.shape),
               "distance_kde": np.empty(shape=plx.shape, dtype=object),
               "distance_err_neg": np.zeros(plx.shape),
               "distance_err_pos": np.zeros(plx.shape)}
    #
    # Calculate parallax distances
    #
    plx_out = parallax(
        glong, plx, plx_err=plx_err, dist_max=dist_max, R0=R0,
        R0_err=R0_err, resample=True, size=num_samples)
    #
    # Get modes and BICs
    #
    kdtypes = ["Rgal", "distance"]
    kdetypes = ["pyqt", "pyqt"]
    for kdtype, kdetype in zip(kdtypes, kdetypes):
        if glong.size == 1:
            kde, mode, lower, upper = calc_hpd(
                plx_out[kdtype], kdetype, alpha=0.683,
                pdf_bins=pdf_bins)
            results[kdtype] = mode
            results[kdtype+"_kde"] = kde
            results[kdtype+"_err_neg"] = mode-lower
            results[kdtype+"_err_pos"] = upper-mode
        else:
            for i in np.ndindex(glong.shape):
                kde, mode, lower, upper = calc_hpd(
                    plx_out[kdtype][i], kdetype, alpha=0.683,
                    pdf_bins=pdf_bins)
                results[kdtype][i] = mode
                results[kdtype+"_kde"][i] = kde
                results[kdtype+"_err_neg"][i] = mode-lower
                results[kdtype+"_err_pos"][i] = upper-mode
    #
    # Plot PDFs and results
    #
    if plot_pdf:
        for i in np.ndindex(glong.shape):
            #
            # Set-up figure
            #
            fig, axes = plt.subplots(2, figsize=(8.5, 5.5))
            axes[0].set_title(
                r"PDFs for ($\ell$, $\pi$) = ("
                "{0:.1f}".format(glong[i])+r"$^\circ$, "
                "{0:.3f}".format(plx[i])+r" mas)")
            #
            # Compute "traditional" parallax distances
            #
            plx_d = parallax(glong[i], plx[i], dist_max=dist_max, R0=R0)
            kdtypes = ["Rgal", "distance"]
            labels = [r"$R$ (kpc)", r"$d$ (kpc)"]
            for ax, kdtype, label in zip(axes, kdtypes, labels):
                # find bad data
                if glong.size == 1:
                    out = plx_out[kdtype]
                else:
                    out = plx_out[kdtype][i]
                out = out[~np.isnan(out)]
                # skip if kde failed (all data is bad)
                if results[kdtype+"_kde"][i] is None:
                    continue
                # set-up bins
                binwidth = (np.max(out)-np.min(out))/20.
                bins = np.arange(
                    np.min(out), np.max(out)+binwidth, binwidth)
                distwidth = (np.max(out)-np.min(out))/200.
                dists = np.arange(
                    np.min(out), np.max(out)+distwidth, distwidth)
                pdf = results[kdtype+"_kde"][i](dists)
                ax.hist(
                    out, bins=bins, density=True, facecolor='white',
                    edgecolor='black', lw=2, zorder=1)
                ax.plot(dists, pdf, 'k-', zorder=3)
                err_dists = np.arange(
                    results[kdtype][i]-results[kdtype+"_err_neg"][i],
                    results[kdtype][i]+results[kdtype+"_err_pos"][i],
                    distwidth)
                err_pdf = results[kdtype+"_kde"][i](err_dists)
                ax.fill_between(
                    err_dists, 0, err_pdf, color='gray', alpha=0.5,
                    zorder=2)
                ax.axvline(
                    results[kdtype][i], linestyle='solid', color='k',
                    zorder=3)
                ax.axvline(
                    plx_d[kdtype], linestyle='dashed', color='k',
                    zorder=3)
                ax.set_xlabel(label)
                ax.set_ylabel("Normalized PDF")
                ax.set_xlim(np.min(out), np.max(out))
                # turn off grid
                ax.grid(False)
            plt.tight_layout()
            fname = "{0}{1}_{2}.pdf".format(
                plot_prefix, glong[i], plx[i])
            plt.savefig(fname)
            plt.close(fig)
    return results
