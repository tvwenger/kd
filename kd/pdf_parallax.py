#!/usr/bin/env python
"""
pdf_parallax.py

Utility to calculate PDF parallax distances.

2019-01-19 Trey V. Wenger
"""

import time

import numpy as np
from scipy.stats.kde import gaussian_kde

import matplotlib.pyplot as plt

from scipy import integrate
from pyqt_fit import kde as pyqt_kde
from pyqt_fit import kde_methods

from kd.parallax import parallax

def parallax_worker(num_samples, glong, plx, plx_err=None,
                    dist_max=30., R0=8.34):
    """
    Resamples parallax and computes distance and Galactocentric
    radius.
    
    Parameters:
      num_samples : integer
                    number of samples
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as plx.
      plx : scalar or 1-D array
            Parallax (milli-arcseconds). If it is an array, it must
            have the same size as glong.
      plx_err : scalar or 1-D
                Parallax uncertainty (milli-arcseconds). If it is
                an array, it must have the same size as plx.
      dist_max : scalar (optional)
                 The maximum parallax distance to compute (kpc)
      R0 : scalar (optional)
           Solar Galactocentric radius (kpc)
    
    Returns: output
      output : list of parallax output for each (glong, plx).
               For example, output[0]["distance"] contains the 1-D
               array of parallax distances calculated for the first
               (glong, plx) point and each sample.

    Raises:
      ValueError : if glong, plx, and plx_err are not 1-D; or
                   if glong, plx, and plx_err are arrays and not the 
                   same size
    """
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, plx_inp, plx_err_inp = np.atleast_1d(glong, plx, plx_err)
    # check shape of inputs
    if glong_inp.ndim != 1 or plx_inp.ndim != 1 or plx_err_inp.ndim != 1:
        raise ValueError("glong, plx, and plx_err must be 1-D")
    if glong_inp.size != plx_inp.size or glong_inp.size != plx_err_inp.size:
        raise ValueError("glong, plx, and plx_err must have same size")
    #
    # Resample parallaxes
    #
    plx_samples = np.random.normal(loc=plx_inp,scale=plx_err_inp,
                                   size=(num_samples,plx_inp.size)).T
    # remove negative parallaxes
    plx_samples[plx_samples < 0.] = np.nan
    #
    # Compute parallax distances
    #
    plx_out = [parallax(np.ones(num_samples)*l, p,
                        dist_max=dist_max, R0=R0)
               for (l,p) in zip(glong_inp, plx_samples)]
    return plx_out

def pdf_parallax_results_worker(plx_samples, kdetype, pdf_bins=100):
    """
    Finds the parallax distance and distance uncertainty from the 
    output of many samples from parallax. See pdf_parallax for more 
    details.

    Parameters:
      plx_samples : 1-D array
                    This array contains the output from parallax
                    for a parallax distance (kpc) for
                    many samples (i.e. it is the "Rgal" array from
                    parallax output)
      kdetype : string
                which KDE method to use
                'pyqt' uses pyqt_fit with linear combination
                   and boundary at 0
                'scipy' uses gaussian_kde with no boundary
      pdf_bins : integer (optional)
                 number of bins used in calculating PDF

    Returns: kde, peak_dist, peak_dist_err_neg, peak_dist_err_pos
      kde : scipy.gaussian_kde object
            The KDE calculated for this kinematic distance
      peak_dist : scalar
                  The distance associated with the peak of the PDF
      peak_dist_err_neg : scalar
                      The negative uncertainty of peak_dist
      peak_dist_err_pos : scalar
                      The positive uncertainty of peak_dist
    """
    #
    # Compute kernel density estimator and PDF
    #
    nans = np.isnan(plx_samples)
    if np.sum(~nans) < 2:
        # skip if fewer than two non-nans
        return (None, np.nan, np.nan, np.nan)
    try:
        if kdetype == 'scipy':
            kde = gaussian_kde(plx_samples[~nans])
        elif kdetype == 'pyqt':
            kde = pyqt_kde.KDE1D(plx_samples[~nans], lower=0,
                                 method=kde_methods.linear_combination)
        else:
            print("INVALIDE KDE METHOD: {0}".format(kdetype))
            return (None, np.nan, np.nan, np.nan)
    except np.linalg.LinAlgError:
        # catch singular matricies (i.e. all values are the same)
        return (None, np.nan, np.nan, np.nan)
    dists = np.linspace(np.nanmin(plx_samples),np.nanmax(plx_samples),
                        pdf_bins)
    pdf = kde(dists)
    #
    # Find index, value, and distance of peak of PDF
    #
    peak_ind = np.argmax(pdf)
    peak_value = pdf[peak_ind]
    peak_dist = dists[peak_ind]
    if np.isnan(peak_value):
        # too few good samples?
        return (None, np.nan, np.nan, np.nan)
    #
    # Walk down from peak of PDF until integral between two
    # bounds is 68.3% of the total integral (=1 because it's
    # normalized). Step size is 1% of peak value.
    #
    for target in np.arange(peak_value,0.,-0.01*peak_value):
        # find bounds
        if peak_ind == 0:
            lower = 0
        else:
            lower = np.argmin(np.abs(target-pdf[0:peak_ind]))
        if peak_ind == len(pdf)-1:
            upper = len(pdf)-1
        else:
            upper = np.argmin(np.abs(target-pdf[peak_ind:]))+peak_ind
        # integrate
        #integral = kde.integrate_box_1d(dists[lower],dists[upper])
        integral = integrate.quad(kde,dists[lower],dists[upper])[0]
        if integral > 0.683:
            peak_dist_err_neg = peak_dist-dists[lower]
            peak_dist_err_pos = dists[upper]-peak_dist
            break
    else:
        return (None, np.nan, np.nan, np.nan)
    #
    # Return results
    #
    return (kde, peak_dist, peak_dist_err_neg, peak_dist_err_pos)

def pdf_parallax(glong,plx,plx_err=None,
                 dist_max=30.,R0=8.34,
                 pdf_bins=100,num_samples=1000,
                 plot_pdf=False,plot_prefix='pdf_',verbose=True):
    """
    Return the parallax distance given a parallax and parallax
    uncertainty. Generate PDF of distance by
    resampling within uncertainty. Peak of PDF is the returned 
    distance and width of PDF such that the area enclosed by the 
    PDF is 68.2% is the returned distance uncertainty.

    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as plx.
      plx : scalar or 1-D array
            Parallax in milli-arcseconds
      plx_err : scalar or 1-D (optional)
                Parallax uncertainty in milli-arcseconds
      dist_max : scalar (optional)
                 The maximum parallax distance to compute (kpc)
      R0 : scalar (optional)
           Solar Galactocentric radius (kpc)
      pdf_bins : integer (optional)
                 number of bins used to calculate PDF
      num_samples : integer (optional)
                    Number of MC samples to use when generating PDF
      plot_pdf : bool (optional)
                 If True, plot each PDF. Filenames are
                 plot_prefix+"{0}plx_{1}err.pdf".
      plot_prefix : string (optional)
                    The prefix for the plot filenames.
      verbose : bool (optional)
                If True, output status updates and total runtime
    
    Returns: output
      output["Rgal"] : scalar or 1-D array
                       Galactocentric radius (kpc).
      output["Rgal_err_neg"] : scalar or 1-D array
                               Galactocentric radius uncertainty in
                               the negative direction (kpc).
      output["Rgal_err_pos"] : scalar or 1-D array
                               Galactocentric radius uncertainty in
                               the positive direction (kpc).
      output["Rgal_kde"] : PyQTFit Kenel Density Estimator
                           The KDE for Galactocentric radius.
      output["distance"] : scalar or 1-D array
                           parallax distance (kpc)
      output["distance_err_neg"] : scalar or 1-D array
                                   distance uncertainty
                                   in the negative direction (kpc)
      output["distance_err_pos"] : scalar or 1-D array
                                   distance uncertainty
                                   in the positive direction (kpc)
      output["distance_kde"] : PyQTFit Kenel Density Estimator
                               The KDE for distance.
      If glong and plx are scalars, each of these is a scalar.
      Otherwise they have shape (glong.size).

    Raises:
      ValueError : if glong and plx are not 1-D; or
                   if glong and plx are arrays and not the same size
    """
    total_start = time.time()
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, plx_inp = np.atleast_1d(glong, plx)
    # check shape of inputs
    if glong_inp.ndim != 1 or plx_inp.ndim != 1:
        raise ValueError("glong and plx_ must be 1-D")
    if glong_inp.size != plx_inp.size:
        raise ValueError("glong and plx must have same size")
    #
    # Storage for final PDF kinematic distance results
    #
    results = {"Rgal": np.zeros(plx_inp.size),
               "Rgal_kde": np.empty(shape=(plx_inp.size,),
                                    dtype=object),
               "Rgal_err_neg": np.zeros(plx_inp.size),
               "Rgal_err_pos": np.zeros(plx_inp.size),
               "distance": np.zeros(plx_inp.size),
               "distance_kde": np.empty(shape=(plx_inp.size,),
                                    dtype=object),
               "distance_err_neg": np.zeros(plx_inp.size),
               "distance_err_pos": np.zeros(plx_inp.size)}
    #
    # Calculate parallax distances
    #
    plx_out = parallax_worker(num_samples, glong_inp, plx_inp,
                              plx_err=plx_err, dist_max=dist_max,
                              R0=R0)
    #
    # Calculate PDF parallax distance
    #
    for plxtype, kdetype in zip(["Rgal","distance"],["pyqt","pyqt"]):
        for i,my_plx_out in enumerate(plx_out):
            my_pdfplx_out = pdf_parallax_results_worker(my_plx_out[plxtype], kdetype,
                                                        pdf_bins=pdf_bins)
            kde, peak_dist, peak_dist_err_neg, peak_dist_err_pos = \
                my_pdfplx_out
            results[plxtype][i] = peak_dist
            results[plxtype+"_kde"][i] = kde
            results[plxtype+"_err_neg"][i] = peak_dist_err_neg
            results[plxtype+"_err_pos"][i] = peak_dist_err_pos
    #
    # Plot PDFs and results
    #
    if plot_pdf:
        #
        # Loop over longitude, plx
        #
        for i,(l,p) in enumerate(zip(glong_inp,plx_inp)):
            #
            # Set-up figure
            #
            fig, (ax1, ax2) = \
              plt.subplots(2, figsize=(8.5,5.5))
            ax1.set_title(r"PDFs for ($\ell$, $\pi$) = ("
                          "{0:.1f}".format(l)+r"$^\circ$, "
                          "{0:.3f}".format(p)+r" mas)")
            #
            # Compute "traditional" parallax distances
            #
            plx_d = parallax(l,p,dist_max=dist_max,R0=R0)
            kdtypes = ["Rgal","distance"]
            labels = [r"$R$ (kpc)",r"$d$ (kpc)"]
            for ax,kdtype,label in zip([ax1,ax2],kdtypes, labels):
                # find bad data
                out = plx_out[i][kdtype]
                out = out[~np.isnan(out)]
                # skip if kde failed (all data is bad)
                if results[kdtype+"_kde"][i] is None:
                    continue
                # set-up bins
                binwidth = (np.max(out)-np.min(out))/20.
                bins = np.arange(np.min(out),
                                 np.max(out)+binwidth,
                                 binwidth)
                distwidth = (np.max(out)-np.min(out))/200.
                dists = np.arange(np.min(out),
                                  np.max(out)+distwidth,
                                  distwidth)
                pdf = results[kdtype+"_kde"][i](dists)
                ax.hist(out,bins=bins,normed=True,
                        facecolor='white',edgecolor='black',lw=2,
                        zorder=1)
                ax.plot(dists,pdf,'k-',zorder=3)
                err_dists = \
                  np.arange(results[kdtype][i]-results[kdtype+"_err_neg"][i],
                            results[kdtype][i]+results[kdtype+"_err_pos"][i],
                            distwidth)
                err_pdf = results[kdtype+"_kde"][i](err_dists)
                ax.fill_between(err_dists,0,err_pdf,color='gray',
                                alpha=0.5,zorder=2)
                ax.axvline(results[kdtype][i],linestyle='solid',
                           color='k',zorder=3)
                ax.axvline(plx_d[kdtype],linestyle='dashed',
                           color='k',zorder=3)
                ax.set_xlabel(label)
                ax.set_ylabel("Normalized PDF")
                ax.set_xlim(np.min(out),
                            np.max(out))
                # turn off grid
                ax.grid(False)
            plt.tight_layout()
            plt.savefig(plot_prefix+"{0}glong_{1}plx.pdf".format(l,p))
            plt.close(fig)
    #
    # Convert results to scalar if necessary
    #
    if glong_inp.size == 1:
        for key in results.keys():
            results[key] = results[key][0]
    total_end = time.time()
    if verbose:
        run_time = total_end-total_start
        time_h = int(run_time/3600.)
        time_m = int((run_time-3600.*time_h)/60.)
        time_s = int(run_time-3600.*time_h-60.*time_m)    
        print("Total Runtime: {0:02}h {1:02}m {2:02}s".\
              format(time_h,time_m,time_s))    
    return results
