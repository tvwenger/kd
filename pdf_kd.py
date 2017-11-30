#!/usr/bin/env python
"""
pdf_kd.py

Utilities to calculate kinematic distances and kinematic distance
uncertainties using the new probability distribution function (PDF)
method.

2017-04-12 Trey V. Wenger
"""

import time
import importlib
import multiprocessing as mp

import numpy as np
from scipy.stats.kde import gaussian_kde
from functools import partial

import matplotlib.pyplot as plt

from scipy import integrate
from pyqt_fit import kde as pyqt_kde
from pyqt_fit import kde_methods

from kd import kd_utils
from kd.rotcurve_kd import rotcurve_kd

def _rotcurve_kd_worker(num_samples,glong=None,velo=None,
                        velo_err=None,rotcurve='reid14_rotcurve',
                        rotcurve_dist_res=1.e-2,
                        rotcurve_dist_max=30.):
    """
    Multiprocessing worker for pdf_kd. Resamples velocity and
    runs rotcurve_kd.
    
    Parameters:
      num_samples : integer
                    number of samples this worker should run
      glong : scalar or 1-D array 
              Galactic longitude (deg). If it is an array, it must
              have the same size as velo.
      velo : scalar or 1-D array
             LSR velocity (km/s). If it is an array, it must
             have the same size as glong.
      velo_err : scalar or 1-D array (optional)
                 LSR velocity uncertainty (km/s). If it is an array,
                 it must have the same size as velo.
                 Otherwise, this uncertainty is applied to all velos.
      rotcurve : string (optional)
                 rotation curve model
      rotcurve_dist_res : scalar (optional)
                          line-of-sight distance resolution when
                          calculating kinematic distance with
                          rotcurve_kd (kpc)
      rotcurve_dist_max : scalar (optional)
                          maximum line-of-sight distance when
                          calculating kinematic distance with
                          rotcurve_kd (kpc)
    
    Returns: output
      (same as rotcurve_kd)

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
        raise ValueError("glong and velo must have same size")
    #
    # Re-sample velocities
    #
    if velo_err is not None:
      velo_resample = \
        np.random.normal(loc=velo_inp,scale=velo_err,
                         size=(num_samples,velo_inp.size)).T
    else:
        velo_resample = np.ones((num_samples,velo_inp.size))
        velo_resample = (velo_resample*velo_inp).T
    #
    # Calculate kinematic distance for each l,v point
    #
    kd_out = [rotcurve_kd(np.ones(num_samples)*l,v,
                          rotcurve=rotcurve,
                          dist_res=rotcurve_dist_res,
                          dist_max=rotcurve_dist_max,
                          resample=True)
              for (l,v) in zip(glong_inp,velo_resample)]
    return kd_out

def _pdf_kd_results_worker(inputs,pdf_bins=100):
    """
    Multiprocessing worker for pdf_kd. Finds the kinematic distance
    and distance uncertainty from the output of many samples from
    rotcurve_kd. See pdf_kd for more details.

    Parameters:
      input : (kd_samples, kd_out_ind, kdtype, kdetype)
        kd_samples : 1-D array
                     This array contains the output from rotcurve_kd
                     for a kinematic distance (kpc) of type kdtype for
                     many samples (i.e. it is the "Rgal" array from
                     rotcurve_kd output)
        kd_out_ind : integer
                     The index of the rotcurve_kd output corresponding
                     to the data in kd_samples.
        kdtype : string
                 The type of kinematic distance in kd_samples
        kdetype : string
                  which KDE method to use
                  'pyqt' uses pyqt_fit with linear combination
                         and boundary at 0
                  'scipy' uses gaussian_kde with no boundary
      pdf_bins : integer (optional)
                 number of bins used in calculating PDF


    Returns: (kd_out_ind, kdtype, kde, peak_dist, peak_dist_err_neg,
              peak_dist_err_pos)
      kd_out_ind : integer
                   Same as input
      kdtype : string
               Same as input
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
    # Unpack inputs
    #
    kd_samples, kd_out_ind, kdtype, kdetype = inputs
    #
    # Compute kernel density estimator and PDF
    #
    nans = np.isnan(kd_samples)
    if np.all(nans):
        # skip if all nans
        return (kd_out_ind, kdtype, None, np.nan, np.nan, np.nan)
    try:
        if kdetype == 'scipy':
            kde = gaussian_kde(kd_samples[~nans])
        elif kdetype == 'pyqt':
            kde = pyqt_kde.KDE1D(kd_samples[~nans], lower=0,
                                 method=kde_methods.linear_combination)
        else:
            print("INVALIDE KDE METHOD: {0}".format(kdetype))
            return (kd_out_ind, kdtype, None, np.nan, np.nan, np.nan)
    except np.linalg.LinAlgError:
        # catch singular matricies (i.e. all values are the same)
        return (kd_out_ind, kdtype, None, np.nan, np.nan, np.nan)
    dists = np.linspace(np.nanmin(kd_samples),np.nanmax(kd_samples),
                        pdf_bins)
    pdf = kde(dists)
    #
    # Find index, value, and distance of peak of PDF
    #
    peak_ind = np.argmax(pdf)
    peak_value = pdf[peak_ind]
    peak_dist = dists[peak_ind]
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
        return (kd_out_ind, kdtype, None, np.nan, np.nan, np.nan)
    #
    # Return results
    #
    return(kd_out_ind, kdtype, kde, peak_dist, peak_dist_err_neg,
           peak_dist_err_pos)

def pdf_kd(glong,velo,velo_err=None,
           rotcurve='reid14_rotcurve',
           rotcurve_dist_res=1.e-3,
           rotcurve_dist_max=30.,
           pdf_bins=100,
           num_samples=1000,
           num_cpu=mp.cpu_count(),chunksize=10,
           plot_pdf=False,plot_prefix='pdf_',verbose=True):
    """
    Return the kinematic near, far, and tanget distance and distance
    uncertainties for a  given Galactic longitude and LSR velocity
    assuming a given rotation curve. Generate PDF of distances by
    resampling within rotation curve parameter and velocity
    uncertainties. Peak of PDF is the returned distance and width of
    PDF such that the area enclosed by the PDF is 68.2% is the
    returned distance uncertainty.

    Parameters:
      glong : scalar or 1-D array
              Galactic longitude (deg). If it is an array, it must
              have the same size as velo.
      velo : scalar or 1-D array
             LSR velocity (km/s). If it is an array, it must
             have the same size as glong.
      velo_err : scalar or 1-D (optional)
                 LSR velocity uncertainty (km/s). If it is an array,
                 it must have the same size as velo.
                 Otherwise, this uncertainty is applied to all velos.
      rotcurve : string (optional)
                 rotation curve model
      rotcurve_dist_res : scalar (optional)
                          line-of-sight distance resolution when
                          calculating kinematic distance with
                          rotcurve_kd (kpc)
      rotcurve_dist_max : scalar (optional)
                          maximum line-of-sight distance when
                          calculating kinematic distance with
                          rotcurve_kd (kpc)
      pdf_bins : integer (optional)
                 number of bins used to calculate PDF
      num_samples : integer (optional)
                    Number of MC samples to use when generating PDF
      num_cpu : integer (optional)
                Number of CPUs to use in multiprocessing.
                If 0, do not use multiprocessing.
      chunksize : integer (optional)
                  Number of tasks per CPU in multiprocessing.
      plot_pdf : bool (optional)
                 If True, plot each PDF. Filenames are
                 plot_prefix+"{0}glong_{1}velo.pdf".
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
      output["Rtan"] : scalar or 1-D array
                       Galactocentric radius of tangent point (kpc).
      output["Rtan_err_neg"] : scalar or 1-D array
                               Galactocentric radius of tangent point
                               uncertainty in the negative direction
                               (kpc).
      output["Rtan_err_pos"] : scalar or 1-D array
                               Galactocentric radius of tangent point
                               uncertainty in the positive direction
                               (kpc).
      output["near"] : scalar or 1-D array
                       kinematic near distance (kpc)
      output["near_err_neg"] : scalar or 1-D array
                               kinematic near distance uncertainty
                               in the negative direction (kpc)
      output["near_err_pos"] : scalar or 1-D array
                               kinematic near distance uncertainty
                               in the positive direction (kpc)
      output["far"] : scalar or 1-D array
                      kinematic far distance (kpc)
      output["far_err_neg"] : scalar or 1-D array
                              kinematic far distance uncertainty in
                              the negative direction (kpc)
      output["far_err_pos"] : scalar or 1-D array
                              kinematic far distance uncertainty in
                              the positive direction (kpc)
      output["tangent"] : scalar or 1-D array
                          kinematic tangent distance (kpc)
      output["tangent_err_neg"] : scalar or 1-D array
                                  kinematic tangent distance
                                  uncertainty in the negative
                                  direction (kpc)
      output["tangent_err_pos"] : scalar or 1-D array
                                  kinematic tangent distance
                                  uncertainty in the positive
                                  direction (kpc)
      output["vlsr_tangent"] : scalar or 1-D array
                               LSR velocity of tangent point (km/s)
      output["vlsr_tangent_err_neg"] : scalar or 1-D array
                                       LSR velocity of tangent
                                       uncertainty in the negative
                                       direction (km/s)
      output["vlsr_tangent_err_pos"] : scalar or 1-D array
                                       LSR velocity of tangent
                                       uncertainty in the positive
                                       direction (km/s)
      If glong and velo are scalars, each of these is a scalar.
      Otherwise they have shape (velo.size).

    Raises:
      ValueError : if glong and velo are not 1-D; or
                   if glong and velo are arrays and not the same size
    """
    total_start = time.time()
    #
    # check inputs
    #
    # convert scalar to array if necessary
    glong_inp, velo_inp = np.atleast_1d(glong, velo)
    # check shape of inputs
    if glong_inp.ndim != 1 or velo_inp.ndim != 1:
        raise ValueError("glong and velo must be 1-D")
    if glong_inp.size != velo_inp.size:
        raise ValueError("glong and velo must have same size")
    #
    # Set-up multiprocesing for rotcurve re-sampling
    #
    num_chunks = int(num_samples/chunksize)
    jobs = [chunksize for c in range(num_chunks)]
    worker = partial(_rotcurve_kd_worker,
                     glong=glong_inp,velo=velo_inp,
                     velo_err=velo_err,rotcurve=rotcurve,
                     rotcurve_dist_res=rotcurve_dist_res,
                     rotcurve_dist_max=rotcurve_dist_max)
    if num_samples % chunksize > 0:
        jobs = jobs + [num_samples % chunksize]
    if num_cpu > 0:
        #
        # Calculate rotcurve kinematic distance for each l,v point in
        # parallel
        #
        if verbose:
            print("Starting multiprocessing for rotation curve "
                  "re-sampling...")
        pool = mp.Pool(processes=num_cpu)
        result = pool.map_async(worker,jobs,chunksize=1)
        if verbose:
            kd_utils.pool_wait(result,len(jobs),1)
        else:
            while not result.ready():
                time.sleep(1)
        pool.close()
        pool.join()
        result = result.get()
    else:
        #
        # Calculate rotcurve kinematic distance in series
        #
        result = [worker(job) for job in jobs]
    #
    # Concatenate results from multiprocessing
    #
    kd_out = [{"Rgal":np.array([]),"Rtan":np.array([]),
               "near":np.array([]),"far":np.array([]),
               "tangent":np.array([]),"vlsr_tangent":np.array([])}
               for i in range(glong_inp.size)]
    for r in result:
        for i in range(glong_inp.size):
            for kdtype in ("Rgal","Rtan","near","far","tangent",
                           "vlsr_tangent"):
                kd_out[i][kdtype] = \
                  np.append(kd_out[i][kdtype],r[i][kdtype])
    #
    # Storage for final PDF kinematic distance results
    #
    results = {"Rgal": np.zeros(glong_inp.size),
               "Rgal_kde": np.empty(shape=(glong_inp.size,),
                                    dtype=object),
               "Rgal_err_neg": np.zeros(glong_inp.size),
               "Rgal_err_pos": np.zeros(glong_inp.size),
               "Rtan": np.zeros(glong_inp.size),
               "Rtan_kde": np.empty(shape=(glong_inp.size,),
                                    dtype=object),
               "Rtan_err_neg": np.zeros(glong_inp.size),
               "Rtan_err_pos": np.zeros(glong_inp.size),
               "near": np.zeros(glong_inp.size),
               "near_kde": np.empty(shape=(glong_inp.size,),
                                    dtype=object),
               "near_err_neg": np.zeros(glong_inp.size),
               "near_err_pos": np.zeros(glong_inp.size),
               "far": np.zeros(glong_inp.size),
               "far_kde": np.empty(shape=(glong_inp.size,),
                                   dtype=object),
               "far_err_neg": np.zeros(glong_inp.size),
               "far_err_pos": np.zeros(glong_inp.size),
               "tangent": np.zeros(glong_inp.size),
               "tangent_kde": np.empty(shape=(glong_inp.size,),
                                       dtype=object),
               "tangent_err_neg": np.zeros(glong_inp.size),
               "tangent_err_pos": np.zeros(glong_inp.size),
               "vlsr_tangent": np.zeros(glong_inp.size),
               "vlsr_tangent_kde": np.empty(shape=(glong_inp.size,),
                                       dtype=object),
               "vlsr_tangent_err_neg": np.zeros(glong_inp.size),
               "vlsr_tangent_err_pos": np.zeros(glong_inp.size)}
    #
    # Set up multiprocessing for PDF kinematic distance calculation
    #
    jobs = []
    for i,out in enumerate(kd_out):
        for kdtype,kdetype in \
            zip(["Rgal","Rtan","near","far","tangent","vlsr_tangent"],
                ["pyqt","pyqt","pyqt","pyqt","pyqt","scipy"]):
            jobs.append((out[kdtype],i,kdtype,kdetype))
    worker = partial(_pdf_kd_results_worker,
                     pdf_bins=pdf_bins)
    if num_cpu > 0:
        #
        # Calculate PDF kinematic distance for each l,v point in
        # parallel
        #
        if verbose:
            print("Starting multiprocessing for PDF kinematic "
                  "distance calculation...")
        pool = mp.Pool(processes=num_cpu)
        result = pool.map_async(worker,jobs,chunksize=1)
        if verbose:
            kd_utils.pool_wait(result,len(jobs),1)
        else:
            while not result.ready():
                time.sleep(1)
        pool.close()
        pool.join()
        result = result.get()
    else:
        #
        # Calculate PDF kinematic distance in series
        #
        result = [worker(job) for job in jobs]
    #
    # Unpack results and save
    #
    for r in result:
        kd_out_ind, kdtype, kde, peak_dist, peak_dist_err_neg, \
          peak_dist_err_pos = r
        results[kdtype][kd_out_ind] = peak_dist
        results[kdtype+"_kde"][kd_out_ind] = kde
        results[kdtype+"_err_neg"][kd_out_ind] = peak_dist_err_neg
        results[kdtype+"_err_pos"][kd_out_ind] = peak_dist_err_pos
    #
    # Plot PDFs and results
    #
    if plot_pdf:
        #
        # Loop over l,v
        #
        for i,(l,v) in enumerate(zip(glong_inp,velo_inp)):
            #
            # Set-up figure
            #
            fig, (ax1, ax2, ax3, ax4, ax5) = \
              plt.subplots(5, figsize=(8.5,11))
            ax1.set_title(r"PDFs for ($\ell$, $v$) = ("
                          "{0:.1f}".format(l)+r"$^\circ$, "
                          "{0:.1f}".format(v)+r"km s$^{-1}$)")
            #
            # Compute "traditional" kinematic distances
            #
            rot_kd = rotcurve_kd(l,v,rotcurve=rotcurve,
                                 dist_res=rotcurve_dist_res,
                                 dist_max=rotcurve_dist_max)
            kdtypes = ["Rgal","Rtan","near","far","tangent"]
            labels = [r"$R$ (kpc)",r"$R_{\rm tan}$ (kpc)",
                      r"$d_{\rm near}$ (kpc)",r"$d_{\rm far}$ (kpc)",
                      r"$d_{\rm tan}$ (kpc)"]
            for ax,kdtype,label in zip([ax1,ax2,ax3,ax4,ax5],
                                       kdtypes, labels):
                # find bad data
                out = kd_out[i][kdtype]
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
                ax.axvline(rot_kd[kdtype],linestyle='dashed',
                           color='k',zorder=3)
                ax.set_xlabel(label)
                ax.set_ylabel("Normalized PDF")
                ax.set_xlim(np.min(out),
                            np.max(out))
                # turn off grid
                ax.grid(False)
            plt.tight_layout()
            plt.savefig(plot_prefix+"{0}glong_{1}velo.pdf".format(l,v))
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
