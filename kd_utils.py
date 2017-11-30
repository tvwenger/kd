#!/usr/bin/env python
"""
kd_utils.py

Utility functions for rotcurve_kd.py, pdf_kd.py, and rotation curves.

2017-04-12 Trey V. Wenger
"""

import time
import numpy as np

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

def pool_wait(result,num_items,chunksize):
    """
    Wait for a multiprocessing pool to finish. Print out status
    updates along the way.

    Parameters:
      result : map_async object
               The object returned from multiprocessing.map_async
      num_items : integer
                  total number of items in the pool
      chunksize : integer
                  size of chunks sent to each processor

    Returns: 
    """
    start_time = time.time()
    # figure out how many actual CPU calls there will be
    cpu_calls = int(num_items/chunksize) + (num_items%chunksize)
    strf = ("[{0:20s}] {1:.2f}% Done: {2} Left: {3} Time: {4:02}h "
            "{5:02}m {6:02}s")
    while not result.ready():
        remaining_cpu_calls = result._number_left
        finished_cpu_calls = cpu_calls - remaining_cpu_calls
        # Estimate remaining runtime
        if finished_cpu_calls > 0:
            time_now = time.time()
            time_per = (time_now-start_time)/finished_cpu_calls
            time_left = remaining_cpu_calls*time_per
            time_h = int(time_left/3600.)
            time_m = int((time_left-3600.*time_h)/60.)
            time_s = int(time_left-3600.*time_h-60.*time_m)
        else:
            time_h = -1
            time_m = -1
            time_s = -1
        print(strf.format('#'*int(20*finished_cpu_calls/cpu_calls),
                          100*finished_cpu_calls/cpu_calls,
                          finished_cpu_calls,remaining_cpu_calls,
                          time_h,time_m,time_s),end='\r')
        time.sleep(1)
    end_time = time.time()
    print(strf.format('#'*(20),100,cpu_calls,0,0,0,0),end='\r')
    print()
    # Compute total runtime
    run_time = end_time-start_time
    time_h = int(run_time/3600.)
    time_m = int((run_time-3600.*time_h)/60.)
    time_s = int(run_time-3600.*time_h-60.*time_m)    
    print("Runtime: {0:02}h {1:02}m {2:02}s".\
          format(time_h,time_m,time_s))
