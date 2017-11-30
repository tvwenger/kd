#!/usr/bin/env python
"""
pdf_kd_lvmap.py

Utilities to calculate kinematic distances and kinematic distance
uncertainties using the new probability distribution function (PDF)
method for a range of Galactic longitudes and LSR velocities.

2017-04-12 Trey V. Wenger
"""

import os

import numpy as np
import multiprocessing as mp
import sqlite3

from kd.pdf_kd import pdf_kd

def pdf_kd_lvmap(dbfilename,
                 glong_min=-180,glong_max=180.,glong_res=5.,
                 velo_min=-120.,velo_max=120.,velo_res=5.,
                 velo_stream=0.,velo_err=0.,
                 rotcurve='reid14_rotcurve',
                 dist_res=1.e-2,dist_max=30.,
                 num_samples=1000,num_cpu=mp.cpu_count(),
                 chunksize=10):
    """
    Compute PDF kinematic distance and distance uncertainties
    for a range of Galactic longitudes and velocities. Save results
    to a sqlite3 database.

    Parameters :
      dbfilename : string
                   Location where DB is written. Must not already
                   exist.
      glong_min : scalar (optional)
                  minimum Galactic longitude (deg)
      glong_max : scalar (optional)
                  maximum Galactic longitude (deg)
      glong_res : scalar (optional)
                  Galactic longitude resolution (deg)
      velo_min : scalar (optional)
                 minimum LSR velocity (km/s)
      velo_max : scalar (optional)
                 maximum LSR velocity (km/s)
      velo_res : scalar (optional)
                 LSR velocity resolution (km/s)
      velo_stream : scalar (optional)
                 LSR velocity streaming uncertainty (km/s).
      velo_err : scalar (optional)
                 LSR velocity additional uncertainty (km/s).
      rotcurve : string (optional)
                 rotation curve model
      dist_res : scalar (optional)
                 line-of-sight distance resolution when calculating
                 kinematic distance (kpc)
      dist_max : scalar (optional)
                 maximum line-of-sight distance when calculating
                 kinematic distance (kpc)
      num_samples : integer (optional)
                    Number of MC samples to use when generating PDF
      num_cpu : integer (optional)
                Number of CPUs to use in multiprocessing.
                If 0, do not use multiprocessing.
      chunksize : integer (optional)
                  Number of tasks per CPU in multiprocessing.

    Returns:
    """
    #
    # Check that we're not over-writing a DB
    #
    if os.path.exists(dbfilename):
        raise IOError("Will not overwrite {0}".format(dbfilename))
    #
    # Define longitude and velocity arrays
    #
    glong_num = int((glong_max - glong_min)/glong_res) + 1
    velo_num = int((velo_max - velo_min)/velo_res) + 1
    glongs = np.linspace(glong_min,glong_max,glong_num)
    velos = np.linspace(velo_min,velo_max,velo_num)
    #
    # Get all glong, velo pairs
    #
    lv_space = np.meshgrid(velos,glongs)
    glongs_space = np.ravel(lv_space[1])
    velos_space = np.ravel(lv_space[0])
    #
    # Total velocity uncertainty combines velocity bin size,
    # streaming motion, and additional error in quadrature
    #
    velo_err_tot = np.sqrt(velo_res**2.+velo_stream**2.+velo_err**2.)
    #
    # Compute PDF kinematic distances
    #
    output = pdf_kd(glongs_space,velos_space,
                    velo_err=velo_err_tot,
                    rotcurve=rotcurve,
                    dist_res=dist_res,dist_max=dist_max,
                    num_samples=num_samples,
                    num_cpu=num_cpu,chunksize=chunksize)
    #
    # Connect to database
    #
    with sqlite3.connect(dbfilename) as con:
        cur = con.cursor()
        #
        # Create database table
        #
        table_format = ("id int, glong real, velo real, "
                        "Rgal real, Rgal_err_neg real, "
                        "Rgal_err_pos real, "
                        "Rtan real, Rtan_err_neg real, "
                        "Rtan_err_pos real, "
                        "near real, near_err_neg real, "
                        "near_err_pos real, "
                        "far real, far_err_neg real, "
                        "far_err_pos real, "
                        "tangent real, tangent_err_neg real, "
                        "tangent_err_pos real")
        cur.execute("CREATE TABLE lvmap({0})".format(table_format))
        #
        # Save results
        # 
        for i,(l,v) in enumerate(zip(glongs_space,velos_space)):
            values = (i,l,v,
                      output["Rgal"][i],output["Rgal_err_neg"][i],
                      output["Rgal_err_pos"][i],
                      output["Rtan"][i],output["Rtan_err_neg"][i],
                      output["Rtan_err_pos"][i],
                      output["near"][i],output["near_err_neg"][i],
                      output["near_err_pos"][i],
                      output["far"][i],output["far_err_neg"][i],
                      output["far_err_pos"][i],
                      output["tangent"][i],
                      output["tangent_err_neg"][i],
                      output["tangent_err_pos"][i])
            cur.execute("INSERT INTO lvmap VALUES"
                        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        values)
