#!/usr/bin/env python
"""
pdf_kd_lvmap_mpi.py

The same as pdf_kd_lvmap.py but set up for high-performace computing
via MPI.

12 April 2017 Trey V. Wenger
"""

import os
import time

import numpy as np
import sqlite3

from mpi4py import MPI

from kd.pdf_kd import pdf_kd

def enum(*sequential, **named):
    """
    Fake enumerated type. From StackOverflow:
    questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

def mpi_master(mpidata,dbfilename,
               glong_min=-180,glong_max=180.,glong_res=5.,
               velo_min=-120.,velo_max=120.,velo_res=5.):
    """
    This is executed by the master process. Sends out workers,
    checks worker status, and saves results.

    Parameters :
      mpidata : tuple
                contains the MPI tags, comm, size, rank, and status
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

    Returns:
    """
    tags,comm,size,rank,status = mpidata
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
                        "tangent_err_pos real, "
                        "vlsr_tangent real, vlsr_tangent_err_neg real, "
                        "vlsr_tangent_err_pos real")
        cur.execute("CREATE TABLE lvmap({0})".format(table_format))
        #
        # Set-up for MPI
        #
        num_workers = size-1
        closed_workers = 0
        current_task = 0
        start = time.time()
        while closed_workers < num_workers:
            #
            # Get status of MPI worker
            #
            data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                             status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            #
            # If the worker is ready, give it some work to do
            #
            if tag == tags.READY:
                #
                # Still some work to do, send a task
                #
                if current_task < glongs_space.size:
                    comm.send((current_task,
                               glongs_space[current_task],
                               velos_space[current_task]),
                               dest=source, tag=tags.START)
                    print("Started task {0}/{1}".\
                          format(current_task,glongs_space.size-1))
                    current_task += 1                
                #
                # All tasks are finished, kill worker
                #
                else:
                    comm.send(None, dest=source, tag=tags.EXIT)
            #
            # If the worker is done, collect the results and save them
            #
            if tag == tags.DONE:
                finished_task, result = data
                glong = glongs_space[finished_task]
                velo = velos_space[finished_task]
                values = (finished_task,glong,velo,
                          result["Rgal"],result["Rgal_err_neg"],
                          result["Rgal_err_pos"],
                          result["Rtan"],result["Rtan_err_neg"],
                          result["Rtan_err_pos"],
                          result["near"],result["near_err_neg"],
                          result["near_err_pos"],
                          result["far"],result["far_err_neg"],
                          result["far_err_pos"],
                          result["tangent"],result["tangent_err_neg"],
                          result["tangent_err_pos"],
                          result["vlsr_tangent"],result["vlsr_tangent_err_neg"],
                          result["vlsr_tangent_err_pos"])
                cur.execute("INSERT INTO lvmap VALUES"
                            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            values)
                print("Finshed task {0}/{1}".\
                    format(finished_task,glongs_space.size-1))
            #
            # If the worker is dead, increment the count
            #
            if tag == tags.EXIT:
                closed_workers += 1
    #
    # All workers have finished, save the results
    #
    end = time.time()
    run_time = end-start
    time_h = int(run_time/3600.)
    time_m = int((run_time-3600.*time_h)/60.)
    time_s = int(run_time-3600.*time_h-60.*time_m)    
    print("Total Runtime: {0:02}h {1:02}m {2:02}s".\
          format(time_h,time_m,time_s))  

def mpi_worker(mpidata,
               velo_res=5.,velo_stream=0.,velo_err=0.,
               rotcurve='reid14_rotcurve',
               rotcurve_dist_res=1.e-3,rotcurve_dist_max=30.,
               pdf_bins=100,
               num_samples=1000):
    """
    This is executed by theworker processes. Request job from
    master, execute job, return results.

    Parameters :
      mpidata : tuple
                contains the MPI tags, comm, size, rank, and status
      velo_res : scalar (optional)
                 LSR velocity resolution (km/s)
      velo_stream : scalar (optional)
                 LSR velocity streaming uncertainty (km/s).
      velo_err : scalar (optional)
                 LSR velocity additional uncertainty (km/s).
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

    Returns:
    """
    tags, comm, size, rank, status = mpidata
    #
    # Total velocity uncertainty combines velocity bin size,
    # streaming motion, and additional error in quadrature
    #
    velo_err_tot = np.sqrt(velo_res**2.+velo_stream**2.+velo_err**2.)
    #
    # Live until we're told to die
    #
    while True:
        #
        # Tell master we are ready
        #
        comm.send(None,dest=0,tag=tags.READY)
        #
        # Get task
        #
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        #
        # Do the work if we are given a task
        #
        if tag == tags.START:
            current_task, glong, velo = data
            output = pdf_kd(glong,velo,
                            velo_err=velo_err_tot,
                            rotcurve=rotcurve,
                            rotcurve_dist_res=rotcurve_dist_res,
                            rotcurve_dist_max=rotcurve_dist_max,
                            pdf_bins=pdf_bins,
                            num_samples=num_samples,num_cpu=0,
                            verbose=False)
            comm.send((current_task,output),dest=0,tag=tags.DONE)
        #
        # If we're told to die, let's die
        #
        elif tag == tags.EXIT:
            break
    #
    # Send death notice
    #
    comm.send(None,dest=0,tag=tags.EXIT)

def pdf_kd_lvmap_mpi(dbfilename,
                     glong_min=-180,glong_max=180.,glong_res=5.,
                     velo_min=-120.,velo_max=120.,velo_res=5.,
                     velo_stream=0.,velo_err=0.,
                     rotcurve='reid14_rotcurve',
                     rotcurve_dist_res=1.e-2,rotcurve_dist_max=30.,
                     pdf_bins=100,
                     num_samples=1000):
    """
    Compute PDF kinematic distance and distance uncertainties
    for a range of Galactic longitudes and velocities using MPI.
    Save results to a sqlite3 database.

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

    Returns:
    """
    #
    # Initialize MPI
    #
    tags = enum('READY', 'DONE', 'EXIT', 'START') 
    comm = MPI.COMM_WORLD   # MPI communicator object
    size = comm.size        # total number of workers + 1 master
    rank = comm.rank        # rank of this process (0 for master)
    status = MPI.Status()   # get MPI status object
    #
    # Run master (rank = 0) or worker (rank > 0) functions
    #
    if rank == 0:
        mpi_master((tags,comm,size,rank,status),dbfilename,
                   glong_min=glong_min,glong_max=glong_max,
                   glong_res=glong_res,
                   velo_min=velo_min,velo_max=velo_max,
                   velo_res=velo_res)
    else:
        mpi_worker((tags,comm,size,rank,status),
                   velo_res=velo_res,velo_stream=velo_stream,
                   velo_err=velo_err,
                   rotcurve=rotcurve,
                   rotcurve_dist_res=rotcurve_dist_res,
                   rotcurve_dist_max=rotcurve_dist_max,
                   num_samples=num_samples)
