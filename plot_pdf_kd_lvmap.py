#!/usr/bin/env python
"""
plot_pdf_kd_lvmap.py

Utilities to plot output from pdf_kd_lvmap.py and pdf_kd_lvmap_mpi.py.

2017-04-12 Trey V. Wenger
"""

import matplotlib.pyplot as plt

import numpy as np

import sqlite3

def plot_pdf_kd_lvmap(dbfilename,prefix="kd_lvmap",
                      vlim1=None,vlim2=None,vlim3=None,vlim4=None):
    """
    Plot the results from pdf_kd_lvmap and pdf_kd_lvmap_mpi. Plots
    1 by 4 panel of distance, + uncertainty, - uncertainty, and
    total uncertainty for Rgal, Rtan, near, far, and tangent
    distance.
    Plots have filenames like prefix+"_Rgal.pdf", etc.

    Parameters:
      dbfilename : string
                   Database file
      prefix : string (optional)
               prefix for generated plots
      vlim1 : tuple of scalars (optional)
      vlim2 : tuple of scalars (optional)
      vlim3 : tuple of scalars (optional)
      vlim4 : tuple of scalars (optional)
              if not None, it should have the form (vmin,vmax)
              where vmin and vmax are the color scale ranges for
              the first through fourth panels

    Returns:
    """
    if vlim1 is None:
        vlim1 = (None,None)
    if vlim2 is None:
        vlim2 = (None,None)
    if vlim3 is None:
        vlim3 = (None,None)
    if vlim4 is None:
        vlim4 = (None,None)
    #
    # Read DB, extract parts, and re-shape
    #
    with sqlite3.connect(dbfilename) as con:
        cur = con.cursor()
        #
        # Get longitudes and velocities
        #
        stmt = cur.execute("SELECT DISTINCT glong FROM lvmap "
                           "ORDER BY glong")
        glongs = np.array([s[0] for s in stmt.fetchall()])
        stmt = cur.execute("SELECT DISTINCT velo FROM lvmap "
                           "ORDER BY velo")
        velos = np.array([s[0] for s in stmt.fetchall()])
        #
        # Create storage for results
        #
        results = {"Rgal":np.zeros((glongs.size,velos.size)),
                   "Rgal_err_neg":np.zeros((glongs.size,velos.size)),
                   "Rgal_err_pos":np.zeros((glongs.size,velos.size)),
                   "Rtan":np.zeros((glongs.size,velos.size)),
                   "Rtan_err_neg":np.zeros((glongs.size,velos.size)),
                   "Rtan_err_pos":np.zeros((glongs.size,velos.size)),
                   "near":np.zeros((glongs.size,velos.size)),
                   "near_err_neg":np.zeros((glongs.size,velos.size)),
                   "near_err_pos":np.zeros((glongs.size,velos.size)),
                   "far":np.zeros((glongs.size,velos.size)),
                   "far_err_neg":np.zeros((glongs.size,velos.size)),
                   "far_err_pos":np.zeros((glongs.size,velos.size)),
                   "tangent":np.zeros((glongs.size,velos.size)),
                   "tangent_err_neg":np.zeros((glongs.size,
                                               velos.size)),
                   "tangent_err_pos":np.zeros((glongs.size,
                                               velos.size))}
        #
        # Get results
        #
        values = ("Rgal","Rgal_err_neg","Rgal_err_pos",
                  "Rtan","Rtan_err_neg","Rtan_err_pos",
                  "near","near_err_neg","near_err_pos",
                  "far","far_err_neg","far_err_pos",
                  "tangent","tangent_err_neg","tangent_err_pos")
        for l_ind,l in enumerate(glongs):
            for v_ind,v in enumerate(velos):
                stmt = cur.execute("SELECT "+",".join(values)+" FROM "
                                   "lvmap WHERE glong=? AND velo=?",
                                   (l,v))
                foo = stmt.fetchall()
                if len(foo) > 1:
                    raise ValueError("Found multiple entries for "
                                     "(l,v) = ({0},{1})".format(l,v))
                for val_ind,val in enumerate(values):
                    results[val][l_ind,v_ind] = foo[0][val_ind]
    #
    # Loop over distances and generate plots
    #
    kdtypes = ("Rgal","Rtan","near","far","tangent")
    titles = ("Galactocentric Radius",
              "Tangent Point Galactocentric Radius",
              "Near Distance","Far Distance","Tangent Point Distance")
    cbar_labels = ("Radius (kpc)","Radius (kpc)",
                   "Distance (kpc)","Distance (kpc)","Distance (kpc)")
    for kdtype,title,cbar_label in zip(kdtypes,titles,cbar_labels):
        dist = results[kdtype]
        err_neg = results[kdtype+"_err_neg"]
        err_pos = results[kdtype+"_err_pos"]
        err_tot = err_neg + err_pos
        #
        # Setup plot
        #
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True,
                                                 figsize=(8.5,11))
        fig.subplots_adjust(hspace=0)
        extent = [np.min(glongs),np.max(glongs),
                  np.min(velos),np.max(velos)]
        ax1.set_title(title)
        #
        # Plot distance in first panel
        #
        plot_dist = ax1.imshow(dist.T,origin='lower',aspect='auto',
                               extent=extent,interpolation='none',
                               vmin=vlim1[0],vmax=vlim1[1],
                               cmap='YlOrRd')
        plot_dist.cmap.set_over("black")
        cbar_dist = plt.colorbar(plot_dist,ax=ax1,fraction=0.046,
                                 pad=0.04,shrink=0.9)
        cbar_dist.set_label(cbar_label)
        cbar_dist.set_clim(vlim1[0],vlim1[1])
        ax1.set_ylabel(r"$V_{\rm LSR}$ (km s$^{-1}$)")
        #
        # Plot negative direction error in second panel
        #
        plot_err_neg = ax2.imshow(err_neg.T,origin='lower',
                                  aspect='auto',extent=extent,
                                  interpolation='none',
                                  vmin=vlim2[0],vmax=vlim2[1],
                                  cmap='YlOrRd')
        plot_err_neg.cmap.set_over("black")
        cbar_err_neg = plt.colorbar(plot_err_neg,ax=ax2,
                                    fraction=0.046,pad=0.04,
                                    shrink=0.9)
        cbar_err_neg.set_label("Neg. Error (kpc)")
        cbar_err_neg.set_clim(vlim2[0],vlim2[1])
        ax2.set_ylabel(r"$V_{\rm LSR}$ (km s$^{-1}$)")
        #
        # Plot positive direction error in third panel
        #
        plot_err_pos = ax3.imshow(err_pos.T,origin='lower',
                                  aspect='auto',extent=extent,
                                  interpolation='none',
                                  vmin=vlim3[0],vmax=vlim3[1],
                                  cmap='YlOrRd')
        plot_err_pos.cmap.set_over("black")
        cbar_err_pos = plt.colorbar(plot_err_pos,ax=ax3,
                                    fraction=0.046,pad=0.04,
                                    shrink=0.9)
        cbar_err_pos.set_label("Pos. Error (kpc)")
        cbar_err_pos.set_clim(vlim3[0],vlim3[1])
        ax3.set_ylabel(r"$V_{\rm LSR}$ (km s$^{-1}$)")
        #
        # Plot total error in fourth panel
        #
        plot_err_tot = ax4.imshow(err_tot.T,origin='lower',
                                  aspect='auto',extent=extent,
                                  interpolation='none',
                                  vmin=vlim4[0],vmax=vlim4[1],
                                  cmap='YlOrRd')
        plot_err_tot.cmap.set_over("black")
        cbar_err_tot = plt.colorbar(plot_err_tot,ax=ax4,
                                    fraction=0.046,pad=0.04,
                                    shrink=0.9)
        cbar_err_tot.set_label("Tot. Error (kpc)")
        cbar_err_tot.set_clim(vlim4[0],vlim4[1])
        ax4.set_ylabel(r"$V_{\rm LSR}$ (km s$^{-1}$)")
        ax4.set_xlabel(r"Galactic Longitude (deg)")
        plt.xlim(np.min(glongs),np.max(glongs))
        plt.ylim(np.min(velos),np.max(velos))
        plt.savefig(prefix+"_"+kdtype+".pdf")
        plt.close(fig)
