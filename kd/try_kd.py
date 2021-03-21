import time
import numpy as np

# from kd import pdf_kd
# # glong = 30.0  # Galactic longitude, degrees
# # glat = 1.0  # Galactic latitude, degrees
# # velo = 20.0  # measured LSR velocity, km/s
# # velo_err = 5.0  # measured LSR velocity uncertainty, km/s
# # glong = np.array([30., 35.])
# # glat = np.array([1., 0.5])
# # velo = np.array([20., 12.])
# # velo_err = np.array([5., 10.])
# glong = np.random.normal(loc=30., scale=2., size=5)
# glat = np.random.normal(loc=0., scale=0.5, size=5)
# velo = np.random.normal(loc=20., scale=20., size=5)
# velo_err = 10.
# rotcurve = "cw21_rotcurve"  # the name of the script containing the rotation curve
# num_samples = 100  # number of re-samples
# peculiar=True
# use_kriging=True
# start = time.time()
# dist = pdf_kd.pdf_kd(
#     glong, glat, velo, velo_err=velo_err, rotcurve=rotcurve, num_samples=num_samples,
#     peculiar=peculiar,
#     use_kriging=use_kriging,
#     processes=None
# )
# end = time.time()

from kd import rotcurve_kd
glong = 30.0  # Galactic longitude, degrees
glat = 1.0  # Galactic latitude, degrees
velo = 20.0  # measured LSR velocity, km/s
# glong = np.array([30., 35.])
# glat = np.array([1., 0.5])
# velo = np.array([20., 12.])
velo_tol = 0.1  # tolerance to determine a "match" between rotation curve and measured LSR velocity (km/s)
rotcurve = "cw21_rotcurve"  # the name of the script containing the rotation curve
start = time.time()
dist = rotcurve_kd.rotcurve_kd(glong, glat, velo, velo_tol=velo_tol, rotcurve=rotcurve,
                               peculiar=True,
                               use_kriging=True,
                               processes=None
                               )
end = time.time()

print(dist)
print(end - start)

# * MC KD STATS WHERE use_kriging=True AND num_samples=1000, use_peculiar=True
# {'Rgal': 6.5679196741081505, 'Rgal_kde': <pyqt_fit.kde.KDE1D object at 0x7fe8c61163a0>,
# 'Rgal_err_neg': 0.3124659942142243, 'Rgal_err_pos': 0.5827738780979574, 'Rtan':
# 4.102692166719391, 'Rtan_kde': <pyqt_fit.kde.KDE1D object at 0x7fe8c6116670>,
# 'Rtan_err_neg': 0.03382081017153471, 'Rtan_err_pos': 0.049040174748724574, 'near':
# 1.1203723723723713, 'near_kde': <pyqt_fit.kde.KDE1D object at 0x7fe8c61164c0>,
# 'near_err_neg': 0.6311951951951946, 'near_err_pos': 0.6379459459459453, 'far':
# 12.637492492492482, 'far_kde': <pyqt_fit.kde.KDE1D object at 0x7fe8c61165e0>,
# 'far_err_neg': 0.7990070070070079, 'far_err_pos': 0.33395995995995875, 'distance':
# 0.43798798798798766, 'distance_kde': <pyqt_fit.kde.KDE1D object at 0x7fe8c61109d0>,
# 'distance_err_neg': 0.41798798798798764, 'distance_err_pos': 12.818298298298286,
# 'tangent': 6.91812612612612, 'tangent_kde': <pyqt_fit.kde.KDE1D object at 0x7fe8c6116c10>,
# 'tangent_err_neg': 0.6737237237237235, 'tangent_err_pos': 0.49278078078078025,
# 'vlsr_tangent': 128.30197539818863, 'vlsr_tangent_kde': <scipy.stats.kde.gaussian_kde
# object at 0x7fe8c6116940>, 'vlsr_tangent_err_neg': 6.115230929207428,
# 'vlsr_tangent_err_pos': 7.632879480981558}

# * MC KD STATS WHERE use_kriging=False AND num_samples=1000, use_peculiar=True
# {'Rgal': 6.772585433766652, 'Rgal_kde': <pyqt_fit.kde.KDE1D object at 0x7f1d1daa9c70>,
# 'Rgal_err_neg': 0.26098633556927275, 'Rgal_err_pos': 0.27950794648064026, 'Rtan':
# 4.091192541639824, 'Rtan_kde': <pyqt_fit.kde.KDE1D object at 0x7f1d1daae400>,
# 'Rtan_err_neg': 0.015789913024458535, 'Rtan_err_pos': 0.0135194680144064, 'near':
# 1.1968098098098086, 'near_kde': <pyqt_fit.kde.KDE1D object at 0x7f1d1daa9d90>,
# 'near_err_neg': 0.39248448448448414, 'near_err_pos': 0.3002742742742739, 'far':
# 12.49089989989989, 'far_kde': <pyqt_fit.kde.KDE1D object at 0x7f1d1daa9e20>,
# 'far_err_neg': 0.33584484484484634, 'far_err_pos': 0.3444014014014005, 'distance':
# 0.7071731731731726, 'distance_kde': <pyqt_fit.kde.KDE1D object at 0x7f1d1c22e1c0>,
# 'distance_err_neg': 0.6641731731731727, 'distance_err_pos': 12.551517517517507, 'tangent':
# 6.967413413413407, 'tangent_kde': <pyqt_fit.kde.KDE1D object at 0x7f1d1daaee20>,
# 'tangent_err_neg': 0.0475775775775773, 'tangent_err_pos': 0.0345795795795798,
# 'vlsr_tangent': 103.71326370375851, 'vlsr_tangent_kde': <scipy.stats.kde.gaussian_kde
# object at 0x7f1d1daaed90>, 'vlsr_tangent_err_neg': 2.5667565844932767,
# 'vlsr_tangent_err_pos': 2.0618208629536383}

# * MC KD STATS WHERE use_kriging=False AND num_samples=10000, use_peculiar=True

# {'Rgal': 6.682977711381716, 'Rgal_kde': <pyqt_fit.kde.KDE1D object at 0x7f6a27837580>,
# 'Rgal_err_neg': 0.18324522806633592, 'Rgal_err_pos': 0.32377070971230015, 'Rtan':
# 4.098630767109873, 'Rtan_kde': <pyqt_fit.kde.KDE1D object at 0x7f6a27837e80>,
# 'Rtan_err_neg': 0.022081795630882972, 'Rtan_err_pos': 0.00897993022322563, 'near':
# 1.2112812812812805, 'near_kde': <pyqt_fit.kde.KDE1D object at 0x7f6a27821100>,
# 'near_err_neg': 0.40017617617617607, 'near_err_pos': 0.30497297297297266, 'far':
# 12.371695695695685, 'far_kde': <pyqt_fit.kde.KDE1D object at 0x7f6a27837100>,
# 'far_err_neg': 0.23130730730730775, 'far_err_pos': 0.40901901901901816, 'distance':
# 0.20599999999999985, 'distance_kde': <pyqt_fit.kde.KDE1D object at 0x7f6a27821220>,
# 'distance_err_neg': 0.0, 'distance_err_pos': 13.07399999999999, 'tangent':
# 6.9570480480480414, 'tangent_kde': <pyqt_fit.kde.KDE1D object at 0x7f6a27837970>,
# 'tangent_err_neg': 0.04925225225225205, 'tangent_err_pos': 0.05156456456456482,
# 'vlsr_tangent': 103.57795938384379, 'vlsr_tangent_kde': <scipy.stats.kde.gaussian_kde
# object at 0x7f6a27821190>, 'vlsr_tangent_err_neg': 2.5660338312901416,
# 'vlsr_tangent_err_pos': 1.9480909902855785}

# {'Rgal': 6.787773834404257, 'Rgal_kde': <pyqt_fit.kde.KDE1D object at 0x7fd6aca4a1c0>,
# 'Rgal_err_neg': 0.22587874002542652, 'Rgal_err_pos': 0.26332387874621954, 'Rtan':
# 4.0960788588515085, 'Rtan_kde': <pyqt_fit.kde.KDE1D object at 0x7fd6aca4ad00>,
# 'Rtan_err_neg': 0.018410237673607277, 'Rtan_err_pos': 0.01810340037904634, 'near':
# 1.183136136136135, 'near_kde': <pyqt_fit.kde.KDE1D object at 0x7fd6aca4a400>,
# 'near_err_neg': 0.3624264264264261, 'near_err_pos': 0.2718198198198194, 'far':
# 12.507828828828817, 'far_kde': <pyqt_fit.kde.KDE1D object at 0x7fd6aca3d0a0>,
# 'far_err_neg': 0.2932872872872867, 'far_err_pos': 0.3270690690690685, 'distance':
# 0.47099999999999964, 'distance_kde': <pyqt_fit.kde.KDE1D object at 0x7fd6aca3d1c0>,
# 'distance_err_neg': 0.0, 'distance_err_pos': 12.713999999999988, 'tangent':
# 6.9764714714714655, 'tangent_kde': <pyqt_fit.kde.KDE1D object at 0x7fd6aca3d130>,
# 'tangent_err_neg': 0.06414414414414438, 'tangent_err_pos': 0.0457027027027026,
# 'vlsr_tangent': 103.05084945564319, 'vlsr_tangent_kde': <scipy.stats.kde.gaussian_kde
# object at 0x7fd6aca3d250>, 'vlsr_tangent_err_neg': 2.199310030823881,
# 'vlsr_tangent_err_pos': 2.840775456480827}

# * TRADITIONAL KD STATS WHERE use_kriging=False AND use_peculiar=True
# {'Rgal': 6.7913453145017195, 'Rtan': 4.089104612447751, 'near': 1.1569999999999998, 'far':
# 12.504999999999999, 'tangent': 6.9590000000000005, 'vlsr_tangent': 103.09739584279757}
# * TRADITIONAL KD STATS WHERE use_kriging=True AND use_peculiar=True
# {'Rgal': 6.63667612110735, 'Rtan': 4.098921586619286, 'near': 1.127, 'far': 12.31,
# 'tangent': 6.772, 'vlsr_tangent': 99.99022165798738}