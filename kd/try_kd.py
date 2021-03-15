from kd import pdf_kd

# ! FIX KRIGING IN cw21_rotcurve
# ! USING TOO MUCH MEMORY WITH KRIGING!
glong = 30.0  # Galactic longitude, degrees
glat = 1.0  # Galactic latitude, degrees
velo = 20.0  # measured LSR velocity, km/s
velo_err = 5.0  # measured LSR velocity uncertainty, km/s
rotcurve = "cw21_rotcurve"  # the name of the script containing the rotation curve
num_samples = 10000  # number of re-samples
peculiar=True
use_kriging=True
dist = pdf_kd.pdf_kd(
    glong, glat, velo, velo_err=velo_err, rotcurve=rotcurve, num_samples=num_samples,
    peculiar=peculiar,
    use_kriging=use_kriging
)
print(dist)

# from kd import rotcurve_kd

# glong = 30.0  # Galactic longitude, degrees
# glat = 1.0  # Galactic latitude, degrees
# velo = 20.0  # measured LSR velocity, km/s
# velo_tol = 0.1  # tolerance to determine a "match" between rotation curve and measured LSR velocity (km/s)
# rotcurve = "cw21_rotcurve"  # the name of the script containing the rotation curve
# dist = rotcurve_kd.rotcurve_kd(glong, glat, velo, velo_tol=velo_tol, rotcurve=rotcurve,
#                                peculiar=True,
#                                use_kriging=True
#                                )
# print(dist)


# * STATS WHERE KRIGING IS FALSE AND num_samples=10000, use_peculiar=True
# {'Rgal': 7.113165261276646, 'Rgal_kde': <pyqt_fit.kde.KDE1D object at 0x7f483d37e490>,
#  'Rgal_err_neg': 0.3289311988094408, 'Rgal_err_pos': 0.34434984875363295,
#  'Rtan': 4.0910762243850085, 'Rtan_kde': <pyqt_fit.kde.KDE1D object at 0x7f483d37e2b0>,
#  'Rtan_err_neg': 0.016299715815669913, 'Rtan_err_pos': 0.012584339416509494,
#  'near': 1.183099099099098, 'near_kde': <pyqt_fit.kde.KDE1D object at 0x7f483d37e5b0>,
#  'near_err_neg': 0.3429129129129125, 'near_err_pos': 0.47225725725725676,
#  'far': 12.902038038038027, 'far_kde': <pyqt_fit.kde.KDE1D object at 0x7f483d37e370>,
#  'far_err_neg': 0.39409409409409335, 'far_err_pos': 0.4290540540540544,
#  'distance': 1.182108108108107,
#  'distance_kde': <pyqt_fit.kde.KDE1D object at 0x7f483d37e880>,
#  'distance_err_neg': 0.703608608608608, 'distance_err_pos': 12.521361361361352,
#  'tangent': 7.087108108108102,
#  'tangent_kde': <pyqt_fit.kde.KDE1D object at 0x7f483d37e280>,
#  'tangent_err_neg': 0.02838738738738744, 'tangent_err_pos': 0.0217567567567567,
#  'vlsr_tangent': 108.45176268476177,
#  'vlsr_tangent_kde': <scipy.stats.kde.gaussian_kde object at 0x7f483d37e1c0>,
#  'vlsr_tangent_err_neg': 6.56097129007523, 'vlsr_tangent_err_pos': 5.890507070651481}

# {'Rgal': 6.815887055972002, 'Rgal_kde': <pyqt_fit.kde.KDE1D object at 0x7fa9f32080d0>,
# 'Rgal_err_neg': 0.2910560890635976, 'Rgal_err_pos': 0.2304194038420153, 'Rtan':
# 4.091961598431878, 'Rtan_kde': <pyqt_fit.kde.KDE1D object at 0x7fa9f32081f0>,
# 'Rtan_err_neg': 0.015260318414514806, 'Rtan_err_pos': 0.013893722735603298, 'near':
# 1.2010330330330319, 'near_kde': <pyqt_fit.kde.KDE1D object at 0x7fa9f3208490>,
# 'near_err_neg': 0.38156756756756716, 'near_err_pos': 0.2920640640640637, 'far':
# 12.55309109109108, 'far_kde': <pyqt_fit.kde.KDE1D object at 0x7fa9f3208280>,
# 'far_err_neg': 0.37684384384384195, 'far_err_pos': 0.2794284284284281, 'distance':
# 1.0722952952952942, 'distance_kde': <pyqt_fit.kde.KDE1D object at 0x7fa9f3208430>,
# 'distance_err_neg': 0.6501431431431425, 'distance_err_pos': 12.172892892892882, 'tangent':
# 6.966801801801796, 'tangent_kde': <pyqt_fit.kde.KDE1D object at 0x7fa9f3208160>,
# 'tangent_err_neg': 0.046270270270270863, 'tangent_err_pos': 0.040165165165165106,
# 'vlsr_tangent': 103.41141726309323, 'vlsr_tangent_kde': <scipy.stats.kde.gaussian_kde
# object at 0x7fa9f3208250>, 'vlsr_tangent_err_neg': 2.1486288506405486,
# 'vlsr_tangent_err_pos': 2.3516488995199865}

# {'Rgal': 6.771681817508346, 'Rgal_kde': <pyqt_fit.kde.KDE1D object at 0x7fb214157460>,
# 'Rgal_err_neg': 0.2468161648518823, 'Rgal_err_pos': 0.277162414628755, 'Rtan':
# 4.091106476662002, 'Rtan_kde': <pyqt_fit.kde.KDE1D object at 0x7fb2141573a0>,
# 'Rtan_err_neg': 0.014488359509440052, 'Rtan_err_pos': 0.014992302448899153, 'near':
# 1.1214514514514506, 'near_kde': <pyqt_fit.kde.KDE1D object at 0x7fb2141575e0>,
# 'near_err_neg': 0.29487487487487474, 'near_err_pos': 0.3810690690690688, 'far':
# 12.479005005004995, 'far_kde': <pyqt_fit.kde.KDE1D object at 0x7fb214157550>,
# 'far_err_neg': 0.3045085085085084, 'far_err_pos': 0.353539539539538, 'distance':
# 1.071869869869869, 'distance_kde': <pyqt_fit.kde.KDE1D object at 0x7fb2141577c0>,
# 'distance_err_neg': 0.6566826826826822, 'distance_err_pos': 12.183559559559548, 'tangent':
# 6.963224224224218, 'tangent_kde': <pyqt_fit.kde.KDE1D object at 0x7fb2141574c0>,
# 'tangent_err_neg': 0.044064064064063935, 'tangent_err_pos': 0.04127127127127128,
# 'vlsr_tangent': 103.42586257157042, 'vlsr_tangent_kde': <scipy.stats.kde.gaussian_kde
# object at 0x7fb214157100>, 'vlsr_tangent_err_neg': 2.1228568357115307,
# 'vlsr_tangent_err_pos': 2.321874664059493}

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