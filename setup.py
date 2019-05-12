from setuptools import setup

setup(
   name='kd',
   version='1.0',
   description='Kinematic distance utilities',
   author='Trey V. Wenger',
   author_email='tvwenger@gmail.com',
   packages=['kd'],
   install_requires=['numpy', 'matplotlib', 'scipy', 'pyqt_fit'],
)
