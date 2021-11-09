"""
Copyright(C) 2017-2021 by
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
"""

from setuptools import setup

setup(
    name="kd",
    version="2.1",
    description="Kinematic distance utilities",
    author="Trey V. Wenger",
    author_email="tvwenger@gmail.com",
    packages=["kd"],
    install_requires=["numpy", "matplotlib", "scipy", "pathos"],
    package_data={"kd": ["curve_data_wise_small.sav", "reid19_params.pkl"]},
)
