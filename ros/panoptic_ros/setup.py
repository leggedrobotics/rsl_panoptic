from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=["panoptic_ros"], package_name="panoptic_ros"
)

setup(**setup_args)
