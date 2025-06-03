"""
setup.py
Author: Jessica Anz
Description: Setup the environment parameters
"""

from setuptools import setup

setup(
    name="cat_env",
    version="0.1",
    packages=["cat_env"],
    install_requires=[
        "gymnasium",
        "mujoco",
        "numpy"
    ],
)
