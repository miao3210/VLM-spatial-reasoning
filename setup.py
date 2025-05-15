'''
Author: Wenhao Ding
Email: wenhaod@andrew.cmu.edu
Date: 2023-04-17 20:21:01
LastEditTime: 2023-09-16 15:47:19
Description: 
'''

from setuptools import find_packages, setup

import os 
if os.path.exists('requirements.txt'):
    with open('requirements.txt') as f:
        required = f.read().splitlines()

setup(
    name="spatok",
    version='0.1',
    packages=find_packages(),
    install_requires=required,
    python_requires=">=3.8",
)
