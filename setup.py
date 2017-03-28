#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='alpha_helix_generator',
    version='0.0.0',
    author='Xingjie Pan',
    author_email='xingjiepan@gmail.com',
    url='https://github.com/xingjiepan/alpha_helix_generator',
    packages=[
        'ProteinFeatureAnalyzer',
    ],
    install_requires=[
        'numpy',
        'pytest',
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    description='alpha_helix_generator generates alpha helix backbones from a set of geometric parameters.',
    long_description=open('README.rst').read(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
    ],
)
