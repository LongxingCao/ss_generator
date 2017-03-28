#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='ss_generator',
    version='0.0.0',
    author='Xingjie Pan',
    author_email='xingjiepan@gmail.com',
    url='https://github.com/xingjiepan/ss_generator',
    packages=[
        'ss_generator',
    ],
    install_requires=[
        'numpy',
        'pytest',
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    description='ss_generator generates protein secondary structure backbones from a set of geometric parameters.',
    long_description=open('README.rst').read(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
    ],
)
