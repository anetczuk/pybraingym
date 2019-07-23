#! /usr/bin/env python2.5
# -*- coding: utf-8 -*-


__author__ = 'Arkadiusz Netczuk, dev.arnet@gmail.com'


from setuptools import setup, find_packages

setup(
    name="pybraingym",
    version="0.1",
    description="Wrapper classes for PyBrain to use OpenAi Gym environments.",
    license="MIT",
    author='Arkadiusz Netczuk',
    author_email='dev.arnet@gmail.com',
    keywords="Neural Networks Machine Learning",
    # url="",
    packages=find_packages(exclude=['examples', 'testpybraingym', 'runtests.py'])
)
