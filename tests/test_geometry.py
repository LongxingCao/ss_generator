#!/usr/bin/env python3

import pytest
import numpy as np

import ss_generator as ssg


def test_screw():
    print("test screw.")

    M, t = ssg.geometry.get_screw_transformation(np.array([0, 0, 1]), 1, 2.3, np.array([3, 4, 0]))

    print(M)
    print(t)

    axis, theta, pitch, u = ssg.geometry.get_screw_parameters(M, t)

    print('axis = ', axis)
    print('theta = ', theta)
    print('pitch = ', pitch)
    print('u = ', u)
