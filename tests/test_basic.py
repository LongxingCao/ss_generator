#!/usr/bin/env python3

import pytest
import numpy as np

import ss_generator as ssg


def test_peptide_bond():
    print("test peptide bond.")

    M, t = ssg.basic.get_peptide_bond_transformation(np.radians(-138.9), np.radians(134.6))

    print(M)
    print(t)

    M2 = np.dot(M, M)
    t2 = t + np.dot(M, t)

    print(M2)
    print(t2)
    print(np.linalg.norm(t2))
