#!/usr/bin/env python3

import pytest
import numpy as np

import ss_generator as ssg


def test_peptide_bond():
    print("test peptide bond.")
    d = ssg.basic.get_peptide_bond_parameters()
    print(d)
