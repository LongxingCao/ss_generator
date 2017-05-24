#!/usr/bin/env python3

import pytest
import numpy as np
np.seterr(all='raise')

import ss_generator as ssg


def test_build_beta_sheet():
    print("test build beta sheet.")
    
    res_list = ssg.alpha_helix.build_ideal_straight_alpha_helix(20)
    ssg.IO.save_residue_list(res_list, "ideal_straight_helix.pdb")


