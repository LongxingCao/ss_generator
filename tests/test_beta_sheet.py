#!/usr/bin/env python3

import pytest
import numpy as np
np.seterr(all='raise')

import ss_generator as ssg


def test_build_beta_sheet():
    print("test build beta sheet.")
    
    res_list = ssg.beta_sheet.build_ideal_flat_beta_sheet('antiparallel', 11, 10)
    ssg.IO.save_residue_list(res_list, "ideal_flat_sheet.pdb")

    for i, strand in enumerate(res_list):
        #ssg.basic.change_torsions(strand, 6, np.radians(-120), np.radians(134.6))
        #ssg.basic.change_torsions(strand, 5, np.radians(-153), np.radians(153))
        
        for j in range(0, len(strand), 2):
            ssg.basic.change_torsions(strand, j, np.radians(-144.8), np.radians(142))
            ssg.basic.change_torsions(strand, j + 1, np.radians(-144.8), np.radians(167))

    bp_map = {i : len(res_list[0]) - 1 - i  for i in range(len(res_list[0]))}
    for i in range(1, len(res_list)):
        res_list[i] = ssg.beta_sheet.attach_beta_strand_to_reference_by_screw(res_list[i], res_list[i - 1], 'antiparallel', bp_map, i % 2)
        #res_list[i] = ssg.beta_sheet.attach_beta_strand_to_reference_byHB(res_list[i], res_list[i - 1], 'parallel', bp_map, 1)

    ssg.IO.save_residue_list(res_list, "perturbed_sheet.pdb")
