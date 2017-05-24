#!/usr/bin/env python3

import pytest
import numpy as np
np.seterr(all='raise')

import ss_generator as ssg


def test_build_beta_sheet():
    print("test build beta sheet.")
    
    res_list = ssg.beta_sheet.build_ideal_flat_beta_sheet('parallel', 10, 2)
    #res_list = ssg.beta_sheet.build_ideal_flat_beta_sheet('antiparallel', 41, 2)
    ssg.IO.save_residue_list(res_list, "ideal_flat_sheet.pdb")


    phi1 = -120 
    psi1 = 125
    phi2 = -120
    psi2 = 125

    for i, strand in enumerate(res_list):
        #ssg.basic.change_torsions(strand, 6, np.radians(-120), np.radians(134.6))
        #ssg.basic.change_torsions(strand, 5, np.radians(-153), np.radians(153))
        
        for j in range(0, len(strand), 2):
            ssg.basic.change_torsions(strand, j, np.radians(phi1), np.radians(psi1))
            ssg.basic.change_torsions(strand, j + 1, np.radians(phi2), np.radians(psi2))


    ssg.IO.save_residue_list(res_list, "perturbed_sheet.pdb")


    res_list = ssg.beta_sheet.build_beta_barrel('parallel', 12, 20, np.radians(20))
    ssg.IO.save_residue_list(res_list, "beta_barrel.pdb")

def test_beta_sheet_skeleton():
    print("test beta sheet skeleton.")

    #topology = [(3, 10, True), (12, 3, False), (4, 12, True), (5, 9, True)]
    topology = [(0, 5, True), (5, 0, False), (1, 6, True), (6, 1, False),
                (2, 7, True), (7, 2, False), (3, 8, True), (8, 3, False),
                (4, 9, True), (9, 4, False), (5, 10, True), (10, 5, False)]
    #creases = [(np.array([5, 0]), np.array([6.5, 3]), np.radians(30)),
    #            (np.array([3.5, 0]), np.array([3.5, 1]), np.radians(10))]
    #angle = 32
    #creases = [(np.array([0, 1]), np.array([5,0]), np.radians(angle)),
    #           (np.array([1, 2]), np.array([5,1]), np.radians(angle)),
    #           (np.array([1, 3]), np.array([6,2]), np.radians(angle)),
    #           (np.array([2, 4]), np.array([6,3]), np.radians(angle)),
    #           (np.array([3, 5]), np.array([7,4]), np.radians(angle)),
    #           (np.array([3, 6]), np.array([7,5]), np.radians(angle)),
    #           (np.array([4, 7]), np.array([8,6]), np.radians(angle)),
    #           (np.array([4, 8]), np.array([8,7]), np.radians(angle)),
    #           (np.array([5, 9]), np.array([9,8]), np.radians(angle)),
    #           (np.array([5, 10]), np.array([9,9]), np.radians(angle)),
    #           (np.array([6, 11]), np.array([10,10]), np.radians(angle))]
    creases = []

    skeleton = ssg.BetaSheetSkeleton(topology, creases)
    
    #print(skeleton.get_skeleton_boundary())
    #print(skeleton.split_boundary_by_crease(creases[0]))
    #print(skeleton.point_on_lower_left(np.array([4.5, 1]), creases[0]))
    #print(skeleton.crease3ds[0].lower_left_poi,t_ids)
    
    ca_list = skeleton.strand3ds
    ssg.IO.save_ca_list(ca_list, "beta_sheet_skeleton.pdb")

    res_list = skeleton.thread_bb()
    ssg.IO.save_residue_list(res_list, "beta_sheet_from_skeleton.pdb")
