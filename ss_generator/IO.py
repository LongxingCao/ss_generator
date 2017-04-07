import numpy as np


def save_ca_list(ca_list, pdb_name):
    '''Save a Ca atoms into a PDB file.
    The input list can either be a list of coordinates, or
    a list of list.
    '''
    # Pack the single level list input

    if isinstance(ca_list[0], np.ndarray):
        ca_list = [ca_list]

    with open(pdb_name, 'w') as f:
        index = 0
        
        for l in ca_list:
            for ca in l:
            
                f.write("ATOM  {0:5d}   CA ALA A".format(index) \
                        + "{0:4d}    {1:8.3f}{2:8.3f}{3:8.3f}".format(index, ca[0], ca[1], ca[2]) \
                        + "                       C  \n")
                index += 1

            index += 1
            f.write("TER\n")
