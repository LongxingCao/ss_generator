def save_ca_list(ca_list, pdb_name):
    '''Save a list of Ca atoms into a PDB file'''
    with open(pdb_name, 'w') as f:
        for i, ca in enumerate(ca_list):
        
            f.write("ATOM  {0:5d}   CA ALA A".format(i + 1) \
                    + "{0:4d}    {1:8.3f}{2:8.3f}{3:8.3f}".format(i + 1, ca[0], ca[1], ca[2]) \
                    + "                       C  \n")
