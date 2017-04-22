import numpy as np

def pdb_string(a_id, r_id, a_name, r_name, position, element):
    '''Generate a pdb string for an atom.'''
    return "ATOM  {0:5d} {1: >4} {2: >3} A".format(a_id, a_name, r_name) \
            + "{0:4d}    {1:8.3f}{2:8.3f}{3:8.3f}".format(r_id, position[0], position[1], position[2]) \
            + "                      {0: >2}  \n".format(element)


def save_ca_list(ca_list, pdb_name):
    '''Save a Ca atoms into a PDB file.
    The input list can either be a list of coordinates, or
    a list of list.
    '''
    # Pack the single level list input

    if isinstance(ca_list[0], np.ndarray):
        ca_list = [ca_list]

    with open(pdb_name, 'w') as f:
        index = 1
        
        for l in ca_list:
            for ca in l:
            
                f.write(pdb_string(index, index, 'CA', 'ALA', ca, 'C'))
                index += 1

            index += 1
            f.write("TER\n")

def save_residue_list(residue_list, pdb_name):
    '''Save a list of residue dictionaries into a PDB file.
    The input list can either be a list of residues, or a
    list of list.
    '''
    # Pack the single level list input

    if isinstance(residue_list[0], dict):
        residue_list =  [residue_list]

    with open(pdb_name, 'w') as f:
        a_index = 1
        r_index = 1

        for l in residue_list:
            for res in l:
                for atom in res.keys():
                
                    f.write(pdb_string(a_index, r_index, atom.upper(), 'ALA',
                        res[atom], atom.upper()[0]))
                    a_index += 1

                r_index += 1

            r_index += 1
            f.write("TER\n")
