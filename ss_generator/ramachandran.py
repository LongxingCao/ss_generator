import numpy as np


def get_cumulative_probability(distribution):
    '''Calculate the cumulative distribution for 
    a ramachanran distribution.
    '''

    cumulative_distribution = [distribution[0]['probability']]

    for i in range(1, len(distribution)):
        cumulative_distribution.append(distribution[i]['probability']
                + cumulative_distribution[-1])

    return cumulative_distribution

def random_torsions(distribution, cumulative_distribution):
    '''Return a random pair of phi, psi torsions from
    a ramachanran distribution and its cumulative_distribution.
    '''

    p = np.random.uniform()

    i = 0
    while cumulative_distribution[i] < p:
        i += 1
    
    d = distribution[i]

    return (d['phi'] + np.random.uniform(0, d['bin_width']),
            d['psi'] + np.random.uniform(0, d['bin_width']))
