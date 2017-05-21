import numpy as np

from . import geometry


def f_equal(f1, f2, cut_off=0.001):
    '''Return true if two float values are equal.'''
    return np.absolute(f1 - f2) < cut_off

class Face:
    '''A face on a beta sheet skeleton.'''
    def __init__(self, boundary):
        '''Initialize a Face with a list of segments
        in counter cyclic order which make the boundary 
        of the face.
        '''
        self.boundary = boundary

class BetaSheetSkeleton:
    '''A Skeleton for a beta sheet.'''
    
    def __init__(self, topology, creases):
        '''Initialize a BetaSheetSkeleton with the topology and
        a list of creases of the skeleton. A topology is a list
        of pairs. Each pair specifies the starting point and the
        ending point of a strand. A crease is a pair of 2D points 
        on the boundary of the skeleton.
        '''
        self.topology = topology
        if self.valid_topology():
            raise Exception("Invalid topology!")

        self.creases = creases

    def strand_end_points(self, strand_id):
        '''Return the left and right end points of
        a strand.
        '''
        p0 = np.array([self.topology[strand_id][0], strand_id])
        p1 = np.array([self.topology[strand_id][1], strand_id])

        return (p0, p1) if p0[0] < p1[0] else (p1, p0)

    def valid_topology(self):
        '''Return true if the topolgy is valid.'''
        n = len(self.topology)
        end_points = [self.strand_end_points(i) for i in range(n)]

        for i in range(n):
            neighbors = []
            if i > 0: neighbors.append(i - 1)
            if i < n: neighbors.append(i + 1)

            valid_left = False
            valid_right = False

            for j in neighbors:
                if end_points[i][0][0] >= end_points[j][0][0]:
                    valid_left = True
                if end_points[i][1][0] <= end_points[j][1][0]:
                    valid_right = True

            if (not valid_left) or (not valid_right):
                return False

        return True

    def get_skeleton_boundary(self):
        '''Calculate the boundary of the beta sheet skeleton 
        from its topology.
        '''
        # Add the bottom boundary

        boundary = [self.strand_end_points(0)]

        # Add the right boundary

        for i in range(len(self.topology) - 1):
            r1 = self.strand_end_points(i)[1]
            r2 = self.strand_end_points(i + 1)[1]

            if f_equal(r1[0], r2[0]):
                boundary.append((r1, r2))
            
            elif r1[0] < r2[0]:
                p = np.array([r1[0], r2[1]])
                boundary.append((r1, p))
                boundary.append((p, r2))

            else:
                p = np.array([r2[0], r1[1]])
                boundary.append((r1, p))
                boundary.append((p, r2))

        # Add the top boundary
        
        top_ends = self.strand_end_points(len(self.topology) - 1)
        boundary.append((top_ends[1], top_ends[0]))

        # Add the left boundary

        for i in range(len(self.topology) - 1, 0, -1):
            l1 = self.strand_end_points(i)[0]
            l2 = self.strand_end_points(i - 1)[0]

            if f_equal(l1[0], l2[0]):
                boundary.append((l1, l2))

            elif l1[0] < l2[0]:
                p = np.array([l2[0], l1[1]])
                boundary.append((l1, p))
                boundary.append((p, l2))

            else:
                p = np.array([l1[0], l2[1]])
                boundary.append((l1, p))
                boundary.append((p, l2))

        return boundary

    def init_face_tree(self):
        '''Initialize the face tree of a skeleton.'''
        self.faces = []
