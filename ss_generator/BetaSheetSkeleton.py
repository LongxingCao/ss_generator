import numpy as np

from . import geometry


def f_equal(f1, f2, cut_off=0.001):
    '''Return true if two float values are equal.'''
    return np.absolute(f1 - f2) < cut_off

def angle_2d(p1, p2, p3):
    '''Return the angle rotating the vector p2p1 to p2p3.
    The points are 2 dimensional.
    '''
    v1 = geometry.normalize(np.array([p1[0] - p2[0], p1[1] - p2[1], 0]))
    v2 = geometry.normalize(np.array([p3[0] - p2[0], p3[1] - p2[1], 0]))

    z = np.array([0, 0, 1])

    s = np.dot(np.cross(v1, v2), z)
    c = np.dot(v1, v2)

    return np.arctan2(s, c)

def to_3d(v):
    '''Change a 2D vector to a 3D vector.'''
    return np.array([v[0], v[1], 0])

class Crease3D:
    '''A 3D crease with some affiliating data'''
    
    def __init__(self, crease):
        '''Initalize the 3D crease with a 2D crease.'''
        self.crease2d = crease
        self.anchor = to_3d(crease[0])
        self.axis = geometry.normalize(to_3d(crease[1] - crease[0]))
        self.angle = crease[2]
        self.lower_left_crease_ids = []
        self.upper_right_crease_ids = []
        self.lower_left_point_ids = []
        self.upper_right_point_ids = []


class BetaSheetSkeleton:
    '''A Skeleton for a beta sheet.'''
    
    def __init__(self, topology, creases):
        '''Initialize a BetaSheetSkeleton with the topology and
        a list of creases of the skeleton. A topology is a list
        of pairs. Each pair specifies the starting point and the
        ending point of a strand. A crease is a pair of 2D points 
        on the boundary of the skeleton and a bending angle.
        '''
        self.topology = topology
        if not self.valid_topology():
            raise Exception("Invalid topology!")

        self.boundary = self.get_skeleton_boundary()
        self.low_left_corner = self.strand_end_points(0)[0] 

        # Initialize the 3D strands

        self.strand3ds = []
        for i in range(len(topology)):
            s = []
            
            for j in range(topology[i][0], topology[i][1] + 1):
                s.append(np.array([j, i, 0]))
            
            self.strand3ds.append(s)

        self.creases = creases
        self.crease3ds = [Crease3D(c) for c in creases]
        self.init_crease3d_data()

    def strand_end_points(self, strand_id):
        '''Return the left and right end points of
        a strand.
        '''
        p0 = np.array([self.topology[strand_id][0], strand_id])
        p1 = np.array([self.topology[strand_id][1], strand_id])

        return (p0, p1) if p0[0] < p1[0] else (p1, p0)

    def valid_topology(self):
        '''Return true if the topology is valid.'''
        n = len(self.topology)
        end_points = [self.strand_end_points(i) for i in range(n)]

        for i in range(n):
            neighbors = []
            if i > 0: neighbors.append(i - 1)
            if i < n - 1: neighbors.append(i + 1)

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

    def find_point_on_boundary(self, point):
        '''Find a point on the boundary.
        Return the index of the edge.
        '''
        for i in range(len(self.boundary)):
            if f_equal(0, geometry.point_segment_distance(point,
                self.boundary[i][0], self.boundary[i][1])):
                return i

    def split_boundary_by_crease(self, crease):
        '''Return two splited boundaries cut by a crease.
        The first splited boundary contains the lower left
        corner.
        '''
        # Find the indices of edges to be splited

        e0 = self.find_point_on_boundary(crease[0])
        e1 = self.find_point_on_boundary(crease[1])

        # Get the splited edges

        se00 = (self.boundary[e0][0], crease[0])
        se01 = (crease[0], self.boundary[e0][1])
        se10 = (self.boundary[e1][0], crease[1])
        se11 = (crease[1], self.boundary[e1][1])

        # Get the first splited boundary

        sb1 = [] if f_equal(0, np.linalg.norm(se01[1] - se01[0])) else [se01]

        i = (e0 + 1) % len(self.boundary)

        while i != e1:
            sb1.append(self.boundary[i])
            i = (i + 1) % len(self.boundary)

        if not f_equal(0, np.linalg.norm(se10[1] - se10[0])):
            sb1.append(se10)

        sb1.append((crease[1], crease[0]))

        # Get the second splited boundary

        sb2 = [] if f_equal(0, np.linalg.norm(se11[1] - se11[0])) else [se11]

        i = (e1 + 1) % len(self.boundary)

        while i != e0:
            sb2.append(self.boundary[i])
            i = (i + 1) % len(self.boundary)

        if not f_equal(0, np.linalg.norm(se00[1] - se00[0])):
            sb2.append(se00)

        sb2.append((crease[0], crease[1]))

        # Find out the order of sb1 and sb2

        if e0 < e1:
            return sb2, sb1
        else:
            return sb1, sb2

    def point_on_lower_left(self, point, crease):
        '''Return true if a point is on the lower left
        side of a crease. 
        '''
        # Get the lower left part of the splited boundary

        sb_ll = self.split_boundary_by_crease(crease)[0]

        # Test if the point is on the boundary

        for edge in sb_ll:
            if f_equal(0, geometry.point_segment_distance(point, *edge)):
                return True

        # Test if the point is inside the lower left boundary

        total_angle = 0

        for edge in sb_ll:
            total_angle += angle_2d(edge[0], point, edge[1])

        if np.absolute(total_angle) > np.pi:
            return True

        return False

    def crease_on_lower_left(self, crease1, crease2):
        '''Return true if the crease1 is on the lower left
        side of the crease2.
        '''
        return self.point_on_lower_left(crease1[0], crease2) \
                and self.point_on_lower_left(crease1[1], crease2)

    def init_crease3d_data(self):
        '''Initialize datas for the crease3ds'''

        for i, c3d in enumerate(self.crease3ds):

            # Initialize the relationships between creases

            for j, c3d2 in enumerate(self.crease3ds):
                if i == j:
                    continue
                elif self.crease_on_lower_left(c3d2.crease2d, c3d.crease2d):
                    c3d.lower_left_crease_ids.append(j)
                else:
                    c3d.upper_right_crease_ids.append(j)

            # Initialize the relationships between creases and points

            for j, strand in enumerate(self.topology):
                for k in range(strand[0], strand[1] + 1):
                    
                    if self.point_on_lower_left(np.array([k, j]), c3d.crease2d):
                        c3d.lower_left_point_ids.append((j, k - strand[0]))
                    else:
                        c3d.upper_right_point_ids.append((j, k - strand[0]))



