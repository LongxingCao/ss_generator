import numpy as np

from . import geometry
from . import basic
from . import beta_sheet


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
    
    def __init__(self, crease, f_3d):
        '''Initalize the 3D crease with a 2D crease and
        a function that transforms a 2d point to a 3d point.
        '''
        self.crease2d = crease
        self.anchor = f_3d(crease[0])
        self.axis = geometry.normalize(f_3d(crease[1]) - f_3d(crease[0]))
        self.angle = crease[2]
        self.lower_left_crease_ids = []
        self.upper_right_crease_ids = []
        self.lower_left_point_ids = []
        self.upper_right_point_ids = []

    def transform(self, M, t):
        '''Apply a Euclidean transformation on the 3D crease.'''
        self.anchor = np.dot(M, self.anchor) + t
        self.axis = np.dot(M, self.axis)

    def get_transformation(self):
        '''Return the transformation defined by this crease.'''
        return geometry.get_screw_transformation(self.axis, 
                self.angle, 0, self.anchor)


class BetaSheetSkeleton:
    '''A Skeleton for a beta sheet.'''
    
    # Set some basic parameters
    
    DI_PEPTIDE_LENGTH = 6.7
    PARALLEL_D_INTER = 4.84
    ANTIPARALLEL_D_INTER_POS = 5.24
    ANTIPARALLEL_D_INTER_NEG = 4.50

    def __init__(self, topology, creases):
        '''Initialize a BetaSheetSkeleton with the topology and
        a list of creases of the skeleton. A topology is a list
        of tuples of three values. The first two values specify
        the starting point and the ending point of a strand. The
        third values is a boolean that tells if the hydrogen bonding
        atoms are upward facing. A crease is a tuple of two 2D points 
        on the boundary of the skeleton and a bending angle.
        '''
        self.topology = topology
        if not self.valid_topology():
            raise Exception("Invalid topology!")

        self.boundary = self.get_skeleton_boundary()
        self.low_left_corner = self.strand_end_points(0)[0] 

        # Initialize the 3D strands and HB directions

        self.strand3ds = []
        self.hb_directions = []
        
        for i in range(len(topology)):
            self.strand3ds.append([self.get_3d_point(p2) for p2
                in self.strand_points(i)])

            self.hb_directions.append( [np.array([0, 1 
                if topology[i][2] else -1, 0])] * len(self.strand3ds[-1]) )

        self.creases = creases
        self.crease3ds = [Crease3D(c, self.get_3d_point) for c in creases]
        self.init_crease3d_data()

        # Fold the skeleton

        self.fold()

    def strand_points(self, strand_id):
        '''Return the points on a strand in 2D'''
        strand = self.topology[strand_id]

        if strand[1] > strand[0]:
            return [np.array([j, strand_id]) for j in 
                    range(strand[0], strand[1] + 1)]
        
        else:
            return [np.array([j, strand_id]) for j in 
                    range(strand[0], strand[1] - 1, -1)]
        

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

    def get_3d_point(self, p2):
        '''Get a 3D point from a 2D point on the skeleton.'''
        x = p2[0] * BetaSheetSkeleton.DI_PEPTIDE_LENGTH

        # Calculate the y coordinate by cut the y axis into
        # strips with different scale factors

        def y_scale_factor(level):
            '''Get the y axis scale factor.'''
            if level < 0 or level + 1 >= len(self.topology):
                return BetaSheetSkeleton.PARALLEL_D_INTER

            elif self.topology[level][2] == self.topology[level + 1][2]:
                return BetaSheetSkeleton.PARALLEL_D_INTER

            elif self.topology[level][2]:
                return BetaSheetSkeleton.ANTIPARALLEL_D_INTER_POS

            else:
                return BetaSheetSkeleton.ANTIPARALLEL_D_INTER_NEG
        
        y = 0

        i = 0
        while i + 1 < p2[1]:
            
            y += y_scale_factor(i)
            i += 1
       
        y += y_scale_factor(i) * (p2[1] - i)

        return np.array([x, y, 0])

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
                for k, p2 in enumerate(self.strand_points(j)):
                    
                    if self.point_on_lower_left(p2, c3d.crease2d):
                        c3d.lower_left_point_ids.append((j, k))
                    else:
                        c3d.upper_right_point_ids.append((j, k))

    def fold(self):
        '''Fold the skeleton along the creases.'''
        for c3d in self.crease3ds:
            M, t = c3d.get_transformation()

            # Transform creases

            for i in c3d.upper_right_crease_ids:

                self.crease3ds[i].transform(M, t)

            # Transform points and their HB directions

            for i, j in c3d.upper_right_point_ids:

                self.strand3ds[i][j] = np.dot(M, self.strand3ds[i][j]) + t
                self.hb_directions[i][j] = np.dot(M, self.hb_directions[i][j])

    def get_dipeptide_bond_directions(self):
        '''Return a list of dipeptide bond directions
        for each strand.
        '''
        all_dipp_directions = []

        for i, strand in enumerate(self.strand3ds):
            dipp_directions = []

            for j in range(len(strand) - 1):
                
                d0 = geometry.normalize(strand[j + 1] - strand[j])
                
                # The HB direction of the dipeptide bond is calculated from
                # the two flanking HB directions

                d1 = -(self.hb_directions[i][j] + self.hb_directions[i][j + 1]) / 2
                
                dipp_directions.append((d0, geometry.normalize(d1 - np.dot(d1, d0) * d0)))

            all_dipp_directions.append(dipp_directions)

        return all_dipp_directions

    def thread_bb(self):
        '''Thread a backbone onto the skeleton.
        Return a residue list of all strands.
        '''
        sheet = []
        all_dipp_directions = self.get_dipeptide_bond_directions()

        for i, strand in enumerate(self.strand3ds):
            strand_bb = beta_sheet.build_beta_strand_from_dipeptide_directions(
                            all_dipp_directions[i])

            com1 = np.mean([strand_bb[j]['ca'] for j in range(0, len(strand_bb), 2)], axis=0)
            com2 = np.mean(strand, axis=0)

            sheet.append(basic.transform_residue_list(strand_bb, np.identity(3), com2 - com1))

        return sheet
