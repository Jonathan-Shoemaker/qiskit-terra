"""Implementation of the Solovay Kitaev Algorithm for
synthesizing single qubit gates in any universal gate set """

import math
import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm, schur
from qiskit.circuit.library import RXGate, RYGate


class SolovayKitaev():
    """ Solovay Kitaevy algorithm for single qubit gates
    Initialize on some gate set that is a list containing tuples
    of the form (name, operations matrix)

    This input gate set is required to be universal for single qubit gates.
    Further, each gate's adjoint should also present in the gate set.

    Given an input gate and an error epsilon the algorithm returns
    a sequence of gates whose product is within error epsilon of the
    input gate to a factor of global phase.

    The length of the gate sequence is polylogarithmic in 1/epsilon

    """

    def __init__(self, instruction_set):
        """ Set up structures given some instruction set

        First, the gates need to be cast into SU(2)

        Then, the adjoints of the gates are calculated

        Then, sequences of the gates are constructed up to some length. This is done in order
        to better calculate basic approximations to any unitary.

        Finally a kd tree is created on these constructed sequences in order to quickly
        find the best basic approximation for any unitary
        """

        # add in original names to reconstruct gate at end to check
        self.orig_names_to_gates = {}
        modified_set = []

        # used to make sure we do not add in the adjoints twice
        already_added = set()

        for gate in instruction_set:
            self.orig_names_to_gates[gate[0]] = gate[1]
            # first check if it is hermitiatian

            # was already added in along with the adjoint
            if gate[0] in already_added:
                continue

            is_hermitian = False

            # hermitian if dist to adjoint is small enough
            if self.matdist(gate[1], gate[1].getH()) < 0.00000001:
                is_hermitian = True
            modversion = self.put_into_su2(gate[1])
            modified_set.append((gate[0], modversion))
            if is_hermitian and self.matdist(modversion, modversion.getH() > 0.00000001):
                adj_version = modversion.getH()
                # append an exclmation point to the name to signify it is a new gate that
                #   just corresponds to a version of the gate
                modified_set.append((gate[0] + "!", adj_version))
            if not is_hermitian:
                # find the adjoiint and then add it in
                mindist = 100000
                my_adjoint = None
                for poss_adjoint in instruction_set:
                    cdist = self.matdist(poss_adjoint[1], gate[1].getH())
                    if cdist < mindist:
                        mindist = cdist
                        my_adjoint = poss_adjoint[0]
                already_added.add(my_adjoint)
                modified_set.append((my_adjoint, modversion.getH()))

        self.base_gates = []
        self.base_sequences = []

        # dict to store strings to their adjont strings
        self.adjoints = {}
        for gate in modified_set:
            mindist = 10000000
            my_adjoint = None
            for poss_adjoint in modified_set:
                cdist = self.matdist(poss_adjoint[1], gate[1].getH())
                if cdist < mindist:
                    mindist = cdist
                    my_adjoint = poss_adjoint[0]
            self.adjoints[gate[0]] = my_adjoint

        # specify max number of sequences allowed to be in the basic approx group
        # larger is better but slower. For small sets the below number seems
        #   to suffice for getting basic approx to be dense in SU(2)
        allowed_size = 1000000

        next_mats = []
        prev_mats = []
        #store as list of basic gates and then the corresponding matrix
        for item in modified_set:
            next_mats.append(([item[0]], item[1]))

        while len(self.base_gates) + len(next_mats)*len(modified_set) <= allowed_size:
            prev_mats.clear()
            for item in next_mats:
                self.base_sequences.append(item[0])
                self.base_gates.append(item[1])
            prev_mats.extend(next_mats)
            next_mats.clear()
            for item in prev_mats:
                for nxgate in modified_set:
                    if item[0][-1] == self.adjoints[nxgate[0]]:
                        continue
                    nlist = []
                    nlist.extend(item[0])
                    nlist.append(nxgate[0])
                    next_mats.append((nlist, item[1] @ nxgate[1]))

        # always do this because it will not overflow
        # done here to ensure we will always add something
        # (in case the input is pre-processed to be big)
        for item in next_mats:
            self.base_sequences.append(item[0])
            self.base_gates.append(item[1])
        # add in identity with specific signification
        self.base_gates.append(np.identity(2, dtype=complex))
        self.base_sequences.append(["id"])
        self.adjoints["id"] = "id"

        #objects are stored as 4-d point, then matrix
        obj_list = []
        cur_string_map = {}

        for index in range(len(self.base_gates)):
            mat = self.base_gates[index]
            coords = (mat.item(0, 0).real, mat.item(0, 1).imag, \
                mat.item(1, 0).real, mat.item(1, 1).imag)
            obj_list.append((coords, mat, index))
            cstring = ""
            for val in self.base_sequences[index]:
                cstring += val + ";"
            cur_string_map[cstring] = index

        self.base_adjoints = [None] * (len(self.base_sequences))
        for index in range(len(self.base_sequences)):
            c_adj_string = ""
            for val in reversed(self.base_sequences[index]):
                c_adj_string += self.adjoints[val] + ";"
            self.base_adjoints[index] = cur_string_map[c_adj_string]

        # initialize kdtree size so we can use binary tree in array trick
        self.kdtree = [None] * (len(self.base_gates)*2+5)
        self.build_tree(obj_list)

    def put_into_su2(self, gate):
        """Cast an arbitrary unitary into SU(2)
        Scale the unitary so that it has determinant 1
        For consistency, ensure the top left entry has positive real part
        """

        mat = np.multiply(gate, 1.0/LA.det(gate) ** 0.5)
        if mat.item(0, 0).real < 0:
            mat = np.multiply(mat, -1.0)
        return mat

    def build_tree(self, objects, axis=0, index=1):
        """ Recursive function for building the kd-tree on a sequence of gates
        Split each level on one of the coordinates (cycle through which coordinate you split on)
        Recurse on each side that is split
        Construction time should be O(nlogn) for n objects
        """

        if not objects:
            return
        objects.sort(key=lambda x: x[0][axis])
        median = len(objects) // 2

        next_axis = (axis+1)%4

        self.kdtree[index] = objects[median]
        self.build_tree(objects[:median], next_axis, 2*index)
        self.build_tree(objects[median+1:], next_axis, 2*index+1)

    def nearest_neighbor(self, point):
        """ Find the nearest neighbor to some 4-dimensional coordinate in the kd-tree
        Store the closest point seen so far.
        Often, one side of the recursion will be such that all points are further away
        from the goal than the closest point seen so far. In that case, do not recurse

        This is certainly the bottleneck of the whole process. Changing the data structure
        might be worth considering.
        """

        # best sequence encountered so far
        # initialize to a 'None' tuple with a placeholder distance
        best_seq = [(None, None, -1), 0.0]

        def point_dist(pt1, pt2):
            return (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + \
                (pt1[2]-pt2[2])**2 + (pt1[3]-pt2[3])**2

        def recurse_kd(index, point, axis=0):
            if index >= len(self.kdtree):
                return
            node = self.kdtree[index]
            if node is None:
                return
            if best_seq[0][0] is None or point_dist(node[0], point) < best_seq[1]:
                best_distance = point_dist(node[0], point)
                best_seq[:] = [node, best_distance]

            plane_distance = (point[axis] - node[0][axis]) ** 2
            close_side, far_side = (2*index, 2*index+1) if point[axis] < node[0][axis] \
                else (2*index+1, 2*index)
            recurse_kd(close_side, point, (axis+1)%4)
            if plane_distance < best_seq[1]:
                recurse_kd(far_side, point, (axis+1)%4)
        recurse_kd(1, point)

        return best_seq[0][2] # return the index of the matrix that is the best

    def matdist(self, mat1, mat2):
        """ Simple function to calculate trace distance between two matrices.
        Used to measure how good each approximation is
        """

        diff = np.asmatrix(np.subtract(mat1, mat2))

        prod = np.matmul(diff.getH(), diff)
        square_root_prod = sqrtm(prod)

        return ((square_root_prod[0][0] + square_root_prod[1][1]).real)/2.0

    def basic_approx(self, gate):
        """ Return a basic approximation to the input unitary.
        Formally, for the algorithm to work this has to give a suitable close
        approximation (however, it is unclear exactly what the barrier for that is)

        The function just runs a query to the kd-tree for the input sequences of gates
        generated in the __init__ function. The distance metric used to find the nearest neighbor
        used is a bit different than that in matdist. This is done so that the kd-tree can
        be searched quickly.
        """

        res = self.nearest_neighbor((gate.item(0, 0).real, gate.item(0, 1).imag, \
            gate.item(1, 0).real, gate.item(1, 1).imag))
        res2 = self.nearest_neighbor((-gate.item(0, 0).real, -gate.item(0, 1).imag, \
            -gate.item(1, 0).real, -gate.item(1, 1).imag))
        if self.matdist(self.base_gates[res2], gate) < self.matdist(self.base_gates[res], gate):
            return [res2]
        return [res]

    def mult_named_gates(self, gates):
        """ Multiply gates based on their names that were originall passed to __init__

        This function is really used to calculate the unitary corresponding the output
        the algorithm returns (helpful for seeing the unitary actually created)
        """
        mat_list = []
        for item in gates:
            mat_list.append(self.orig_names_to_gates[item])
        return self.mult_matrix(mat_list)

    def mult_matrix(self, matrices):
        """ Simple function to return a product of a list of matrices"""

        res = np.identity(len(matrices[0]))
        for val in matrices:
            res = np.matmul(res, val)
        return res

    def gc_decompose(self, gate):
        """ Implementation of group-commutator decompose function of sk algorithm
        Takes an input unitary and return two gates V and W such that V*W*V-adj*W-adj
        is equal to the input gate. This is done because recursively approximating V and W
        will allow some error approximations to cancel and give us a better approximation to
        the input gate than if we just directly recursively approximated the input gate
        """

        citem = gate.item(0, 0).real
        if abs(citem) > 1.000000000001:
            pass # this is indicative of error elsewhere
        if citem < -1.0:
            citem = -1.0
        if citem > 1.0:
            citem = 1.0

        theta = 2.0*math.acos(citem)
        sin_theta = math.sin(theta/2.0)

        phi = 2.0 * math.asin(math.sqrt(math.sqrt(0.5*(1-math.sqrt(1-sin_theta*sin_theta)))))

        # the following variables correspond to V and W in the SK paper
        v_rotate = RXGate(phi)
        w_rotate = RYGate(phi)

        v_matrix = np.asmatrix(v_rotate.to_matrix())
        w_matrix = np.asmatrix(w_rotate.to_matrix())

        # compute the middle part where U = S*(mid)*(S-dagger)
        mid = self.mult_matrix([v_matrix, w_matrix, v_matrix.getH(), w_matrix.getH()])

        # diagonalize input gate
        gate_diag_array, gate_evec_array = schur(gate)
        gate_diag = np.asmatrix(gate_diag_array)
        gate_evec = np.asmatrix(gate_evec_array)

        # diagonalize middle product
        mid_diag_array, mid_evec_array = schur(mid)
        mid_diag = np.asmatrix(mid_diag_array)
        mid_evec = np.asmatrix(mid_evec_array)

        # S is the matrix such that the input gate = S * (mid) * S-dag
        s_mat = np.matmul(gate_evec, mid_evec.getH())
        if abs(gate_diag.item(0, 0)-mid_diag.item(0, 0)) > 0.0000001:
            # matrix used to flip eigenvalues
            # the matrices are guaranteed to be similar but the decomposition might
            #   permute the eigenvalues
            flipmat = np.matrix([[0, 1], [1, 0]])
            s_mat = self.mult_matrix([gate_evec, flipmat, mid_evec.getH()])

        # the tilde versions of V and W are the final, correct group commutator
        v_tilde = self.mult_matrix([s_mat, v_matrix, s_mat.getH()])
        w_tilde = self.mult_matrix([s_mat, w_matrix, s_mat.getH()])

        return (v_tilde, w_tilde)

    def list_adjoint(self, gate_list):
        """ Return the adjoint of a list of gate sequences
        This is done so that we can reconstruct the gate sequence when the algorithm is done
        """
        res = []
        for gate in gate_list:
            # need better way of getting the inverse for some sequence
            # we know that the adjoint exists in the gate set - we could just dig it up for now
            gate_adj = self.base_adjoints[gate]
            res.append(gate_adj)
        res.reverse()
        return res

    def get_gates(self, gate_list):
        """ The base sequences generated are passed around as numbers in general
        in order to prevent passing around massive gate sequences.

        This function returns the bigger gate sequences that correspond to some sequence
        of these base-sequence numbers.
        """
        res = []
        for gate in gate_list:
            res.append(self.base_gates[gate])
        return res

    def recurse(self, gate, depth, cache=None):
        """ Recursively approximate some gate.
        Larger depth leads to better approximations and also longer sequences.
        cache is a potentially pre-stored gate meant to correspond to recurse(gate, depth-1)
        This is done so we don't re-recurse if we have previously done so
        """

        if depth == 0:
            retval = self.basic_approx(gate)

            return retval

        gate_prev = cache
        if cache is None:
            gate_prev = self.recurse(gate, depth-1)

        #figure out how to do below
        tot_list = []
        tot_list.append(gate)

        tot_list.extend(self.get_gates(self.list_adjoint(gate_prev)))

        # U * U_{n-1}-dagger
        tempo = self.mult_matrix(tot_list)

        decomposition = self.gc_decompose(tempo)
        dec_v = decomposition[0]
        dec_w = decomposition[1]

        dec_v_prev = self.recurse(dec_v, depth-1)
        dec_w_prev = self.recurse(dec_w, depth-1)

        gate_approx = []
        gate_approx.extend(dec_v_prev)
        gate_approx.extend(dec_w_prev)
        gate_approx.extend(self.list_adjoint(dec_v_prev))
        gate_approx.extend(self.list_adjoint(dec_w_prev))
        gate_approx.extend(gate_prev)

        return gate_approx

    def run(self, gate, depth=-1, epsilon=None):
        """ Approximate input gate to some error epsilon or with some recursive depth
        Specifying epsilon will allow the algorithm to recurse until the output sequence is
        within some desired error of the input unitary.
        """

        gate = self.put_into_su2(gate)

        recursive_result = None
        if depth == -1:
            if epsilon is None:
                # this should actually raise an error
                pass
            cur_depth = 0
            cur_res = self.recurse(gate, 0)
            while self.matdist(gate, self.mult_matrix(self.get_gates(cur_res))) > epsilon:
                cur_depth = cur_depth+1
                # pass in previous approximation so we don't have to do calculation twice
                cur_res = self.recurse(gate, cur_depth, cur_res)

            recursive_result = cur_res
        else:
            # case when specified depth is not -1
            recursive_result = self.recurse(gate, depth)

        # modify this back to something we can return
        remove_cancels = []
        for compound_gate in recursive_result:
            for item in self.base_sequences[compound_gate]:
                if item == "id":
                    # eliminate all identity matrices
                    continue
                if len(remove_cancels) == 0:
                    remove_cancels.append(item)
                    continue
                if remove_cancels[-1] == self.adjoints[item]:
                    # cancel out the two items if different
                    remove_cancels.pop()
                    continue
                # add the new gate in because it does not cancel anything out
                remove_cancels.append(item)
        ret = []
        for item in remove_cancels:
            if item[-1] == '!':
                ret.append(item[:-1])
            else:
                ret.append(item)
        return ret
