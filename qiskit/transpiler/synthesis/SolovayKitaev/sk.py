from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
import math
import numpy as np
from numpy import linalg as LA
from scipy.linalg import sqrtm, schur
from qiskit.circuit.library import RXGate, RYGate


class sk():

	def __init__(self, instruction_set):
		# want to generate the cross output here
		# want to return names of gates
		# assumes instruction set comes with a name
		# instruction set is tuple of name, gate (in form of a numpy matrix)
		# require that the adjoint of the gate is in this set

		# cast the original gates to a version that is in SU(2)

		# add in original names to reconstruct gate at end to check
		self.orig_names_to_gates = {}
		modified_set = []

		for gate in instruction_set: 
			self.orig_names_to_gates[gate[0]] = gate[1]
			# first check if it is hermitiatian
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
			# print('Inside version of ', gate[0], '\n', modversion)

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
			# print(gate[0], 'has adjoint of ', my_adjoint)

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
				for nxgate in modified_set :
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

		print("Total LENGTH: ", len(self.base_gates))

		#objects are stored as 4-d point, then matrix
		obj_list = []
		cur_string_map = {}
		# gate_id = 0
		for index in range(len(self.base_gates)):
			mat = self.base_gates[index]
			coords = (mat.item(0,0).real, mat.item(0,1).imag, \
				mat.item(1,0).real, mat.item(1,1).imag)
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

		print('Tree has been built')

	def put_into_su2(self, gate):
		mat = np.multiply(gate, 1.0/LA.det(gate) ** 0.5)
		if mat.item(0,0).real < 0:
			mat = np.multiply(gate, -1.0)
		return mat

	def build_tree(self, objects, axis = 0, index = 1):
		if not objects: 
			return None
		objects.sort(key=lambda x: x[0][axis])
		median = len(objects) // 2

		next_axis = (axis+1)%4

		self.kdtree[index] = objects[median]
		self.build_tree(objects[:median], next_axis, 2*index)
		self.build_tree(objects[median+1:], next_axis, 2*index+1)

	def nearest_neighbor(self, point):
		best = [None, 0.0]

		def point_dist(pt1, pt2):
			return (pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + \
				(pt1[2]-pt2[2])**2 + (pt1[3]-pt2[3])**2

		def recurse_kd(index, point, axis = 0):
			# print(node, point)
			if index >= len(self.kdtree):
				return 
			node = self.kdtree[index]
			if node is None:
				return 
			if best[0] is None or point_dist(node[0], point) < best[1]:
				best_distance = point_dist(node[0], point)
				best[:] = [node, best_distance]

			plane_distance = (point[axis] - node[0][axis]) ** 2
			close_side, far_side = (2*index, 2*index+1) if point[axis] < node[0][axis] \
				else (2*index+1, 2*index)
			recurse_kd(close_side, point, (axis+1)%4)
			if plane_distance < best[1]:
				recurse_kd(far_side, point, (axis+1)%4)
		recurse_kd(1, point)

		return best[0][2] # return the index of the matrix that is the best

	def matdist(self, mat1, mat2):

		diff = np.subtract(mat1,mat2)

		prod = np.matmul(diff.getH(), diff)
		sq = sqrtm(prod)

		return ((sq[0][0] + sq[1][1]).real)/2.0

	def basic_approx(self, gate):
		res = self.nearest_neighbor((gate.item(0,0).real, gate.item(0,1).imag, \
			gate.item(1,0).real, gate.item(1,1).imag))
		res2 = self.nearest_neighbor((-gate.item(0,0).real, -gate.item(0,1).imag, \
			-gate.item(1,0).real, -gate.item(1,1).imag))
		if self.matdist(self.base_gates[res2], gate) < self.matdist(self.base_gates[res], gate):
			return [res2]
		# print('Res', LA.det(res))
		return [res]

	def mult_named_gates(self, gates):

		#mostly used for checking the result product
		mat_list = []
		for item in gates:
			mat_list.append(self.orig_names_to_gates[item])
		return self.mult_matrix(mat_list)

	def mult_matrix(self, matrices):
		res = np.identity(len(matrices[0]))
		for val in matrices:
			res = np.matmul(res, val)
		return res

	def gc_decompose(self, gate):

		citem = gate.item(0,0).real
		if abs(citem) > 1.000000000001 :
			print('Off by A ton')
		if citem < -1.0 : 
			citem = -1.0
		if citem > 1.0 : 
			citem = 1.0

		theta = 2.0*math.acos(citem)
		sin_theta = math.sin(theta/2.0)

		phi = 2.0 * math.asin(math.sqrt(math.sqrt(0.5*(1-math.sqrt(1-sin_theta*sin_theta)))))

		V = RXGate(phi)
		W = RYGate(phi)

		Vm = np.asmatrix(V.to_matrix())
		Wm = np.asmatrix(W.to_matrix())

		# compute the middle part where U = S*(mid)*(S-dagger)
		mid = np.matmul(Vm, np.matmul(Wm, np.matmul(Vm.getH(), Wm.getH())))

		# nangle = self.one_q_dec.angles(mid)[0]
		nangle = 2.0*math.acos(mid.item(0,0).real)
		# print('Nangle', nangle)

		# compute S 
		# print('Gate that is here: ', gate)
		# print('Gate det down here: ', LA.det(gate))
		uDm, uEm = schur(gate)
		uD = np.asmatrix(uDm)
		uE = np.asmatrix(uEm)

		kDm, kEm = schur(mid)
		kD = np.asmatrix(kDm)
		kE = np.asmatrix(kEm)
		# print('Mid decomp dets (kD, kE): ', LA.det(np.diag(kD)), LA.det(kE))
		prod2 = self.mult_matrix([kE, kD, kE.getH()])

		S = np.matmul(uE, kE.getH())
		if abs(uD.item(0,0)-kD.item(0,0)) > 0.0000001:
			# matrix used to fliop eigenvalues
			flipmat = np.matrix([[0,1],[1,0]])
			S = self.mult_matrix([uE, flipmat, kE.getH()])

		Vtilde = np.matmul(S, np.matmul(Vm, S.getH()))
		Wtilde = np.matmul(S, np.matmul(Wm, S.getH()))

		groupcom = self.mult_matrix([Vtilde, Wtilde, Vtilde.getH(), Wtilde.getH()])

		return (Vtilde, Wtilde)

	def list_adjoint(self, gate_list):
		res = []
		for gate in gate_list:
			# need better way of getting the inverse for some sequence
			# we know that the adjoint exists in the gate set - we could just dig it up for now
			gate_adj = self.base_adjoints[gate]
			res.append(gate_adj)
		res.reverse()
		return res

	def get_gates(self, gate_list):
		res = []
		for gate in gate_list:
			res.append(self.base_gates[gate])
		return res

	def recurse(self, gate, depth):

		if depth == 0:
			retval =  self.basic_approx(gate)

			return retval

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

	def run(self, gate, depth):
		# run on the unitary gate "gate"
		# calculate the needed depth (n) and then run
		# curdep = 0
		# while True:
		# 	tempo = self.recurse(gate, curdep)
		# 	if matdist(tempo, gate) < allowed_error:
		# 		return tempo
		# 	curdep = curdep+1
		recursive_result = self.recurse(gate, depth)

		nprod = self.mult_matrix(self.get_gates(recursive_result))
		print('Inner Dist: ', self.matdist(gate, nprod), '\n', nprod)

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
