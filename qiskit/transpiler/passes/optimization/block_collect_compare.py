""" Try to test different block collection algorithms
"""


import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CollectMultiQBlocks
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.transpiler.passes import AlternateCollect
from qiskit.test import QiskitTestCase

from numpy.random import seed
from numpy.random import rand
from numpy.random import randint
import time
from qiskit.circuit.random import random_circuit


def do_random_gate(circ_size, qc, gate_size):
	# must give me an odd size
	nums = np.random.choice(circ_size, gate_size, replace=False)
	x = (gate_size + 1)/2
	controll_bits = []
	targ_bit = []
	ancilla_bits = []
	for i in range(gate_size):
		if i < x:
			controll_bits.append(nums[i])
		elif i == x:
			targ_bit = nums[i]
		else:
			ancilla_bits.append(nums[i])
	if gate_size > 2:
		qc.mct(controll_bits, targ_bit, ancilla_bits, 'v-chain')


def do_random():
	circ_size = 20
	qc = QuantumCircuit(circ_size)

	
	# below is pure random cx gates
	
	for _ in range(100):
		vals = randint(0, circ_size, 2)
		if vals[0] == vals[1]:
			pass
			# qc.h(vals[0])
		else:
			qc.cx(vals[0], vals[1])
	
	# for i in range(100):
	# 	do_random_gate(circ_size, qc, 3)


	pass_manager= PassManager()	
	pass_manager.append(AlternateCollect(max_block_size=4))

	# pass_manager2 = PassManager()
	# pass_manager2.append(Collect2qBlocks())

	pass_manager.run(qc)    
	# pass_manager2.run(qc)

	# if (len(pass_manager.property_set['block_list']) != len(pass_manager2.property_set['block_list'])):
	# 	print(qc.draw())
	# 	print("New size:", len(pass_manager.property_set['block_list']))
	# 	print("Old size:", len(pass_manager2.property_set['block_list']))

	return len(pass_manager.property_set['block_list'])
	# return [len(pass_manager.property_set['block_list']), len(pass_manager2.property_set['block_list'])]  

def built_in():
	qc = random_circuit(num_qubits=30, depth=50, max_operands=3, conditional=True, reset=True)
	pass_manager = PassManager()
	pass_manager.append(AlternateCollect(max_block_size=2))

	pass_manager2 = PassManager()
	pass_manager2.append(CollectMultiQBlocks(max_block_size=2))
	# pass_manager.append(Collect2qBlocks())

	pass_manager.run(qc)
	pass_manager2.run(qc)
	
	if len(pass_manager.property_set['block_list']) != len(pass_manager2.property_set['block_list']):
		print(qc.draw())
		print("Vals: ", len(pass_manager.property_set['block_list']), len(pass_manager2.property_set['block_list']))
		# pass_manager3 = PassManager()
		# pass_manager3.append(CollectMultiQBlocks(max_block_size=2, printout=True))
		# pass_manager3.run(qc)
		for block in pass_manager.property_set['block_list']:
			print("list:", end=" ")
			for val in block:
				print(val.name, end=" ")
			print()
		print("--------")
		for block in pass_manager2.property_set['block_list']:
			print("list:", end=" ")
			for val in block:
				print(val.name, end=" ")
			print()
	
		# pass_manager3 = PassManager()
		# pass_manager3.append(Collect2qBlocks())
		# pass_manager3.run(qc)
		# print("FIRST VAL: ", len(pass_manager3.property_set['block_list']))

	return (len(pass_manager.property_set['block_list']), len(pass_manager2.property_set['block_list']))


sum1 = 0
sum2 = 0
startTime = int(round(time.time()*1000))
seed(23)
for i in range(100):
	ngates = built_in()
	sum1 += ngates[0]
	sum2 += ngates[1]
	# sum2 += ngates[1]
# print(sum1)
endtime = int(round(time.time()*1000))
print("Millis: ", endtime-startTime)
print("Sums (first is Alternate: ", sum1, sum2)



