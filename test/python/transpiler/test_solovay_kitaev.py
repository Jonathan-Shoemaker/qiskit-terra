"""Test Solovay Kitaev synthesis"""

import numpy as np
from qiskit.transpiler.synthesis import SolovayKitaev
from qiskit.test import QiskitTestCase
from qiskit.circuit.library import (U3Gate, HGate, TGate, RXGate)

class TestSolovayKitaev(QiskitTestCase):
    """Test the Solovay-Kitaev Algorithm Implementation."""

    def test_three_gates(self):
        """Test the simplest set of three gates universal for single qubits"""

        simple_set = []
        simple_set.append(("h", np.asmatrix(HGate().to_matrix())))
        simple_set.append(("t", np.asmatrix(TGate().to_matrix())))
        simple_set.append(("tdag", np.asmatrix(TGate().to_matrix()).getH()))
        simple_sk = SolovayKitaev(simple_set)

        # any three doubles below should work
        random_gate = np.asmatrix(U3Gate(.1341324, .6141, .234).to_matrix())

        sequence = simple_sk.run(random_gate, epsilon=1e-5)
        resulting_product = simple_sk.mult_named_gates(sequence)

        # measure distance, ignoring global phase
        distance = simple_sk.matdist(simple_sk.put_into_su2(random_gate),  \
            simple_sk.put_into_su2(resulting_product))
        self.assertTrue(distance < 1e-5)

    def test_h_with_random(self):
        """Tes group of hadmard along with some other random gate"""

        simple_set = []
        simple_set.append(("h", np.asmatrix(HGate().to_matrix())))
        simple_set.append(("r", np.asmatrix(U3Gate(.1241324, .16412, 13413).to_matrix())))
        simple_set.append(("rdag", np.asmatrix(U3Gate(.1241324, .16412, 13413).to_matrix()).getH()))
        simple_sk = SolovayKitaev(simple_set)

        # test a small rotation
        random_gate = np.asmatrix(RXGate(3.14159265358/32).to_matrix())

        sequence = simple_sk.run(random_gate, epsilon=1e-5)
        resulting_product = simple_sk.mult_named_gates(sequence)

        # measure distance, ignoring global phase
        distance = simple_sk.matdist(simple_sk.put_into_su2(random_gate),  \
            simple_sk.put_into_su2(resulting_product))
        self.assertTrue(distance < 1e-5)
