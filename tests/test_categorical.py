# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import numpy as np

from pyhgf.model import HGF


class Testbinary(TestCase):
    def test_categorical_state_node(self):
        # generate some categorical inputs data
        input_data = np.array(
            [np.random.multinomial(n=1, pvals=[0.1, 0.2, 0.7]) for _ in range(3)]
        ).T

        # create the categorical HGF
        categorical_hgf = (
            HGF(model_type=None, verbose=False)
            .add_input_node(
                kind="categorical",
                categorical_parameters={"n_categories": 10},
                binary_parameters={"omega_2": -2.0},
            )
            .init()
        )

        # fitting the model forwards
        categorical_hgf.input_data(input_data=input_data.T)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
