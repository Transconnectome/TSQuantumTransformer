import itertools
from math import log2

import torch
import torchquantum as tq


def sim14_encoder(n_wires, layers=1):
    enc = []
    counter = itertools.count(0)
    for _ in range(layers):
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crx', 'wires': [i, (i + 1) % n_wires]}
                   for i in range(n_wires - 1, -1, -1)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'ry', 'wires': [i]} for i in range(n_wires)])
        enc.extend([{'input_idx': [next(counter)], 'func': 'crx', 'wires': [i, (i - 1) % n_wires]}
                    for i in [n_wires - 1] + list(range(n_wires - 1))])
    return enc


def evaluate_polynomial_state(base_states, unitary_params, enc, qdev, n_qbs, lcu_coeffs, poly_coeffs):
    acc = poly_coeffs[0] * base_states
    working_register = base_states

    for c in poly_coeffs[1:]:
        working_register = apply_unitaries(working_register, unitary_params, enc, qdev, n_qbs, lcu_coeffs)
        acc = acc + c * working_register

    return acc / torch.linalg.vector_norm(poly_coeffs, ord=1)


def apply_unitaries(base_states, unitary_params, enc, qdev, n_qbs, coeffs):
    repeated_base = base_states.repeat(1, unitary_params.shape[1]).view(-1, 2 ** n_qbs)
    qdev.set_states(repeated_base)
    enc(qdev, unitary_params.view(-1, unitary_params.shape[-1]))
    states = qdev.get_states_1d().view(*unitary_params.shape[:2], 2 ** n_qbs)
    lcs = torch.einsum('bwi,bw->bi', states, coeffs)
    return lcs


class QuixerTimeSeries(torch.nn.Module):
    def __init__(self,
                 n_qubits: int,
                 n_timesteps: int,
                 degree: int,
                 n_ansatz_layers: int,
                 feature_dim: int,
                 output_dim: int,
                 dropout: float,
                 device):
        """
        n_qubits: int
            Number of qubits per timestep.
        n_timesteps: int
            Length of the time-series sequence.
        degree: int
            Degree of polynomial.
        n_ansatz_layers: int
            Number of layers of circ 14.
        feature_dim: int
            Number of features per timestep.
        output_dim: int
            Size of the final output layer.
        dropout: float
            Dropout rate.
        device:
            Torch device.
        """

        super().__init__()

        self.n_timesteps = n_timesteps
        self.n_qubits = n_qubits
        self.degree = degree
        self.device = device

        assert n_timesteps != 0
        self.n_ctrl_qubits = int(log2(n_timesteps))

        # Sim14 spec
        self.n_rots = 4 * n_qubits * n_ansatz_layers

        # Feature projection instead of embedding
        self.feature_projection = torch.nn.Linear(feature_dim, self.n_rots)

        self.dropout = torch.nn.Dropout(dropout)
        self.rot_sigm = torch.nn.Sigmoid()

        self.q_device = tq.QuantumDevice(n_wires=self.n_qubits)

        # Preparation of timestep unitaries
        self.timestep_qencoder = tq.GeneralEncoder(sim14_encoder(n_qubits, n_ansatz_layers))
        self.timestep_qencoder.n_wires = self.n_qubits

        self.n_poly_coeffs = self.degree + 1
        self.poly_coeffs = torch.nn.Parameter(torch.rand(self.n_poly_coeffs))
        self.mix_coeffs = torch.nn.Parameter(torch.rand(self.n_timesteps, dtype=torch.complex64))

        self.qff = tq.GeneralEncoder(sim14_encoder(n_qubits))
        self.qff_params = torch.nn.Parameter(torch.rand(self.n_rots))

        self.measure_all_xyz = tq.MeasureMultipleTimes(
            [{'wires': range(n_qubits), 'observables': ['x'] * n_qubits},
             {'wires': range(n_qubits), 'observables': ['y'] * n_qubits},
             {'wires': range(n_qubits), 'observables': ['z'] * n_qubits}])

        self.n_measures = 3 * n_qubits

        self.output_ff = torch.nn.Linear(self.n_measures, output_dim)

    def forward(self, x):
        bsz = x.shape[0]

        # Normalize and project features
        x = self.feature_projection(self.dropout(x))

        timestep_params = self.rot_sigm(x)

        base_states = torch.zeros(bsz, 2 ** self.n_qubits, dtype=torch.complex64, device=self.device)
        base_states[:, 0] = 1.0

        mixed_timestep = evaluate_polynomial_state(base_states,
                                                   timestep_params,
                                                   self.timestep_qencoder,
                                                   self.q_device,
                                                   self.n_qubits,
                                                   self.mix_coeffs.repeat(bsz, 1),
                                                   self.poly_coeffs)

        final_probs = torch.linalg.vector_norm(mixed_timestep, dim=-1)

        self.q_device.set_states(torch.nn.functional.normalize(mixed_timestep, dim=-1))
        self.qff(self.q_device, self.qff_params.repeat(1, bsz))

        exps = self.measure_all_xyz(self.q_device)
        exps = exps.reshape(3, bsz, self.n_qubits).moveaxis(0,1).reshape(bsz, -1)

        op = self.output_ff(exps)
        return op, torch.mean(final_probs)

    