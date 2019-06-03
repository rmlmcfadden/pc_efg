#!/usr/bin/python3

# command line argument parser
from argparse import ArgumentParser

# tools for creating the supercell using the .cif as input
from ase.atoms import Atoms
from ase.build import find_optimal_cell_shape_pure_python, make_supercell
from ase.geometry import get_duplicate_atoms
from ase.io import read
from ase.spacegroup import crystal

# working with arrays
import numpy as np

# physical constants needed in the calculations
from scipy.constants import physical_constants

# read the yaml input files
import yaml


# calculate the contribution to the (cartesian) EFG tensor in V/A^2
def EFG(q_1, position_1, q_2, position_2, r_cutoff):
    # define some constants
    k_e = 1.0 / 4.0 / np.pi / physical_constants["electric constant"][0]
    q_e = physical_constants["elementary charge"][0]
    A3_to_m3 = 1e-30
    A2_to_m2 = 1e-20
    C = k_e * q_e * q_1 * q_2 * A2_to_m2 / A3_to_m3

    # calculate the distance between the two charges
    distance = np.array(position_1) - np.array(position_2)
    x, y, z = distance
    r = np.linalg.norm(distance)

    # cartesian EFG tensor
    V = np.zeros((3, 3))

    # only add the contribution from charges within the cutoff radius
    if r <= r_cutoff:
        # cartiasian components
        V_xx = C * (3 * x * x - 1 * r ** 2) / r ** 5
        V_xy = C * (3 * x * y - 0 * r ** 2) / r ** 5
        V_xz = C * (3 * x * z - 0 * r ** 2) / r ** 5
        V_yx = C * (3 * y * x - 0 * r ** 2) / r ** 5
        V_yy = C * (3 * y * y - 1 * r ** 2) / r ** 5
        V_yz = C * (3 * y * z - 0 * r ** 2) / r ** 5
        V_zx = C * (3 * z * x - 0 * r ** 2) / r ** 5
        V_zy = C * (3 * z * y - 0 * r ** 2) / r ** 5
        V_zz = C * (3 * z * z - 1 * r ** 2) / r ** 5
        # put them in an array
        V = np.array([[V_xx, V_xy, V_xz], [V_yx, V_yy, V_yz], [V_zx, V_zy, V_zz]])

    return V


# calculate then eigenvalues/vectors (l and v) for a real/hermitian symmetric
# matrix M and return the results sorted in the NMR literature convention for
# EFG tensors (i.e., |V_11| <= |V_22| <= |V_33|)
# https://stackoverflow.com/a/50562995
def eigh_sorted(M):
    evals, evecs = np.linalg.eigh(M)
    # sort based on ascending magnitude
    sort_order = np.argsort(evals ** 2)
    evals = evals[sort_order]
    evecs = evecs[:, sort_order]
    return (evals, evecs)


# EFG asymmetry parameter
def efg_asymmetry(evals):
    return (evals[0] - evals[1]) / evals[2]


# calculate the quadrupole coupling constant C_q (Hz)
# efg = the principal component of the EFG (V/A^2)
# Q = nuclear electric quadrupole moment (mb)
# antishielding_factor = Sternhiemer antishielding factor for ion (unitless)
def quadrupole_coupling(efg, Q, antishielding_factor=0.0):
    shielding = 1.0 - antishielding_factor
    e = physical_constants["elementary charge"][0]
    h = physical_constants["Planck constant"][0]
    eq = np.max(np.abs(efg))
    return shielding * e * (eq * 1e20) * (Q * 1e-31) / h


# https://stackoverflow.com/a/13849249
def unit_vector(v):
    return v / np.linalg.norm(v)


#
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


# calculate the polar angles betwen the EFG PAS and the applied field
def polar_angles(V, B):
    theta = 0
    phi = 0
    return (theta, phi)


# 1st order angular correction for the quadrupole frequency
def angular_factor(theta, phi, eta):
    return 0.5 * (
        np.cos(theta) * np.cos(theta)
        - 1.0
        + eta * np.sin(theta) * np.sin(theta) * np.cos(2 * phi)
    )


# calculate the quadrupole frequency
def quadrupole_frequency(
    eq, Q, I, theta=0.0, phi=0.0, eta=0.0, antishielding_factor=0.0
):
    shielding = 1.0 - antishielding_factor
    C_q = quadrupole_coupling(eq, Q, antishielding_factor)
    f_1 = angular_factor(theta, phi, eta)
    return 3.0 * C_q * f_1 * shielding / (4.0 * I * (2.0 * I - 1.0))


# main routine

if __name__ == "__main__":
    # setup the parser
    parser = ArgumentParser(
        prog="pc_efg",
        description="pc_efg: an EFG calculator for an isolated impurity in a (point charge) crystal lattice",
        epilog="Copyright (c) 2019 Ryan M. L. McFadden",
    )
    # positional arguments
    parser.add_argument(
        "yaml_input",
        help=".yaml input file specifying the calculation details",
        type=str,
    )
    # optional arguments
    parser.add_argument("-v", "--version", action="version", version="%(prog)s v0.1")
    # parse the arguments
    args = parser.parse_args()
    # read the input control file
    with open(args.yaml_input, "r") as fh:
        ctl = yaml.load(fh, Loader=yaml.SafeLoader)

    # make the superlattice
    cif = read(ctl["lattice"]["cif"])
    xtal = crystal(cif, size=(1, 1, 1))

    impurity_position = xtal.cell.dot(np.array(ctl["impurity"]["position"]))

    n_x, n_y, n_z = ctl["lattice"]["size"]

    # create the 8 combinations of transformation matricies
    p1 = [[n_x, 0, 0], [0, n_y, 0], [0, 0, n_z]]
    p2 = [[-n_x, 0, 0], [0, -n_y, 0], [0, 0, -n_z]]
    p3 = [[-n_x, 0, 0], [0, n_y, 0], [0, 0, n_z]]
    p4 = [[n_x, 0, 0], [0, -n_y, 0], [0, 0, n_z]]
    p5 = [[n_x, 0, 0], [0, n_y, 0], [0, 0, -n_z]]
    p6 = [[-n_x, 0, 0], [0, -n_y, 0], [0, 0, n_z]]
    p7 = [[n_x, 0, 0], [0, -n_y, 0], [0, 0, -n_z]]
    p8 = [[-n_x, 0, 0], [0, n_y, 0], [0, 0, -n_z]]

    p_transform = [p1, p2, p3, p4, p5, p6, p7, p8]

    # transform the unit cell into a supercell using the tranformation matricies
    sc = Atoms()
    for p in p_transform:
        sc += make_supercell(xtal, p)

    # remove any duplicate atoms
    get_duplicate_atoms(sc, cutoff=0.1, delete=True)

    # set the charge of each atom
    initial_charges = [
        ctl["lattice"]["charges"][cs] for cs in sc.get_chemical_symbols()
    ]
    sc.set_initial_charges(initial_charges)

    # calculate the EFG
    V = np.zeros((3, 3))
    for p, q in zip(sc.get_positions(), sc.get_initial_charges()):
        V += EFG(
            q,
            p,
            ctl["impurity"]["charge"],
            impurity_position,
            ctl["calculation"]["cutoff_radius"],
        )

    # calculate the eigenvalues/eigenvectors
    l, v = eigh_sorted(V)

    # calculate NMR quantities derived from the EFG
    eta = efg_asymmetry(l)
    C_q = quadrupole_coupling(
        l, ctl["impurity"]["quadrupole_moment"], ctl["impurity"]["antishielding_factor"]
    )
    theta, phi = polar_angles(V, ctl["calculation"]["magnetic_field"])
    nu_q = quadrupole_frequency(
        l,
        ctl["impurity"]["quadrupole_moment"],
        ctl["impurity"]["spin"],
        theta,
        phi,
        eta,
        ctl["impurity"]["antishielding_factor"],
    )

    # add the results to the dictionary
    ctl["results"] = {}
    ctl["results"]["EFG"] = {}
    ctl["results"]["EFG"]["tensor (V/A^2)"] = V.tolist()
    ctl["results"]["EFG"]["eigenvalues (V/A^2)"] = l.tolist()
    ctl["results"]["EFG"]["eigenvectors"] = v.tolist()
    ctl["results"]["C_q (Hz)"] = float(C_q)
    ctl["results"]["nu_q (Hz)"] = float(nu_q)
    ctl["results"]["theta"] = float(theta)
    ctl["results"]["phi"] = float(phi)
    ctl["results"]["eta"] = float(eta)

    # prune all the unused charges from the dictionary
    unique_symbols = list(set(sc.get_chemical_symbols()))
    for atom in list(ctl["lattice"]["charges"]):
        is_same = [atom == s for s in unique_symbols]
        is_used = np.any(is_same)
        if is_used == False:
            del ctl["lattice"]["charges"][atom]

    # print the results to the terminal if no output is specified
    if ctl["calculation"]["output_file"] == None:
        print(yaml.dump(ctl["results"], default_flow_style=False))
    # or write the updated dictionary to the output_file
    else:
        with open(ctl["calculation"]["output_file"], "w") as fh:
            yaml.dump(ctl, fh, default_flow_style=False)
    # all done
