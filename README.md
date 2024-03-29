# `pc_efg`

`pc_efg`: an [EFG] calculator for an isolated impurity in a (point charge) crystal lattice.

A key observable in many [β-NMR] experiments is the quadrupole splitting of the [NMR] line, which is proportional to the [EFG] at the site of the implanted hyperfine probe. Unique to the technique (i.e., distinct from "conventional" [NMR]), the probe is really isolated and should be treated accordingly in the simulation. This project, `pc_efg`, aims to make simulating such situations, for both arbitrary _probe ion_ and _crystal structure_, painless. Moreover, `pc_efg` uses standard `.yaml` markup for both its input/outpt, allowing for easy generation/processing and further aleviating the tedium of the task!

# Installation

`pc_efg` is just a simple [Python] script which uses other libraries to do all the heavy lifting (e.g., parsing the input, creating the supercell etc.). The core dependencies are:

- [ASE]
- [NumPy]
- [PyYAML]

One possible way to obtain these is through your system's package manager. This is accomplished, for example, on [Fedora] with:

```
dnf install python3-numpy python3-ase python3-pyyaml
```

Alternatively, if the required packages are unavailable, `pip` provides a convenient mechanism to obtain (and maintain) them:

```
pip3 install --user numpy
pip3 install --user ase
pip3 install --user pyyaml
```

With all the pieces in hand, `pc_efg` is ready for action!

For convenience, a `Makefile` is included for lazy integration into [Unix-like] systems. Installation is as easy as:

```
make install
```

Note that (as per usual) the default install location `/usr/local/bin` requires superuser privilages for access.

# Use

All `pc_efg` requires as input as a `.yaml` file containing instructions for the simulation. This includes:

- the location and properties of the [NMR] probe
- the `.cif` file defining the lattice and the size of the desired supercell
- the ionic charge of each element in the lattice

See `input.yaml` as an example of a typical control file. Running a calculation is simply:

```
python3 pc_efg.py input.yaml
```

or (if you ran `make install`):

```
pc_efg input.yaml
```

In all likelihood, you'll want to do more than a single calculation and stich together the results from several simulations. The `Python` script `example.py` demonstrates a possible use case.

# Caveats

Currently, it is not possible to deal with cases where the same element in a lattice has different oxidation states (e.g., at different sites).

[fedora]: https://getfedora.org/
[unix-like]: https://en.wikipedia.org/wiki/Unix-like
[python]: https://www.python.org/
[efg]: https://en.wikipedia.org/wiki/Electric_field_gradient
[nmr]: https://en.wikipedia.org/wiki/Nuclear_magnetic_resonance
[ase]: https://wiki.fysik.dtu.dk/ase/
[β-nmr]: https://doi.org/10.1016/j.ssnmr.2015.02.004
[numpy]: https://www.numpy.org/
[pyyaml]: https://pyyaml.org/

