#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from os import system
import yaml


# example use of pc_efg for rutile TiO2
# plot the EFG eigenvalues at different intersitial positions along the c-axis tunnels

position = np.linspace(0.0, 1.0, 101)
eta = []
V_11 = []
V_22 = []
V_33 = []

# get the "base" imput file
with open("input.yaml", "r") as fh:
    ctl = yaml.load(fh, Loader=yaml.SafeLoader)

# make the lattice small, just to speed things up!
ctl["lattice"]["size"] = [3, 3, 6]
ctl["calculation"]["cutoff_radius"] = 10.0

for p in position:
    # update the z position
    ctl["impurity"]["position"] = [0.0, 0.5, float(p)]
    
    # update the dictionary with temporary input/output filenames
    tmp_input = "p_%.3f_in.yaml" % p
    tmp_output = "p_%.3f_out.yaml" % p
    ctl["calculation"]["output_file"] = tmp_output
    
    # write a temporary yaml for feeding to pc_efg
    with open(tmp_input, "w") as fh:
        yaml.dump(ctl, fh, default_flow_style=False)

    # run pc_efg using a system call
    system("python3 pc_efg.py %s" % tmp_input)

    # read the results
    with open(tmp_output, "r") as fh:
        res = yaml.load(fh, Loader=yaml.SafeLoader)

    # add the results to lists
    eta.append(res["results"]["eta"])
    V_11.append(res["results"]["EFG"]["eigenvalues (V/A^2)"][0])
    V_22.append(res["results"]["EFG"]["eigenvalues (V/A^2)"][1])
    V_33.append(res["results"]["EFG"]["eigenvalues (V/A^2)"][2])

    # cleanup the temporary files
    system("rm %s" % tmp_input)
    system("rm %s" % tmp_output)


# plot the results!
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=False, constrained_layout=True)

ax1.plot(position, eta, "-")

ax1.set_ylabel(r"$\eta$")

ax2.plot(position, V_11, "-", label="$V_{11}$")
ax2.plot(position, V_22, "-", label="$V_{22}$")
ax2.plot(position, V_33, "-", label="$V_{33}$")

ax2.axhline(0, linestyle="--", color="gray", zorder=0)

ax2.set_xlabel("$^{8}$Li$^{+}$ position along $(0.0, 0.5, z / c)$")
ax2.set_ylabel(r"$V_{ij}$ (V $\mathrm{\AA}^{-2}$)")

ax2.legend(
    title="$^{8}$Li$^{+}$ EFG in rutile TiO$_{2}$ (with $|V_{11}| \leq |V_{22}| \leq |V_{33}|$)",
    ncol=3,
    loc="upper left",
    framealpha=1.0,
    bbox_to_anchor=(0.0, 1.5),
)

ax1.set_ylim(0, 1)
ax2.set_xlim(position.min(), position.max())

plt.savefig("example.pdf")

plt.show()
