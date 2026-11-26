# WHAT: A fast sanity check of the simulated qq->Zg data and of the data functions in Simulation_utils.py.
# WHY: Before a long L scan, make sure the data is read correctly (column order, units), the target belongs to the
#      momenta, and the coordinate transforms give finite, sensible numbers. Runs in a few seconds.

# HOW:
    # 1. The raw data: the shape, no nan or inf, and |M|^2 > 0 (the networks learn log10 |M|^2).
    # 2. The physics of the Cartesian columns: m_Z = 91.188 and m_g = 0 in every event, the incoming partons along the
    #    beam, and momentum conservation (q1 + q2 = Z + g).
    # 3. The target: |M|^2 of qq->Zg is proportional to (t^2 + u^2 + 2 s m_Z^2) / (t u), so the ratio of the two
    #    must be the same number in every event. This only holds if the target belongs to the momenta of its own row.
    # 4. The pT + angles transform: finite, the range of every column, and transformed back to Cartesian it must give
    #    the original momenta.
    # 5. The preparation of the scan for both bases: drop constant columns, split, standardize, and the Lipschitz
    #    estimate on a subsample.

# ======== IMPORTS ============#
import os
import numpy as np
from Simulation_utils import (get_simulated_data, cartesian_to_pt_angles, cartesian_to_logpt_angles,
                              logpt_angles_columns, drop_constant_columns, split_sim_data, standardize,
                              lipschitz_bound_estimate_of_data)

# ======== SETTINGS ============#
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_TO_SIM_DATA = os.path.join(SIM_DIR, "zg.npy")
M_Z = 91.188            # GeV, the Z mass of the generator
TOLERANCE_GEV = 1e-2    # GeV. The data is stored with a limited number of digits, so E^2 - p^2 of a massless gluon
                        # with E ~ 1000 GeV is not exactly 0: a mass of up to ~0.03 GeV comes from rounding alone.
TOLERANCE_REL = 1e-5    # relative, for the ratio of the target to the |M|^2 formula
N_EVENTS_L = 50000      # the Lipschitz estimate on a subsample, to be fast. NOT the L_data of the scan: with fewer
                        # events the neighbours lie further apart, which changes the estimate.

PT_ANGLES_COLUMNS = ["E_1", "E_2", "E_Z", "pT_Z", "phi_Z", "y_Z", "E_g", "pT_g", "phi_g", "y_g"]

failures = [] # The names of the checks that failed, printed at the end


def report(description, passed):
    # One line per check: OK or FAILED, and the failed ones are remembered for the summary at the end
    if passed:
        print(f"  OK      {description}")
    else:
        print(f"  FAILED  {description}")
        failures.append(description)


def minkowski_square(p):
    # p: (events, 4) with the columns (E, px, py, pz). Returns E^2 - px^2 - py^2 - pz^2 for every event:
    # the mass squared of a particle, or s, t, u for a sum or difference of two particles
    return p[:, 0]**2 - p[:, 1]**2 - p[:, 2]**2 - p[:, 3]**2


# ======== 1. THE RAW DATA ============#
print("1. The raw data")
data, target = get_simulated_data(PATH_TO_SIM_DATA)
print(f"  {data.shape[0]} events, {data.shape[1]} input columns")
report("16 input columns: four particles with (E, px, py, pz)", data.shape[1] == 16)
report("no nan or inf in the inputs", np.all(np.isfinite(data)))
report("no nan or inf in the target", np.all(np.isfinite(target)))
report(f"|M|^2 > 0 in every event (from {target.min():.3g} to {target.max():.3g}), so log10 |M|^2 exists",
       np.all(target > 0))

# The four particles: incoming quark (+z), incoming antiquark (-z), Z boson, gluon
q1, q2, Z, g = data[:, 0:4], data[:, 4:8], data[:, 8:12], data[:, 12:16]


# ======== 2. THE PHYSICS OF THE CARTESIAN COLUMNS ============#
print("2. The physics of the Cartesian columns")
m_Z = np.sqrt(minkowski_square(Z))
m_g = np.sqrt(np.abs(minkowski_square(g))) # abs: rounding can make E^2 - p^2 of the massless gluon slightly negative
report(f"m_Z = {M_Z} GeV in every event (from {m_Z.min():.4f} to {m_Z.max():.4f})",
       np.all(np.abs(m_Z - M_Z) < TOLERANCE_GEV))
report(f"the gluon is massless (largest mass {m_g.max():.4f} GeV, rounding)", np.all(m_g < 0.1))

# The incoming partons move along the beam: px = py = 0, and E = |pz| (massless)
report("incoming partons have px = py = 0", np.all(q1[:, 1:3] == 0) and np.all(q2[:, 1:3] == 0))
report("incoming quark moves along +z and the antiquark along -z (E = pz and E = -pz)",
       np.allclose(q1[:, 0], q1[:, 3]) and np.allclose(q2[:, 0], -q2[:, 3]))

# Momentum conservation: what comes in goes out, in all four components
violation = np.abs(q1 + q2 - Z - g).max(axis=0) # The largest violation of E, px, py, pz over the events
report(f"momentum conservation q1 + q2 = Z + g (largest violation {violation.max():.1e} GeV)",
       np.all(violation < TOLERANCE_GEV))


# ======== 3. THE TARGET BELONGS TO THE MOMENTA ============#
print("3. The target")
# The Mandelstam variables. For qq->Zg: s + t + u = m_Z^2
s = minkowski_square(q1 + q2)
t = minkowski_square(q1 - g)
u = minkowski_square(q2 - g)
report(f"s + t + u = m_Z^2 (largest deviation {np.abs(s + t + u - M_Z**2).max():.1e} GeV^2)",
       np.allclose(s + t + u, M_Z**2, rtol=0, atol=1.0))

# |M|^2 = constant * (t^2 + u^2 + 2 s m_Z^2) / (t u). The constant holds the couplings; we only need it to be the same
# in every event. A shuffled target, or a wrong column order, would give a different ratio in every event.
ratio = target / ((t**2 + u**2 + 2 * s * M_Z**2) / (t * u))
largest_deviation = np.abs(ratio / ratio.mean() - 1).max()
report(f"|M|^2 / formula is the same in every event ({ratio.mean():.6f}, largest relative deviation "
       f"{largest_deviation:.1e})", largest_deviation < TOLERANCE_REL)


# ======== 4. THE PT + ANGLES TRANSFORM ============#
print("4. The pT + angles transform")
pt_angles = cartesian_to_pt_angles(data)
report(f"shape {pt_angles.shape}: one row per event, {len(PT_ANGLES_COLUMNS)} columns",
       pt_angles.shape == (data.shape[0], len(PT_ANGLES_COLUMNS)))
report("no nan or inf (the incoming partons, with infinite rapidity, are left out)", np.all(np.isfinite(pt_angles)))

print(f"  {'column':>8} {'min':>10} {'max':>10}")
for i in range(len(PT_ANGLES_COLUMNS)):
    print(f"  {PT_ANGLES_COLUMNS[i]:>8} {pt_angles[:, i].min():10.3f} {pt_angles[:, i].max():10.3f}")
report("pT_Z = pT_g in every event (momentum balance)", np.allclose(pt_angles[:, 3], pt_angles[:, 7]))
report("phi between -pi and pi", np.all(np.abs(pt_angles[:, [4, 8]]) <= np.pi))

# Back to Cartesian: px = pT cos(phi), py = pT sin(phi), pz = E tanh(y) (since y = atanh(pz / E)).
# If the transform is right, this gives the original momenta of the Z and the gluon.
for name, E, pT, phi, y, original in [("Z", pt_angles[:, 2], pt_angles[:, 3], pt_angles[:, 4], pt_angles[:, 5], Z),
                                      ("gluon", pt_angles[:, 6], pt_angles[:, 7], pt_angles[:, 8], pt_angles[:, 9], g)]:
    back = np.stack([E, pT * np.cos(phi), pT * np.sin(phi), E * np.tanh(y)], axis=1)
    largest_difference = np.abs(back - original).max()
    report(f"{name}: transformed back to Cartesian gives the original momenta (largest difference "
           f"{largest_difference:.1e} GeV)", largest_difference < 1e-6)


# ======== 4b. THE LOG PT + COS/SIN ANGLES TRANSFORMS ============#
print("4b. The log pT + (cos phi, sin phi) transforms")
logpt_bases = {}
for polar in ("rapidity", "cos_theta"):
    columns = logpt_angles_columns(polar)
    inputs = cartesian_to_logpt_angles(data, polar)
    logpt_bases[polar] = inputs
    report(f"{polar}: shape {inputs.shape}, {len(columns)} columns", inputs.shape == (data.shape[0], len(columns)))
    report(f"{polar}: no nan or inf (pT > 0 in every event, so log pT exists)", np.all(np.isfinite(inputs)))
    print(f"  {'column':>12} {'min':>10} {'max':>10}")
    for i in range(len(columns)):
        print(f"  {columns[i]:>12} {inputs[:, i].min():10.3f} {inputs[:, i].max():10.3f}")
    in_range = [i for i, name in enumerate(columns) if name.startswith(("cos_", "sin_"))]
    report(f"{polar}: cos phi, sin phi{' and cos theta' if polar == 'cos_theta' else ''} between -1 and 1",
           np.all(np.abs(inputs[:, in_range]) <= 1))
    # Back to Cartesian: pT = exp(log pT), px = pT cos(phi), py = pT sin(phi), and pz = E tanh(y), or
    # pz = |p| cos(theta) with |p| = pT / sin(theta) = pT / sqrt(1 - cos(theta)^2)
    for name, start, original in [("Z", 2, Z), ("gluon", 7, g)]:
        E, log_pT, cos_phi, sin_phi, along_beam = inputs[:, start:start + 5].T
        pT = np.exp(log_pT)
        report(f"{polar}, {name}: cos(phi)^2 + sin(phi)^2 = 1", np.allclose(cos_phi**2 + sin_phi**2, 1))
        if polar == "rapidity":
            pz = E * np.tanh(along_beam)
        else:
            pz = pT * along_beam / np.sqrt(1 - along_beam**2)
        back = np.stack([E, pT * cos_phi, pT * sin_phi, pz], axis=1)
        largest_difference = np.abs(back - original).max()
        report(f"{polar}, {name}: transformed back to Cartesian gives the original momenta (largest difference "
               f"{largest_difference:.1e} GeV)", largest_difference < 1e-6)


# ======== 5. THE PREPARATION OF THE SCAN, FOR ALL BASES ============#
print("5. The preparation of the scan")
log_target = np.log10(target)
for basis, inputs in [("Cartesian", data), ("pT + angles", pt_angles),
                      ("log pT + cos/sin phi + y", logpt_bases["rapidity"]),
                      ("log pT + cos/sin phi + cos theta", logpt_bases["cos_theta"])]:
    inputs = drop_constant_columns(inputs)
    train_data, val_data, test_data = split_sim_data(inputs, log_target)
    train_data, val_data, test_data = standardize(train_data, val_data, test_data)
    X_train, y_train = train_data

    print(f"  {basis}: {inputs.shape[1]} input columns, {len(X_train)} / {len(val_data[0])} / {len(test_data[0])} "
          f"training / validation / test events")
    report(f"{basis}: the split uses every event once", len(X_train) + len(val_data[0]) + len(test_data[0]) == len(data))
    report(f"{basis}: standardized training inputs have mean 0 and std 1",
           X_train.mean(dim=0).abs().max() < 1e-3 and (X_train.std(dim=0) - 1).abs().max() < 1e-3)

    L_estimate = lipschitz_bound_estimate_of_data(X_train[:N_EVENTS_L], y_train[:N_EVENTS_L])
    report(f"{basis}: Lipschitz estimate on {N_EVENTS_L} training events = {L_estimate:.3f} (finite and > 0)",
           np.isfinite(L_estimate) and L_estimate > 0)


# ======== SUMMARY ============#
print()
if len(failures) == 0:
    print("All checks passed.")
else:
    print(f"{len(failures)} checks FAILED:")
    for description in failures:
        print(f"  {description}")
