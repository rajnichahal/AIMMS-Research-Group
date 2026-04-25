# -*- coding: utf-8 -*-
import re
import sys
import numpy as np
import freud
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

_lattice_re = re.compile(r'\bLattice\s*=\s*"?([0-9eE\.\-\+ ]+)"?')

# =============================================================================
# PAIR_CUTOFFS ? update after debug_bonds.py
# =============================================================================
PAIR_CUTOFFS = {
    frozenset(['Be', 'F']):  2.10,
    frozenset(['F',  'Li']): 2.10,
    frozenset(['Cs', 'F']):  0.00, #['Cs', 'F']):  3.20 for li-cs
    frozenset(['Be', 'Be']): 0.00,
    frozenset(['Be', 'Li']): 0.00,
    frozenset(['Be', 'Cs']): 0.00,
    frozenset(['F',  'F']):  0.00,
    frozenset(['Li', 'Li']): 0.00,
    frozenset(['Cs', 'Cs']): 0.00,
    frozenset(['Cs', 'Li']): 0.00,
}

# =============================================================================
# USAGE:
#   python test_chains_second.py myfile.xyz polymer
#   python test_chains_second.py myfile.xyz molten_salt
# =============================================================================

if len(sys.argv) < 3:
    print("Usage: python test_chains_second.py <file.xyz> <polymer|molten_salt>")
    sys.exit(1)

xyz_file    = sys.argv[1]
mode        = sys.argv[2].lower().replace(" ", "_").replace("-", "_")

if mode not in ("polymer", "molten_salt"):
    print("ERROR: mode must be 'polymer' or 'molten_salt'")
    sys.exit(1)

# =============================================================================
# READ XYZ
# =============================================================================
with open(xyz_file, "r") as f:
    N       = int(f.readline().strip())
    comment = f.readline()
    m       = _lattice_re.search(comment)
    vals    = [float(x) for x in m.group(1).split()]
    lat     = np.array(vals).reshape(3, 3)
    symbols, pos = [], np.zeros((N, 3))
    for i in range(N):
        parts = f.readline().split()
        symbols.append(parts[0])
        pos[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

sym_arr = np.array(symbols)
box     = freud.box.Box.from_matrix(lat)

# =============================================================================
# BUILD CONNECTIVITY
# =============================================================================
max_cut = max((v for v in PAIR_CUTOFFS.values() if v > 0), default=2.0)
aq      = freud.locality.AABBQuery(box, pos)
nlist   = aq.query(pos, {"r_max": max_cut, "exclude_ii": True}).toNeighborList()

rows, cols = [], []
for k in range(len(nlist)):
    i  = nlist.query_point_indices[k]
    j  = nlist.point_indices[k]
    d  = nlist.distances[k]
    cut = PAIR_CUTOFFS.get(frozenset([sym_arr[i], sym_arr[j]]), 0.00)
    if d <= cut:
        rows.append(i); cols.append(j)

mat = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
n_clusters, cluster_ids = connected_components(mat, directed=False)
sizes        = np.bincount(cluster_ids)
unique_sizes = sorted(set(sizes.tolist()))
composition  = Counter(symbols)

# =============================================================================
# PRINT BASIC INFO (same for both modes)
# =============================================================================
W = 56
print("\n" + "="*W)
print(f"  TEST CHAINS ? MODE: {mode.upper().replace('_',' ')}")
print("="*W)
print(f"  File         : {xyz_file}")
print(f"  Total atoms  : {N}")
print(f"  Composition  : {dict(composition)}")
print(f"  Total clusters: {n_clusters}")
print(f"  Cluster sizes : {unique_sizes}")
print(f"  Min size     : {sizes.min()}")
print(f"  Max size     : {sizes.max()}")
print("="*W)

# =============================================================================
# POLYMER CHECKS
# =============================================================================
if mode == "polymer":
    most_common_size = int(np.bincount(sizes).argmax()) if len(unique_sizes) > 1 else unique_sizes[0]
    print(f"\nAuto-detected dominant chain size: {most_common_size} atoms")
    user_input = input(f"Press Enter to use {most_common_size}, or type a different size: ").strip()
    expected_chain_size = int(user_input) if user_input else most_common_size
    tolerance = 5

    print("\n" + "="*W)
    print("  POLYMER VERIFICATION CHECKS")
    print("="*W)

    # Check 1: isolated atoms
    n_isolated = int(np.sum(sizes == 1))
    print(f"\n[1] Isolated atoms (size=1): {n_isolated}")
    print(f"    {'PASS' if n_isolated == 0 else 'FAIL -- cutoffs may be too tight'}")

    # Check 2: oversized chains
    max_expected = expected_chain_size + tolerance
    n_big = int(np.sum(sizes > max_expected))
    print(f"\n[2] Oversized chains (>{max_expected} atoms): {n_big}")
    print(f"    {'PASS' if n_big == 0 else 'FAIL -- cutoffs may be too loose, chains merging'}")

    # Check 3: uniform sizes
    print(f"\n[3] Unique chain sizes: {unique_sizes}")
    print(f"    {'PASS -- uniform chains' if len(unique_sizes) == 1 else 'WARNING -- mixed sizes (ok if polymer is irregular)'}")

    # Check 4: atom count
    total_accounted = int(sizes.sum())
    print(f"\n[4] Atoms accounted for: {total_accounted} / {N}")
    print(f"    {'PASS' if total_accounted == N else 'FAIL -- some atoms missing'}")

    # Check 5: expected chain count
    expected_chains = N // expected_chain_size
    diff = abs(n_clusters - expected_chains)
    print(f"\n[5] Expected chains: ~{expected_chains} | Found: {n_clusters}")
    print(f"    {'PASS' if diff <= 2 else 'WARNING -- chain count differs by ' + str(diff)}")

    print("="*W)

# =============================================================================
# MOLTEN SALT CHECKS
# =============================================================================
elif mode == "molten_salt":
    total_accounted = int(sizes.sum())
    n_giant         = int(np.sum(sizes > 100))
    n_free_ions     = int(np.sum(sizes == 1))
    n_small_cluster = int(np.sum((sizes > 1) & (sizes <= 20)))
    max_cluster_frac = sizes.max() / N * 100

    print("\n" + "="*W)
    print("  MOLTEN SALT VERIFICATION CHECKS")
    print("="*W)

    # Check 1: atom count
    print(f"\n[1] Atoms accounted for: {total_accounted} / {N}")
    print(f"    {'PASS' if total_accounted == N else 'FAIL -- some atoms missing'}")

    # Check 2: giant network exists
    print(f"\n[2] Giant network clusters (>100 atoms): {n_giant}")
    print(f"    {'PASS -- percolating network exists' if n_giant >= 1 else 'FAIL -- no network found, cutoffs too tight'}")

    # Check 3: giant network fraction
    print(f"\n[3] Largest cluster: {sizes.max()} atoms ({max_cluster_frac:.1f}% of system)")
    if max_cluster_frac > 50:
        print(f"    PASS -- dominant percolating network (>50%)")
    elif max_cluster_frac > 20:
        print(f"    WARNING -- network exists but fragmented ({max_cluster_frac:.1f}%)")
    else:
        print(f"    FAIL -- system too fragmented, check cutoffs")

    # Check 4: free ions
    print(f"\n[4] Free ions (size=1): {n_free_ions} ({n_free_ions/N*100:.1f}% of atoms)")
    if n_free_ions / N < 0.15:
        print(f"    PASS -- reasonable free ion fraction (<15%)")
    elif n_free_ions / N < 0.30:
        print(f"    WARNING -- high free ion fraction ({n_free_ions/N*100:.1f}%), check cutoffs")
    else:
        print(f"    FAIL -- too many free ions (>{30}%), cutoffs likely too tight")

    # Check 5: small clusters
    print(f"\n[5] Small clusters (2-20 atoms): {n_small_cluster}")
    print(f"    INFO -- these are BeF4^2-, BeF3-, LiF units not yet in network")

    # Check 6: composition sanity
    print(f"\n[6] Composition: {dict(composition)}")
    total_comp = sum(composition.values())
    print(f"    PASS -- {total_comp} atoms identified" if total_comp == N else f"    FAIL -- count mismatch")

    # Check 7: per-element cluster participation
    print(f"\n[7] Per-element free ion count:")
    for elem in sorted(composition.keys()):
        idx_free = [i for i in range(N) if sym_arr[i] == elem and sizes[cluster_ids[i]] == 1]
        pct = len(idx_free) / composition[elem] * 100
        flag = "OK" if pct < 20 else "HIGH"
        print(f"    {elem:3s}: {len(idx_free):4d} / {composition[elem]:4d} free  ({pct:5.1f}%)  [{flag}]")

    print("\n" + "="*W)
    print("  INTERPRETATION GUIDE")
    print("="*W)
    print("  Giant network  -> the molten salt backbone (correct)")
    print("  Free ions      -> uncoordinated ions in melt (normal <15%)")
    print("  Small clusters -> BeF4^2-, LiF, CsF units (normal in melt)")
    print("  If too many free ions -> increase Be-F or F-Li cutoff slightly")
    print("  If giant network too small -> cutoffs too tight")
    print("="*W)

# example
# For your polymer
#python test_chains_second.py mypolymer.xyz polymer

# For FLiBe+CsF
#python test_chains_second.py FlibeCsF_510C_27x_train1_nvt_equi.xyz molten_salt
