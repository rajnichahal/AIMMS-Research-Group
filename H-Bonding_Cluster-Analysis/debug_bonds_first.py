import re
import numpy as np
import freud
from collections import defaultdict

_lattice_re = re.compile(r'\bLattice\s*=\s*"?([0-9eE\.\-\+ ]+)"?')

with open("FlibeCsF_510C_27x_train1_nvt_equi.xyz", "r") as f:          # ← only change this
    N = int(f.readline().strip())
    comment = f.readline()
    m = _lattice_re.search(comment)
    vals = [float(x) for x in m.group(1).split()]
    lat = np.array(vals).reshape(3, 3)
    symbols, pos = [], np.zeros((N, 3))
    for i in range(N):
        parts = f.readline().split()
        symbols.append(parts[0])
        pos[i] = [float(parts[1]), float(parts[2]), float(parts[3])]

sym_arr = np.array(symbols)
box = freud.box.Box.from_matrix(lat)

aq = freud.locality.AABBQuery(box, pos)
nlist = aq.query(pos, {"r_max": 3.5, "exclude_ii": True}).toNeighborList()   # Change rmax based on the material

pair_distances = defaultdict(list)
for k in range(len(nlist)):
    i = nlist.query_point_indices[k]
    j = nlist.point_indices[k]
    d = nlist.distances[k]
    pair = tuple(sorted([sym_arr[i], sym_arr[j]]))
    pair_distances[pair].append(d)

print("Per-pair distance distribution:")
print("=" * 60)
for pair in sorted(pair_distances.keys()):
    dists = np.array(pair_distances[pair])
    print(f"\n{pair[0]}-{pair[1]}  (total: {len(dists)})")
    bins = np.arange(0.8, 3.55, 0.05)                          # Change 3.05 based on your rmax
    hist, edges = np.histogram(dists, bins=bins)
    for i in range(len(hist)):
        if hist[i] > 0:
            bar = "█" * min(30, hist[i] // 10)
            print(f"  {edges[i]:.2f}-{edges[i+1]:.2f} Å : {hist[i]:>6}  {bar}")
