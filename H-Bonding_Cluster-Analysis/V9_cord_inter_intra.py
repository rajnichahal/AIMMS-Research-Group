import argparse
import re
import numpy as np
import freud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import plotly.graph_objects as go
import plotly.io as pio
from collections import Counter, defaultdict
import os

_lattice_re = re.compile(r'\bLattice\s*=\s*"?([0-9eE\.\-\+ ]+)"?')
_int_line_re = re.compile(r"^\s*(\d+)\s*$")

PAIR_COLORS = [
    {'intra': 'green',      'inter': 'royalblue',  'intra_ls': '-',  'inter_ls': '--'},
    {'intra': 'darkorange', 'inter': 'purple',      'intra_ls': '-',  'inter_ls': '--'},
    {'intra': 'deepskyblue','inter': 'crimson',     'intra_ls': '-',  'inter_ls': '--'},
    {'intra': 'lime',       'inter': 'darkviolet',  'intra_ls': '-',  'inter_ls': '--'},
    {'intra': 'gold',       'inter': 'darkred',     'intra_ls': '-',  'inter_ls': '--'},
]

CHAIN_PALETTE = [
    'darkorange','purple','saddlebrown','deeppink','olive','teal',
    'crimson','navy','darkgreen','goldenrod','indigo','coral',
    'steelblue','sienna','mediumorchid','chocolate','darkviolet',
    'firebrick','seagreen','peru',
]

CLUSTER_PALETTE_3D = [
    'red','blue','limegreen','darkorange','purple','deepskyblue',
    'crimson','gold','teal','hotpink','saddlebrown','navy',
    'olive','darkviolet','coral','seagreen','indigo','firebrick',
    'steelblue','mediumorchid',
]

PAIR_CUTOFFS = {
    frozenset(['Be','F']): 2.10, frozenset(['F','Li']): 2.10,
    frozenset(['Cs','F']): 0.00, frozenset(['Be', 'Be']): 0.00,
    frozenset(['Be', 'Li']): 0.00, frozenset(['Be', 'Cs']): 0.00,
    frozenset(['F',  'F']): 0.00, frozenset(['Li', 'Li']): 0.00,
    frozenset(['Cs', 'Cs']): 0.00, frozenset(['Cs', 'Li']): 0.00,
}

# ── Automatically derived from PAIR_CUTOFFS — do NOT edit below ─────────────
# Rule: any atom X where X-H appears in PAIR_CUTOFFS with cutoff > 0,
#       AND X is not C (C-H is never a real H-bond donor),
#       is treated as a valid donor automatically.
#
# To support a new system (e.g. with S-H thiol bonds):
#   Step 1: run debug_bonds_first.py  →  find X-H bond distances
#   Step 2: add to PAIR_CUTOFFS above →  frozenset(['S','H']): 1.40
#   That is ALL. Everything below updates automatically.
#   No other edits needed anywhere in the script.
NON_DONORS = {'C'}   # C-H is never a real H-bond donor — everything else is
DONOR_COVALENT_CUTOFFS = {
    pair: cut
    for pair, cut in PAIR_CUTOFFS.items()
    if 'H' in pair and cut > 0.0
    and any(atom not in NON_DONORS and atom != 'H' for atom in pair)
}
VALID_DONORS = tuple(sorted({
    atom
    for pair in DONOR_COVALENT_CUTOFFS
    for atom in pair
    if atom != 'H' and atom not in NON_DONORS
}))
_DONOR_SEARCH_RADIUS = (max(DONOR_COVALENT_CUTOFFS.values()) + 0.05
                        if DONOR_COVALENT_CUTOFFS else 1.30)


#  PARSERS / READERS
def _parse_lattice(line):
    m = _lattice_re.search(line)
    if m is None: raise ValueError("Could not find Lattice")
    return np.array([float(x) for x in m.group(1).split()], dtype=float).reshape(3,3)

def _try_read_frame(f):
    pos0  = f.tell()
    nline = f.readline()
    if not nline: return None
    mN = _int_line_re.match(nline)
    if mN is None: f.seek(pos0); return None
    N = int(mN.group(1))
    comment = f.readline()
    if not comment or "Lattice" not in comment: f.seek(pos0); return None
    try:    lat = _parse_lattice(comment)
    except: f.seek(pos0); return None
    symbols, pos = [], np.zeros((N,3), dtype=float)
    for i in range(N):
        line = f.readline()
        if not line: f.seek(pos0); return None
        p = line.split()
        symbols.append(p[0])
        pos[i] = [float(p[1]), float(p[2]), float(p[3])]
    return lat, symbols, pos

def read_xyz_frames(path):
    with open(path, "r", errors="ignore") as f:
        while True:
            frame = _try_read_frame(f)
            if frame: yield frame
            elif not f.readline(): return

def build_type_map(symbols):
    seen = []
    for s in symbols:
        if s not in seen: seen.append(s)
    nid = {s:i for i,s in enumerate(seen)}
    return np.array([nid[s] for s in symbols], dtype=np.int32), seen

def identify_chains(box, pos, bond_cutoff=1.8, symbols=None):
    n = len(pos)
    sa = np.array(symbols) if symbols is not None else None
    aq = freud.locality.AABBQuery(box, pos)
    nl = aq.query(pos, {"r_max":1.72,"exclude_ii":True}).toNeighborList()
    rows, cols = [], []
    for k in range(len(nl)):
        i,j,d = nl.query_point_indices[k], nl.point_indices[k], nl.distances[k]
        cut = PAIR_CUTOFFS.get(frozenset([sa[i],sa[j]]),0.00) if sa is not None else bond_cutoff
        if d <= cut: rows.append(i); cols.append(j)
    cm = csr_matrix((np.ones(len(rows),dtype=int),(rows,cols)), shape=(n,n))
    _, chain_ids = connected_components(cm, directed=False)
    return chain_ids

def coordination_number_intra_inter_detailed(box, A, B, cA, cB, r_cut, same_type=False):
    nl = freud.locality.AABBQuery(box,B).query(A,{"r_max":r_cut}).toNeighborList()
    ic = np.zeros(len(A),dtype=int); ec = np.zeros(len(A),dtype=int)
    id_, ed_ = [], []
    for i in range(len(nl)):
        qi,pi,d = nl.query_point_indices[i], nl.point_indices[i], nl.distances[i]
        if same_type and qi==pi: continue
        if cA[qi]==cB[pi]: ic[qi]+=1; id_.append(d)
        else:               ec[qi]+=1; ed_.append(d)
    return np.mean(ic), np.mean(ec), np.array(id_), np.array(ed_)

def mic_vector(box, ri, rj):
    dx = np.array(ri,dtype=float)-np.array(rj,dtype=float)
    dx[0] -= box.Lx*np.round(dx[0]/box.Lx)
    dx[1] -= box.Ly*np.round(dx[1]/box.Ly)
    dx[2] -= box.Lz*np.round(dx[2]/box.Lz)
    return dx

def get_hbonds_for_pair(box, A_pos, B_pos, A_ch, B_ch, cn_cut):
    nl = freud.locality.AABBQuery(box,B_pos).query(A_pos,{"r_max":cn_cut}).toNeighborList()
    intra, inter = [], []
    for i in range(len(A_pos)):
        for j in nl.point_indices[nl.query_point_indices==i]:
            (intra if A_ch[i]==B_ch[j] else inter).append((i,j))
    return intra, inter

#  V5 NEW: DONOR MAP + ANGLE FILTER

def build_donor_map(box, pos, symbols):
    """
    Build a map from every H atom's global index -> its covalently bonded
    heavy-atom (donor D) global index.

    Uses DONOR_COVALENT_CUTOFFS and _DONOR_SEARCH_RADIUS which are
    automatically derived from PAIR_CUTOFFS at module level.
    To support a new atom type (e.g. S-H): add it to PAIR_CUTOFFS only.

    Returns
    -------
    donor_map : dict  {H_global_idx: D_global_idx}
    """
    # DONOR_COVALENT_CUTOFFS and _DONOR_SEARCH_RADIUS are automatically
    # derived from PAIR_CUTOFFS at module level — edit PAIR_CUTOFFS only.
    sa = np.array(symbols)
    aq = freud.locality.AABBQuery(box, pos)
    nl = aq.query(pos, {"r_max": _DONOR_SEARCH_RADIUS, "exclude_ii": True}).toNeighborList()
    donor_map = {}
    for k in range(len(nl)):
        i = nl.query_point_indices[k]
        j = nl.point_indices[k]
        d = nl.distances[k]
        cut = DONOR_COVALENT_CUTOFFS.get(frozenset([sa[i], sa[j]]), 0.0)
        if cut == 0.0:
            continue
        if sa[i] == 'H' and d <= cut and i not in donor_map:
            donor_map[i] = j          # H=i bonded to heavy atom=j
        elif sa[j] == 'H' and d <= cut and j not in donor_map:
            donor_map[j] = i          # H=j bonded to heavy atom=i
    return donor_map


def calc_dha_angle(box, D_pos, H_pos, A_pos):
    """
    Compute the D-H···A angle in degrees using the minimum image convention.

    The angle is measured at H: the angle between vectors H->D and H->A.
    A value close to 180° means the bond is linear (ideal H-bond).

    Parameters
    ----------
    box    : freud box
    D_pos  : (3,) array — donor heavy atom position
    H_pos  : (3,) array — hydrogen position
    A_pos  : (3,) array — acceptor position

    Returns
    -------
    angle : float, degrees
    """
    vec_HD = mic_vector(box, D_pos, H_pos)   # H -> D
    vec_HA = mic_vector(box, A_pos, H_pos)   # H -> A
    norm_HD = np.linalg.norm(vec_HD)
    norm_HA = np.linalg.norm(vec_HA)
    if norm_HD < 1e-10 or norm_HA < 1e-10:
        return 0.0
    cos_a = np.dot(vec_HD, vec_HA) / (norm_HD * norm_HA)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def get_hbonds_for_pair_v5(box, A_pos, B_pos, A_ch, B_ch, cn_cut,
                            donor_map, pos, sym_arr, angle_cut=150.0,
                            A_global_indices=None, B_global_indices=None,
                            valid_donors=None):
    """
    Same logic as get_hbonds_for_pair() but with:
      1. Donor-element filter: only H atoms bonded to N or O are considered
         (excludes C-H noise which dominates intra-chain at short distances)
      2. D-H···A angle filter: angle >= angle_cut

    Pair convention: A = acceptor (O or N), B = hydrogen (H)
    """
    _vd = valid_donors if valid_donors is not None else VALID_DONORS
    nl = freud.locality.AABBQuery(box, B_pos).query(A_pos, {"r_max": cn_cut}).toNeighborList()
    intra, inter = [], []
    skipped_no_donor = 0

    for i in range(len(A_pos)):
        for j in nl.point_indices[nl.query_point_indices == i]:
            g_j = int(B_global_indices[j]) if B_global_indices is not None else j
            if g_j not in donor_map:
                skipped_no_donor += 1
                continue

            D_global = donor_map[g_j]

            # Skip non-donors — derived automatically from PAIR_CUTOFFS
            if sym_arr[D_global] not in _vd:
                continue

            # Skip the covalent bond itself: when A is the same atom as D
            # (e.g. N-H pair: the N covalently bonded to H is also the acceptor)
            g_i = int(A_global_indices[i]) if A_global_indices is not None else i
            if D_global == g_i:
                continue

            angle = calc_dha_angle(box, pos[D_global], B_pos[j], A_pos[i])
            if angle < angle_cut:
                continue
            (intra if A_ch[i] == B_ch[j] else inter).append((i, j))

    return intra, inter, skipped_no_donor


def coordination_number_intra_inter_detailed_v5(box, A, B, cA, cB, r_cut,
                                                 donor_map, pos, sym_arr,
                                                 angle_cut,
                                                 A_global_indices,
                                                 B_global_indices,
                                                 same_type=False):
    """
    Same as coordination_number_intra_inter_detailed() but applies:
      1. Donor-element filter (N-H and O-H only, excludes C-H noise)
      2. D-H···A angle filter

    Pair convention: A = acceptor (O or N), B = hydrogen (H)
    """
    nl = freud.locality.AABBQuery(box, B).query(A, {"r_max": r_cut}).toNeighborList()
    ic = np.zeros(len(A), dtype=int)
    ec = np.zeros(len(A), dtype=int)
    id_, ed_ = [], []

    for i in range(len(nl)):
        qi, pi, d = nl.query_point_indices[i], nl.point_indices[i], nl.distances[i]
        if same_type and qi == pi:
            continue

        # B[pi] is the H atom — look it up in donor_map
        g_pi = int(B_global_indices[pi]) if B_global_indices is not None else pi
        if g_pi not in donor_map:
            continue
        D_global = donor_map[g_pi]

        # Skip non-donors — derived automatically from PAIR_CUTOFFS
        if sym_arr[D_global] not in VALID_DONORS:
            continue

        # Skip the covalent bond itself: when acceptor A[qi] is the same atom as D
        g_qi_global = int(A_global_indices[qi]) if A_global_indices is not None else qi
        if D_global == g_qi_global:
            continue

        angle = calc_dha_angle(box, pos[D_global], B[pi], A[qi])
        if angle < angle_cut:
            continue

        if cA[qi] == cB[pi]:
            ic[qi] += 1; id_.append(d)
        else:
            ec[qi] += 1; ed_.append(d)

    return np.mean(ic), np.mean(ec), np.array(id_), np.array(ed_)

# ─────────────────────────────────────────────────────────────────────────────
#  V5 D-A MODE: search donor D near acceptor A (VMD distance convention)
# ─────────────────────────────────────────────────────────────────────────────

def coordination_number_intra_inter_detailed_v5_DA(box, A, B, cA, cB, r_cut,
                                                    donor_map, pos, sym_arr,
                                                    angle_cut,
                                                    A_global_indices,
                                                    B_global_indices,
                                                    donor_pos_map,
                                                    same_type=False):
    """
    D-A distance mode: searches donor heavy atoms (D) near acceptor atoms (A)
    within r_cut, then finds the H bonded to each D, then checks D-H···A angle.

    This matches VMD's distance convention: r_cut is the D···A distance,
    not H···A. Use --cn-cut ~3.5 with this mode (VMD default).

    donor_pos_map : dict  {element -> (positions, chain_ids, global_indices)}
                    built in main loop for each valid donor element
    """
    # Build reverse map: global H index -> chain ID (for intra/inter check)
    # We need the chain of the H atom, not the D atom
    ic = np.zeros(len(A), dtype=int)
    ec = np.zeros(len(A), dtype=int)
    id_, ed_ = [], []

    # For each donor element, search its positions near A
    for elem, (D_pos, D_ch, D_glob) in donor_pos_map.items():
        nl = freud.locality.AABBQuery(box, D_pos).query(A, {"r_max": r_cut}).toNeighborList()
        for k in range(len(nl)):
            qi = nl.query_point_indices[k]   # index into A
            pi = nl.point_indices[k]          # index into D_pos
            d  = nl.distances[k]              # D···A distance

            if same_type and A_global_indices is not None:
                if A_global_indices[qi] == D_glob[pi]:
                    continue

            # Find the H bonded to this D atom
            D_global = D_glob[pi]
            # Look for H in donor_map values that match D_global
            # Build H from donor_map: donor_map[H_idx] = D_idx
            # We need the reverse: D_idx -> H_idx
            # Use the pre-built reverse map
            H_global = _reverse_donor_map.get(D_global)
            if H_global is None:
                continue

            # Skip covalent self-bond: D and A are the same atom
            if A_global_indices is not None and D_global == A_global_indices[qi]:
                continue

            # Angle check: D-H···A
            angle = calc_dha_angle(box, pos[D_global], pos[H_global], A[qi])
            if angle < angle_cut:
                continue

            # Intra/inter: compare chain of D with chain of A
            # (D and its H are on the same chain, so D_ch == H_ch)
            if cA[qi] == D_ch[pi]:
                ic[qi] += 1; id_.append(d)
            else:
                ec[qi] += 1; ed_.append(d)
    total_cn = np.mean(ic + ec)
    return np.mean(ic), np.mean(ec), total_cn, np.array(id_), np.array(ed_)


# CLUSTER ANALYSIS  (V4 — Molten Salt / OVITO-style)
def analyse_clusters_frame(box, pos, symbols, pair, cluster_cut, count_atom):
    sym_arr = np.array(symbols)
    a, b    = pair
    idx_a   = np.where(sym_arr == a)[0]
    idx_b   = np.where(sym_arr == b)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        return 0, [], np.zeros(len(pos),dtype=int), np.array([],dtype=int), np.array([],dtype=int)
    pos_a   = pos[idx_a]
    pos_b   = pos[idx_b]
    len_a   = len(idx_a)
    n_total = len_a + len(idx_b)
    nl = freud.locality.AABBQuery(box, pos_b).query(pos_a, {"r_max": cluster_cut}).toNeighborList()
    bond_list = []
    rows, cols = [], []
    for k in range(len(nl)):
        i_a = nl.query_point_indices[k]; i_b = nl.point_indices[k]
        bond_list.append((i_a, i_b))
        rows.append(i_a);         cols.append(len_a + i_b)
        rows.append(len_a + i_b); cols.append(i_a)
    if not rows:
        return 0, [], np.zeros(len(pos),dtype=int), np.array([],dtype=int), np.array([],dtype=int)
    cm = csr_matrix((np.ones(len(rows),dtype=int),(rows,cols)), shape=(n_total,n_total))
    n_clusters, labels = connected_components(cm, directed=False)
    labels_a = labels[:len_a]; labels_b = labels[len_a:]
    target_local = np.arange(len_a) if count_atom == a else np.arange(len_a, n_total)
    size_list = []
    for cid in range(n_clusters):
        members  = np.where(labels == cid)[0]
        n_target = int(np.sum(np.isin(members, target_local)))
        if n_target > 0: size_list.append(n_target)
    return len(size_list), size_list, labels, labels_a, labels_b
    

def plot_3d_clusters(box, pos, symbols, pair, cluster_cut, count_atom, out_prefix, frame_num):
    sym_arr = np.array(symbols)
    a, b    = pair
    idx_a   = np.where(sym_arr == a)[0]; idx_b = np.where(sym_arr == b)[0]
    if len(idx_a)==0 or len(idx_b)==0: print("  [cluster 3D] atoms not found"); return
    pos_a  = pos[idx_a]; pos_b = pos[idx_b]
    len_a  = len(idx_a); n_total = len_a + len(idx_b)
    nl = freud.locality.AABBQuery(box, pos_b).query(pos_a, {"r_max": cluster_cut}).toNeighborList()
    bond_list = []; rows, cols = [], []
    for k in range(len(nl)):
        i_a = nl.query_point_indices[k]; i_b = nl.point_indices[k]
        bond_list.append((i_a, i_b))
        rows.append(i_a); cols.append(len_a+i_b)
        rows.append(len_a+i_b); cols.append(i_a)
    if not rows: print(f"  [cluster 3D] No bonds for {a}-{b}"); return
    cm = csr_matrix((np.ones(len(rows),dtype=int),(rows,cols)), shape=(n_total,n_total))
    n_clusters, labels = connected_components(cm, directed=False)
    labels_a = labels[:len_a]; labels_b = labels[len_a:]
    active_clusters = sorted(set(labels_a) if count_atom==a else set(labels_b))
    cmap = {cid: CLUSTER_PALETTE_3D[i%len(CLUSTER_PALETTE_3D)] for i,cid in enumerate(active_clusters)}
    pair_str = a + "-" + b

    # --- PNG ---
    fig = plt.figure(figsize=(13,9)); ax = fig.add_subplot(111, projection='3d')
    plotted_a = set()
    for i, pa in enumerate(pos_a):
        cid   = labels_a[i]; color = cmap.get(cid,'gray')
        lbl   = ("_nolegend_" if cid in plotted_a else
                 "Cluster " + str(cid) + ": " + str(int(np.sum(labels_a==cid))) + " " + a +
                 ", " + str(int(np.sum(labels_b==cid))) + " " + b)
        ax.scatter(*pa, c=color, s=280, edgecolors='black', lw=1.5, alpha=0.9, zorder=10, label=lbl)
        plotted_a.add(cid)
    for j, pb_raw in enumerate(pos_b):
        cid = labels_b[j]; color = cmap.get(cid,'gray')
        ax.scatter(*pb_raw, c=color, s=110, edgecolors='black', lw=1.0,
                   marker='s', alpha=0.75, zorder=8, label='_nolegend_')
    for i_a, i_b in bond_list:
        pa = pos_a[i_a]; pb = pa + mic_vector(box, pos_b[i_b], pa)
        color = cmap.get(labels_a[i_a],'gray')
        ax.plot([pa[0],pb[0]],[pa[1],pb[1]],[pa[2],pb[2]],'-',color=color,lw=1.8,alpha=0.55)
    legend_handles = []
    for cid in active_clusters[:14]:
        legend_handles.append(Patch(facecolor=cmap[cid], edgecolor='black',
            label="Cluster "+str(cid)+": "+str(int(np.sum(labels_a==cid)))+" "+a+
                  ", "+str(int(np.sum(labels_b==cid)))+" "+b))
    if len(active_clusters) > 14:
        legend_handles.append(Patch(facecolor='white', edgecolor='black',
            label="... +" + str(len(active_clusters)-14) + " more"))
    legend_handles.append(Line2D([0],[0],marker='o',color='w',markerfacecolor='gray',
        markersize=11, label=a+" atoms (circle)"))
    legend_handles.append(Line2D([0],[0],marker='s',color='w',markerfacecolor='gray',
        markersize=9, label=b+" atoms (square)"))
    ax.legend(handles=legend_handles, fontsize=8, loc='upper left', framealpha=0.95)
    ax.set_xlabel('X (A)',fontsize=12,fontweight='bold')
    ax.set_ylabel('Y (A)',fontsize=12,fontweight='bold')
    ax.set_zlabel('Z (A)',fontsize=12,fontweight='bold')
    ax.set_title("Cluster 3D: "+pair_str+" | Frame "+str(frame_num)+" | Cutoff "+
                 str(round(cluster_cut,2))+" A | "+str(n_clusters)+" clusters",
                 fontsize=12, fontweight='bold')
    ax.text2D(0.02,0.02,
              "Clusters: "+str(n_clusters)+" | "+a+": "+str(len_a)+" | "+
              b+": "+str(len(idx_b))+" | Bonds: "+str(len(bond_list)),
              transform=ax.transAxes, fontsize=9, va='bottom',
              bbox=dict(boxstyle='round,pad=0.5',facecolor='lightyellow',edgecolor='black',alpha=0.85))
    ax.view_init(elev=25, azim=40); plt.tight_layout()
    png_out = out_prefix + "_cluster_" + pair_str + "_3D.png"
    plt.savefig(png_out, dpi=300, bbox_inches='tight'); plt.close()
    print("  Cluster 3D PNG : " + png_out)

    # --- Plotly HTML ---
    fig_p = go.Figure()
    for cid in active_clusters:
        color = cmap.get(cid,'gray')
        mask_a = labels_a == cid; mask_b = labels_b == cid
        if np.any(mask_a):
            fig_p.add_trace(go.Scatter3d(
                x=pos_a[mask_a,0], y=pos_a[mask_a,1], z=pos_a[mask_a,2],
                mode='markers',
                marker=dict(size=10, color=color, line=dict(color='black',width=1)),
                name="Cluster "+str(cid)+" - "+a+" ("+str(int(np.sum(mask_a)))+")",
                legendgroup="cl_"+str(cid),
                hovertemplate="Cluster "+str(cid)+" | "+a+"<br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}"))
        if np.any(mask_b):
            fig_p.add_trace(go.Scatter3d(
                x=pos_b[mask_b,0], y=pos_b[mask_b,1], z=pos_b[mask_b,2],
                mode='markers',
                marker=dict(size=6, color=color, symbol='square', line=dict(color='black',width=1)),
                name="Cluster "+str(cid)+" - "+b+" ("+str(int(np.sum(mask_b)))+")",
                legendgroup="cl_"+str(cid),
                hovertemplate="Cluster "+str(cid)+" | "+b+"<br>x=%{x:.2f} y=%{y:.2f} z=%{z:.2f}"))
    for i_a, i_b in bond_list:
        pa = pos_a[i_a]; pb = pa + mic_vector(box, pos_b[i_b], pa)
        color = cmap.get(labels_a[i_a],'gray')
        fig_p.add_trace(go.Scatter3d(
            x=[pa[0],pb[0],None], y=[pa[1],pb[1],None], z=[pa[2],pb[2],None],
            mode='lines', line=dict(color=color,width=3),
            legendgroup="cl_"+str(labels_a[i_a]),
            showlegend=False, hoverinfo='skip'))
    fig_p.update_layout(
        title=dict(text="Cluster 3D: "+pair_str+" | Frame "+str(frame_num)+
                   " | Cutoff "+str(round(cluster_cut,2))+" A | "+str(n_clusters)+" clusters<br>"+
                   a+": "+str(len_a)+" | "+b+": "+str(len(idx_b))+" | Bonds: "+str(len(bond_list)),
                   font=dict(size=13)),
        scene=dict(
            xaxis=dict(title='X (A)',backgroundcolor='rgb(245,245,245)'),
            yaxis=dict(title='Y (A)',backgroundcolor='rgb(245,245,245)'),
            zaxis=dict(title='Z (A)',backgroundcolor='rgb(235,235,235)'),
            camera=dict(eye=dict(x=1.5,y=1.5,z=1.2))),
        legend=dict(yanchor='top',y=0.99,xanchor='left',x=1.01,
                    bgcolor='rgba(255,255,240,0.95)',bordercolor='black',
                    borderwidth=1,font=dict(size=10),tracegroupgap=2),
        margin=dict(l=0,r=260,t=150,b=0), width=1260, height=900, paper_bgcolor='white')
    html_out = out_prefix + "_cluster_" + pair_str + "_3D_interactive.html"
    pio.write_html(fig_p, html_out, include_plotlyjs='cdn')
    print("  Cluster HTML   : " + html_out)

def plot_cluster_probability(all_size_lists, count_atom, pair, out_prefix):
    all_sizes = []
    for sizes in all_size_lists: all_sizes.extend(sizes)
    if not all_sizes: print("  [cluster] No data to plot"); return
    total   = len(all_sizes); counter = Counter(all_sizes)
    max_n   = max(counter.keys()); ns = np.arange(1, max_n+1)
    probs   = np.array([counter.get(n,0)/total for n in ns])
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(ns, probs, color='steelblue', edgecolor='navy', alpha=0.85, width=0.6)
    ax.set_xlabel("Number of "+count_atom+" per cluster (n)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Probability P(n)", fontsize=12, fontweight='bold')
    ax.set_title("Cluster Size Distribution: "+pair[0]+"-"+pair[1]+
                 " | Counting "+count_atom+" | "+str(len(all_size_lists))+" frames",
                 fontsize=12, fontweight='bold')
    ax.set_xticks(ns); ax.grid(True, alpha=0.3, axis='y')
    for n, p in zip(ns, probs):
        if p > 0: ax.text(n, p+0.005, str(round(p,3)), ha='center', va='bottom', fontsize=8)
    mean_size = np.mean(all_sizes)
    ax.axvline(mean_size, color='red', lw=1.5, ls='--', alpha=0.7,
               label="Mean = "+str(round(mean_size,2)))
    ax.legend(fontsize=10); plt.tight_layout()
    out = out_prefix + "_cluster_prob_" + count_atom + ".png"
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print("  Cluster prob plot: " + out)

def write_cluster_summary(path, all_size_lists, n_clusters_per_frame,
                           pair, cluster_cut, count_atom, n_frames):
    all_sizes = []
    for s in all_size_lists: all_sizes.extend(s)
    if not all_sizes: return
    counter = Counter(all_sizes); total = len(all_sizes)
    with open(path, 'w') as f:
        f.write("="*60+"\n CLUSTER ANALYSIS SUMMARY\n"+"="*60+"\n\n")
        f.write("  Pair           : "+pair[0]+"-"+pair[1]+"\n")
        f.write("  Cluster cutoff : "+str(round(cluster_cut,3))+" Angstrom\n")
        f.write("  Counting atom  : "+count_atom+"\n")
        f.write("  Frames         : "+str(n_frames)+"\n\n")
        f.write("  Mean clusters per frame : "+str(round(float(np.mean(n_clusters_per_frame)),2))+"\n")
        f.write("  Mean cluster size       : "+str(round(float(np.mean(all_sizes)),4))+"\n")
        f.write("  Std  cluster size       : "+str(round(float(np.std(all_sizes)),4))+"\n\n")
        f.write("  [ SIZE DISTRIBUTION ]\n")
        f.write("  " + "-"*50 + "\n")
        for n in sorted(counter.keys()):
            p = counter[n]/total; bar = "#"*int(p*40)
            f.write("  n="+str(n).rjust(4)+"  count="+str(counter[n]).rjust(8)+
                    "  P="+str(round(p,4)).rjust(8)+"  "+bar+"\n")
    print("  Cluster summary: " + path)


# =============================================================================
# Be-F-Be ANGLE DISTRIBUTION  (V8 NEW)
# =============================================================================
def analyse_aba_angles_frame(box, pos, symbols, be_atom, f_atom, cluster_cut):
    """
    For every F atom bonded to >= 2 Be atoms within cluster_cut,
    compute all Be-F-Be angles (degrees) using MIC.
    Returns list of ALL angles for this frame (used by --aba-angle plot).
    """
    sym_arr = np.array(symbols)
    idx_be  = np.where(sym_arr == be_atom)[0]
    idx_f   = np.where(sym_arr == f_atom)[0]
    if len(idx_be) == 0 or len(idx_f) == 0:
        return []
    pos_be = pos[idx_be]
    pos_f  = pos[idx_f]

    nl = freud.locality.AABBQuery(box, pos_be).query(pos_f, {"r_max": cluster_cut}).toNeighborList()
    f_to_be = defaultdict(list)
    for k in range(len(nl)):
        i_f  = nl.query_point_indices[k]
        i_be = nl.point_indices[k]
        f_to_be[i_f].append(i_be)

    angles = []
    for i_f, be_list in f_to_be.items():
        if len(be_list) < 2:
            continue
        pF = pos_f[i_f]
        for ii in range(len(be_list)):
            for jj in range(ii+1, len(be_list)):
                pBe1 = pos_be[be_list[ii]]
                pBe2 = pos_be[be_list[jj]]
                v1 = mic_vector(box, pBe1, pF)
                v2 = mic_vector(box, pBe2, pF)
                n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
                if n1 < 1e-10 or n2 < 1e-10: continue
                cos_a = np.dot(v1, v2) / (n1 * n2)
                angles.append(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    return angles


def analyse_aba_angles_tagged_frame(box, pos, symbols, be_atom, f_atom, cluster_cut):
    """
    Same as analyse_aba_angles_frame but returns angles TAGGED by sharing type.
    Classification is by shared F-atom count (geometry-first, no angle threshold).

    Returns:
        edge_angles   : list of Be-F-Be angles where that F is shared by exactly 2 Be
                        (i.e. the pair is edge-sharing)
        corner_angles : list of Be-F-Be angles where that F is shared by exactly 1 Be pair
                        (i.e. the pair is corner-sharing)
        all_angles    : list of all angles combined
    """
    sym_arr = np.array(symbols)
    idx_be  = np.where(sym_arr == be_atom)[0]
    idx_f   = np.where(sym_arr == f_atom)[0]
    if len(idx_be) == 0 or len(idx_f) == 0:
        return [], [], []
    pos_be = pos[idx_be]
    pos_f  = pos[idx_f]

    # Step 1: build F -> list of Be neighbors
    nl = freud.locality.AABBQuery(box, pos_be).query(pos_f, {"r_max": cluster_cut}).toNeighborList()
    f_to_be = defaultdict(list)
    for k in range(len(nl)):
        i_f  = nl.query_point_indices[k]
        i_be = nl.point_indices[k]
        f_to_be[i_f].append(i_be)

    # Step 2: build Be -> set of F neighbors
    be_neighbors = defaultdict(set)
    for i_f, be_list in f_to_be.items():
        for i_be in be_list:
            be_neighbors[i_be].add(i_f)

    # Step 3: for each Be-Be pair that shares F atoms, tag angles by shared count
    # shared=1 → corner, shared=2 → edge
    be_list_all = list(be_neighbors.keys())
    edge_angles = []; corner_angles = []

    for ii in range(len(be_list_all)):
        for jj in range(ii+1, len(be_list_all)):
            i_be = be_list_all[ii]; j_be = be_list_all[jj]
            shared_f = be_neighbors[i_be] & be_neighbors[j_be]
            n_shared = len(shared_f)
            if n_shared == 0:
                continue
            # Compute Be-F-Be angle through each shared F
            for i_f in shared_f:
                pF   = pos_f[i_f]
                pBe1 = pos_be[i_be]
                pBe2 = pos_be[j_be]
                v1 = mic_vector(box, pBe1, pF)
                v2 = mic_vector(box, pBe2, pF)
                n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
                if n1 < 1e-10 or n2 < 1e-10: continue
                cos_a = np.dot(v1, v2) / (n1 * n2)
                angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
                if n_shared == 1:
                    corner_angles.append(angle)
                elif n_shared == 2:
                    edge_angles.append(angle)
                # face-sharing (n_shared>=3) excluded — too rare, would skew stats

    all_angles = edge_angles + corner_angles
    return edge_angles, corner_angles, all_angles


def plot_aba_angle_distribution(all_angles_list, be_atom, f_atom, out_prefix, n_frames, cluster_cut):
    """
    Plain wave-style A-B-A angle distribution plot.
    Only shows the wave line + fill and the pair legend (n=...).
    No shaded regions, no vertical lines, no stats box.
    """
    all_angles = []
    for a in all_angles_list: all_angles.extend(a)
    if not all_angles:
        print("  [aba-angle] No angles found — check cutoff or atom names"); return
    all_angles = np.array(all_angles)

    fig, ax = plt.subplots(figsize=(10, 5))

    counts, bin_edges = np.histogram(all_angles, bins=90, range=(0, 180))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.plot(bin_centers, counts, color='steelblue', lw=2.5,
            label=f'{be_atom}-{f_atom}-{be_atom}  (n={len(all_angles)})', alpha=0.9)
    ax.fill_between(bin_centers, counts, alpha=0.18, color='steelblue')

    ax.set_xlabel(f"{be_atom}-{f_atom}-{be_atom} Angle (degrees)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_title(f"{be_atom}-{f_atom}-{be_atom} Angle Distribution\n"
                 f"Cutoff: {cluster_cut} Å  |  {n_frames} frames  |  {len(all_angles)} angles total",
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 15))
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--')

    plt.tight_layout()
    out = out_prefix + f"_{be_atom}{f_atom}{be_atom}_angle_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print("  ABA angle PNG : " + out)


# =============================================================================
# CORNER vs EDGE SHARING ANALYSIS  (V8 NEW)
# =============================================================================
def analyse_corner_edge_sharing_frame(box, pos, symbols, be_atom, f_atom, cluster_cut):
    """
    For every pair of Be atoms (tetrahedra) that share at least one F neighbor:
      - 1 shared F  -> corner-sharing
      - 2 shared F  -> edge-sharing
      - 3+ shared F -> face-sharing (rare)

    Returns:
        n_corner  : int   number of corner-sharing Be-Be pairs this frame
        n_edge    : int   number of edge-sharing Be-Be pairs this frame
        n_face    : int   number of face-sharing Be-Be pairs this frame
        corner_pairs : list of (i_be_local, j_be_local) for corner pairs
        edge_pairs   : list of (i_be_local, j_be_local) for edge pairs
        labels_be    : np.array of cluster labels for Be atoms
        bond_list_be_f: list of (i_be_local, i_f_local) bonds
    """
    sym_arr = np.array(symbols)
    idx_be  = np.where(sym_arr == be_atom)[0]
    idx_f   = np.where(sym_arr == f_atom)[0]
    if len(idx_be) == 0 or len(idx_f) == 0:
        return 0, 0, 0, [], [], np.array([]), []
    pos_be = pos[idx_be]
    pos_f  = pos[idx_f]

    # Find Be-F bonds within cluster_cut
    nl = freud.locality.AABBQuery(box, pos_f).query(pos_be, {"r_max": cluster_cut}).toNeighborList()
    # be_neighbors[i_be] = set of local F indices bonded to Be i
    be_neighbors = defaultdict(set)
    bond_list_be_f = []
    for k in range(len(nl)):
        i_be = nl.query_point_indices[k]
        i_f  = nl.point_indices[k]
        be_neighbors[i_be].add(i_f)
        bond_list_be_f.append((i_be, i_f))

    # For every pair of Be atoms, count shared F neighbors
    n_corner = 0; n_edge = 0; n_face = 0
    corner_pairs = []; edge_pairs = []
    be_list = list(be_neighbors.keys())
    for ii in range(len(be_list)):
        for jj in range(ii+1, len(be_list)):
            i_be = be_list[ii]; j_be = be_list[jj]
            shared = be_neighbors[i_be] & be_neighbors[j_be]
            n_shared = len(shared)
            if n_shared == 1:
                n_corner += 1; corner_pairs.append((i_be, j_be))
            elif n_shared == 2:
                n_edge   += 1; edge_pairs.append((i_be, j_be))
            elif n_shared >= 3:
                n_face   += 1

    return n_corner, n_edge, n_face, corner_pairs, edge_pairs, bond_list_be_f


def write_corner_edge_summary(path, corner_per_frame, edge_per_frame, face_per_frame,
                               be_atom, f_atom, cluster_cut, n_frames,
                               edge_angles_all=None, corner_angles_all=None):
    corner_arr = np.array(corner_per_frame, dtype=float)
    edge_arr   = np.array(edge_per_frame,   dtype=float)
    face_arr   = np.array(face_per_frame,   dtype=float)
    total_arr  = corner_arr + edge_arr + face_arr

    # Compute data-derived angle statistics
    if edge_angles_all is not None and len(edge_angles_all) > 10:
        ea = np.array(edge_angles_all)
        e_med = np.median(ea); e_lo = np.percentile(ea,5); e_hi = np.percentile(ea,95)
        e_mean = ea.mean(); e_std = ea.std()
        has_edge_angles = True
    else:
        has_edge_angles = False

    if corner_angles_all is not None and len(corner_angles_all) > 10:
        ca = np.array(corner_angles_all)
        c_med = np.median(ca); c_lo = np.percentile(ca,5); c_hi = np.percentile(ca,95)
        c_mean = ca.mean(); c_std = ca.std()
        has_corner_angles = True
    else:
        has_corner_angles = False

    with open(path, 'w') as f:
        f.write("="*60+"\n CORNER / EDGE / FACE SHARING SUMMARY\n"+"="*60+"\n\n")
        f.write(f"  Pair           : {be_atom}-{f_atom}\n")
        f.write(f"  Cluster cutoff : {round(cluster_cut,3)} Angstrom\n")
        f.write(f"  Frames         : {n_frames}\n\n")

        # Publication-grade classification block
        f.write("  " + "-"*56 + "\n")
        f.write("  CLASSIFICATION METHOD\n")
        f.write("  " + "-"*56 + "\n")
        f.write(f"  Method   : Geometry-first (shared F-atom count)\n")
        f.write(f"             NOT based on any angle threshold.\n\n")
        f.write(f"  Rule: corner-sharing = exactly 1 shared F neighbor\n")
        f.write(f"        edge-sharing   = exactly 2 shared F neighbors\n")
        f.write(f"        face-sharing   = 3+ shared F neighbors\n\n")
        f.write(f"  ANGLE DISTRIBUTION (validation — NOT used for classification)\n")
        if has_edge_angles:
            f.write(f"  Edge   median angle : {e_med:.1f} deg\n")
            f.write(f"  Edge   mean  +/- std: {e_mean:.1f} +/- {e_std:.1f} deg\n")
            f.write(f"  Edge   95% range    : {e_lo:.1f} deg -- {e_hi:.1f} deg\n")
        else:
            f.write(f"  Edge   angles       : insufficient data\n")
        if has_corner_angles:
            f.write(f"  Corner median angle : {c_med:.1f} deg\n")
            f.write(f"  Corner mean  +/- std: {c_mean:.1f} +/- {c_std:.1f} deg\n")
            f.write(f"  Corner 95% range    : {c_lo:.1f} deg -- {c_hi:.1f} deg\n")
        else:
            f.write(f"  Corner angles       : insufficient data\n")
        f.write("  " + "-"*56 + "\n\n")

        # Counts table
        f.write(f"  {'Type':<18} {'Mean':>10} {'Std':>10} {'Min':>8} {'Max':>8}\n")
        f.write(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8} {'-'*8}\n")
        for label, arr in [('Corner-sharing', corner_arr),
                            ('Edge-sharing',   edge_arr),
                            ('Face-sharing',   face_arr),
                            ('Total pairs',    total_arr)]:
            f.write(f"  {label:<18} {arr.mean():>10.2f} {arr.std():>10.2f} "
                    f"{arr.min():>8.0f} {arr.max():>8.0f}\n")
        f.write(f"\n")
        mean_total = total_arr.mean()
        if mean_total > 0:
            f.write(f"  Corner fraction : {corner_arr.mean()/mean_total*100:.1f}%\n")
            f.write(f"  Edge   fraction : {edge_arr.mean()/mean_total*100:.1f}%\n")
            f.write(f"  Face   fraction : {face_arr.mean()/mean_total*100:.1f}%\n")
    print("  Corner/Edge summary: " + path)


def plot_corner_edge_angle_distribution(all_angles_list, be_atom, f_atom,
                                         out_prefix, n_frames, cluster_cut,
                                         edge_angles_all=None, corner_angles_all=None):
    """
    Wave-style Be-F-Be angle distribution plot.

    Classification is geometry-first (shared F-atom count) — NOT angle thresholds.
    The angle ranges shown on the plot are COMPUTED from the actual data:
      Edge   region : 5th–95th percentile of angles from edge-sharing pairs
      Corner region : 5th–95th percentile of angles from corner-sharing pairs
    This makes the plot fully data-driven and publication-defensible.
    """
    all_angles = []
    for a in all_angles_list: all_angles.extend(a)
    if not all_angles:
        print("  [corner/edge angle] No angles found"); return
    all_angles = np.array(all_angles)

    # ── Compute ranges from actual tagged angle populations ──
    if edge_angles_all is not None and len(edge_angles_all) > 10:
        ea = np.array(edge_angles_all)
        EDGE_MED  = float(np.median(ea))
        EDGE_LO   = float(np.percentile(ea,  5))
        EDGE_HI   = float(np.percentile(ea, 95))
    else:
        # fallback if no edge angles collected (very rare system)
        EDGE_MED = 90.0; EDGE_LO = 75.0; EDGE_HI = 105.0

    if corner_angles_all is not None and len(corner_angles_all) > 10:
        ca = np.array(corner_angles_all)
        CORNER_MED = float(np.median(ca))
        CORNER_LO  = float(np.percentile(ca,  5))
        CORNER_HI  = float(np.percentile(ca, 95))
    else:
        CORNER_MED = 128.0; CORNER_LO = 105.0; CORNER_HI = 180.0

    fig, ax = plt.subplots(figsize=(11, 5))

    # ── wave (all angles combined) ──
    counts, bin_edges = np.histogram(all_angles, bins=90, range=(0, 180))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ax.plot(bin_centers, counts, color='steelblue', lw=2.5,
            label=f'{be_atom}-{f_atom}-{be_atom}  (n={len(all_angles)})', alpha=0.9)
    ax.fill_between(bin_centers, counts, alpha=0.18, color='steelblue')

    ymax = counts.max()
    n_edge_plot   = int(np.sum((all_angles >= EDGE_LO)   & (all_angles <= EDGE_HI)))
    n_corner_plot = int(np.sum((all_angles >= CORNER_LO) & (all_angles <= CORNER_HI)))
    n_mid_plot    = len(all_angles) - n_edge_plot - n_corner_plot

    # ── Edge-sharing region (data-derived) ──
    ax.axvspan(EDGE_LO, EDGE_HI, alpha=0.13, color='red')
    ax.axvline(EDGE_LO,  color='red', lw=1.5, ls='--', alpha=0.85)
    ax.axvline(EDGE_HI,  color='red', lw=1.5, ls='--', alpha=0.85)
    #ax.axvline(EDGE_MED, color='darkred', lw=2.0, ls='-', alpha=0.9)
    ax.annotate('', xy=(EDGE_HI, ymax*0.88), xytext=(EDGE_LO, ymax*0.88),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.8))
    ax.text((EDGE_LO+EDGE_HI)/2, ymax*0.91,
            f"Edge-sharing\n{EDGE_LO:.1f}\u00b0\u2013{EDGE_HI:.1f}\u00b0 \n"
            f"median {EDGE_MED:.1f}\u00b0  |  2 shared F\n"
            f"population: {n_edge_plot/len(all_angles)*100:.1f}% \u00b1 {n_mid_plot/len(all_angles)*100/2:.1f}%",
            ha='center', va='bottom', fontsize=8, color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose',
                      edgecolor='red', alpha=0.90))

    # ── Corner-sharing region (data-derived) ──
    ax.axvspan(CORNER_LO, CORNER_HI, alpha=0.13, color='green')
    ax.axvline(CORNER_LO,  color='green', lw=1.5, ls='--', alpha=0.85)
    #ax.axvline(CORNER_MED, color='darkgreen', lw=2.0, ls='-', alpha=0.9)
    ax.annotate('', xy=(min(CORNER_HI-1, 179), ymax*0.88), xytext=(CORNER_LO, ymax*0.88),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.8))
    ax.text((CORNER_LO+CORNER_HI)/2, ymax*0.91,
            f"Corner-sharing\n{CORNER_LO:.1f}\u00b0\u2013{CORNER_HI:.1f}\u00b0 \n"
            f"median {CORNER_MED:.1f}\u00b0  |  1 shared F\n"
            f"population: {n_corner_plot/len(all_angles)*100:.1f}% \u00b1 {n_mid_plot/len(all_angles)*100/2:.1f}%",
            ha='center', va='bottom', fontsize=8, color='darkgreen', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='honeydew',
                      edgecolor='green', alpha=0.90))

    # ── Intermediate gap label ──
    mid_x = (EDGE_HI + CORNER_LO) / 2
    ax.text(mid_x, ymax*0.30,
            f"Intermediate\n{EDGE_HI:.1f}\u00b0\u2013{CORNER_LO:.1f}\u00b0\n"
            f"(error: {n_mid_plot/len(all_angles)*100:.1f}%)",
            ha='center', va='center', fontsize=8, color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.75))

    ax.set_xlabel(f"{be_atom}-{f_atom}-{be_atom} Angle (degrees)",
                  fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    ax.set_title(f"{be_atom}-{f_atom}-{be_atom} Angle Distribution  —  Corner vs Edge Sharing\n"
                 f"Cutoff: {cluster_cut} \u00c5  |  {n_frames} frames  |  {len(all_angles)} angles total  |  "
                 f"Regions derived from data",
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 15))
    ax.legend(fontsize=9, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--')

    # ── Stats box ──
    #n_edge_plot   = int(np.sum((all_angles >= EDGE_LO)   & (all_angles <= EDGE_HI)))
    #n_corner_plot = int(np.sum((all_angles >= CORNER_LO) & (all_angles <= CORNER_HI)))
    #n_mid_plot    = len(all_angles) - n_edge_plot - n_corner_plot
    plt.tight_layout()
    out = out_prefix + f"_{be_atom}{f_atom}{be_atom}_corner_edge_angle_dist.png"
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print("  Corner/Edge angle dist PNG : " + out)


def plot_corner_edge_time_series(corner_per_frame, edge_per_frame, face_per_frame,
                                  be_atom, f_atom, out_prefix, start, stride):
    frames = start + np.arange(len(corner_per_frame)) * stride
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frames, corner_per_frame, 'g-o', ms=4, lw=1.8, label='Corner-sharing pairs', alpha=0.85)
    ax.plot(frames, edge_per_frame,   'r-s', ms=4, lw=1.8, label='Edge-sharing pairs',   alpha=0.85)
    if any(v > 0 for v in face_per_frame):
        ax.plot(frames, face_per_frame, 'b-^', ms=4, lw=1.8, label='Face-sharing pairs', alpha=0.85)
    ax.set_xlabel("Frame", fontsize=12, fontweight='bold')
    ax.set_ylabel("Number of pairs", fontsize=12, fontweight='bold')
    ax.set_title(f"{be_atom}-{f_atom}-{be_atom} Corner vs Edge Sharing over Time",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.95); ax.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()
    out = out_prefix + f"_{be_atom}{f_atom}{be_atom}_corner_edge_time_series.png"
    plt.savefig(out, dpi=300, bbox_inches='tight'); plt.close()
    print("  Corner/Edge time series: " + out)


def plot_3d_corner_edge_interactive(box, pos, symbols, be_atom, f_atom,
                                     cluster_cut, out_prefix, frame_num):
    """
    Interactive 3D Plotly HTML showing:
      - Be atoms colored by role: corner-sharing (green), edge-sharing (red), isolated (gray)
      - F atoms (small, silver)
      - Be-F bonds colored by type
      - Corner-sharing Be-Be connections (green dashed)
      - Edge-sharing Be-Be connections (red solid)
    """
    sym_arr = np.array(symbols)
    idx_be  = np.where(sym_arr == be_atom)[0]
    idx_f   = np.where(sym_arr == f_atom)[0]
    if len(idx_be) == 0 or len(idx_f) == 0:
        print("  [corner/edge 3D] atoms not found"); return
    pos_be = pos[idx_be]; pos_f = pos[idx_f]

    n_corner, n_edge, n_face, corner_pairs, edge_pairs, bond_list_be_f = \
        analyse_corner_edge_sharing_frame(box, pos, symbols, be_atom, f_atom, cluster_cut)

    # Classify each Be atom
    corner_be = set(); edge_be = set()
    for i,j in corner_pairs: corner_be.add(i); corner_be.add(j)
    for i,j in edge_pairs:   edge_be.add(i);   edge_be.add(j)
    # edge takes priority over corner
    corner_only = corner_be - edge_be

    be_colors = []
    for i in range(len(pos_be)):
        if i in edge_be:   be_colors.append('red')
        elif i in corner_only: be_colors.append('green')
        else:              be_colors.append('gray')

    fig_p = go.Figure()

    # Be atoms
    for color, label, mask_fn in [
        ('red',   f'Be edge-sharing ({len(edge_be)} atoms)',
         lambda i: be_colors[i]=='red'),
        ('green', f'Be corner-sharing ({len(corner_only)} atoms)',
         lambda i: be_colors[i]=='green'),
        ('gray',  f'Be isolated ({sum(1 for c in be_colors if c=="gray")} atoms)',
         lambda i: be_colors[i]=='gray'),
    ]:
        idx = [i for i in range(len(pos_be)) if mask_fn(i)]
        if not idx: continue
        arr = pos_be[np.array(idx)]
        fig_p.add_trace(go.Scatter3d(
            x=arr[:,0], y=arr[:,1], z=arr[:,2], mode='markers',
            marker=dict(size=10, color=color, line=dict(color='black', width=1)),
            name=label,
            hovertemplate=f'{be_atom} ({label})<br>x=%{{x:.2f}} y=%{{y:.2f}} z=%{{z:.2f}}<extra></extra>'))

    # F atoms
    fig_p.add_trace(go.Scatter3d(
        x=pos_f[:,0], y=pos_f[:,1], z=pos_f[:,2], mode='markers',
        marker=dict(size=5, color='silver', line=dict(color='gray', width=0.5)),
        name=f'{f_atom} atoms',
        hovertemplate=f'{f_atom}<br>x=%{{x:.2f}} y=%{{y:.2f}} z=%{{z:.2f}}<extra></extra>'))

    # Be-F bonds
    for i_be, i_f in bond_list_be_f:
        pB = pos_be[i_be]; pF = pos_be[i_be] + mic_vector(box, pos_f[i_f], pos_be[i_be])
        color = be_colors[i_be]
        fig_p.add_trace(go.Scatter3d(
            x=[pB[0],pF[0],None], y=[pB[1],pF[1],None], z=[pB[2],pF[2],None],
            mode='lines', line=dict(color=color, width=2),
            showlegend=False, hoverinfo='skip'))

    # Corner-sharing Be-Be links (green dashed)
    for i_be, j_be in corner_pairs:
        pA = pos_be[i_be]; pB2 = pos_be[i_be] + mic_vector(box, pos_be[j_be], pos_be[i_be])
        fig_p.add_trace(go.Scatter3d(
            x=[pA[0],pB2[0],None], y=[pA[1],pB2[1],None], z=[pA[2],pB2[2],None],
            mode='lines', line=dict(color='green', width=3, dash='dash'),
            showlegend=False, hoverinfo='skip'))

    # Edge-sharing Be-Be links (red solid)
    for i_be, j_be in edge_pairs:
        pA = pos_be[i_be]; pB2 = pos_be[i_be] + mic_vector(box, pos_be[j_be], pos_be[i_be])
        fig_p.add_trace(go.Scatter3d(
            x=[pA[0],pB2[0],None], y=[pA[1],pB2[1],None], z=[pA[2],pB2[2],None],
            mode='lines', line=dict(color='red', width=4),
            showlegend=False, hoverinfo='skip'))

    fig_p.update_layout(
        title=dict(
            text=(f"Corner/Edge Sharing: {be_atom}-{f_atom} | Frame {frame_num} | "
                  f"Cutoff {round(cluster_cut,2)} Å<br>"
                  f"Corner-sharing pairs: {n_corner} | Edge-sharing pairs: {n_edge} | "
                  f"Face-sharing pairs: {n_face}"),
            font=dict(size=13)),
        scene=dict(
            xaxis=dict(title='X (Å)', backgroundcolor='rgb(245,245,245)'),
            yaxis=dict(title='Y (Å)', backgroundcolor='rgb(245,245,245)'),
            zaxis=dict(title='Z (Å)', backgroundcolor='rgb(235,235,235)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.01,
                    bgcolor='rgba(255,255,240,0.95)', bordercolor='black',
                    borderwidth=1, font=dict(size=11), tracegroupgap=3),
        margin=dict(l=0, r=260, t=150, b=0),
        width=1260, height=900, paper_bgcolor='white')

    html_out = out_prefix + f"_{be_atom}{f_atom}{be_atom}_corner_edge_3D_interactive.html"
    pio.write_html(fig_p, html_out, include_plotlyjs='cdn')
    print("  Corner/Edge HTML : " + html_out)


#  SUMMARY WRITER
def _bar(fraction, width=30, char_full='█', char_empty='░'):
    filled = int(round(fraction * width))
    return char_full * filled + char_empty * (width - filled)

def write_summary(path, pairs, n_frames, args,
                  pair_cn_total, pair_cn_intra, pair_cn_inter, all_distances):
    W = 72  # total line width
    sep  = '=' * W
    sep2 = '-' * W
    thin = '·' * W

    def box_line(text='', fill=' '):
        inner = W - 4
        return f'  | {text:<{inner}} |'

    def section_header(title):
        pad = (W - len(title) - 4) // 2
        return f'  +{"-"*pad} {title} {"-"*(W-4-pad-len(title)-1)}+'

    pairs_str = ', '.join(f'{a}-{b}' for a,b in pairs)
    traj_name = os.path.basename(args.xyz_file)

    with open(path, 'w', encoding='utf-8') as f:

        # ── MAIN HEADER ──
        f.write(f'\n{"#"*W}\n')
        f.write(f'#{"HYDROGEN BOND ANALYSIS  —  FULL SUMMARY REPORT":^{W-2}}#\n')
        if args.mode == "ovito":
            mode_str = "Distance Only — OVITO/V3 compatible (no angle filter)"
        elif args.mode == "vmd":
            mode_str = f"VMD: D...A Distance + Angle = 150 deg fixed (V5)"
        else:
            mode_str = f"Geometric: Distance + Angle >= {args.angle_cut:.0f} deg (V5)"
        f.write(f'#{mode_str:^{W-2}}#\n')
        f.write(f'{"#"*W}\n\n')

        # ── RUN PARAMETERS ──
        f.write(f'  {"RUN PARAMETERS":}\n')
        f.write(f'  {sep2[:W-4]}\n')
        f.write(f'  {"Trajectory":<22}: {traj_name}\n')
        f.write(f'  {"Frames analysed":<22}: {n_frames}\n')
        f.write(f'  {"Atom pairs":<22}: {pairs_str}\n')
        f.write(f'  {"Mode":<22}: {args.mode}\n')
        f.write(f'  {"H-bond cutoff":<22}: {args.cn_cut:.3f} Angstrom\n')
        if args.mode == "geometric":
            f.write(f'  {"Angle cutoff":<22}: {args.angle_cut:.1f} degrees (D-H...A)\n')
            f.write(f'  {"Distance measured":<22}: H...A (hydrogen to acceptor)\n')
        elif args.mode == "vmd":
            f.write(f'  {"Angle cutoff":<22}: 150.0 degrees (D-H...A, fixed)\n')
            f.write(f'  {"Distance measured":<22}: D...A (donor to acceptor, VMD convention)\n')
        else:
            f.write(f'  {"Angle cutoff":<22}: none (distance only)\n')
            f.write(f'  {"Distance measured":<22}: H...A (hydrogen to acceptor)\n')
        f.write(f'  {"Covalent bond cutoff":<22}: {args.bond_cut:.3f} Angstrom\n')
        f.write(f'  {"RDF max distance":<22}: {args.rmax:.3f} Angstrom\n')
        f.write(f'  {"Backbone chains":<22}: {"specified="+str(args.backbone_chains) if args.backbone_chains else "auto top-"+str(args.max_backbone_chains)}\n')
        f.write(f'\n')

        # ── PER-PAIR SECTIONS ──
        for pidx, (a, b) in enumerate(pairs):
            pk = (a, b)
            ct = np.array(pair_cn_total[pk])
            ci = np.array(pair_cn_intra[pk])
            ce = np.array(pair_cn_inter[pk])
            di = all_distances[pk]['intra']
            de = all_distances[pk]['inter']
            pct_i = ci.mean() / ct.mean() * 100 if ct.mean() > 0 else 0.0
            pct_e = ce.mean() / ct.mean() * 100 if ct.mean() > 0 else 0.0

            f.write(f'  {sep}\n')
            if args.mode == "geometric":
                angle_info = f"H...A, angle >= {args.angle_cut:.0f} deg"
            elif args.mode == "vmd":
                angle_info = "D...A, angle = 150 deg fixed (VMD)"
            else:
                angle_info = "H...A, no angle filter (OVITO)"
            f.write(f'  PAIR {pidx+1}/{len(pairs)} :  {a}  —  {b}  '
                    f'(cutoff {args.cn_cut:.2f} Angstrom, {angle_info})\n')
            f.write(f'  {sep}\n\n')

            # ── Coordination Numbers sub-section ──
            f.write(f'  [ COORDINATION NUMBERS ]\n')
            f.write(f'  {thin}\n')
            f.write(f'  {"Quantity":<20}  {"Mean":>10}  {"Std Dev":>10}  {"Share":>8}\n')
            f.write(f'  {"-"*20}  {"-"*10}  {"-"*10}  {"-"*8}\n')
            f.write(f'  {"Total CN":<20}  {ct.mean():>10.4f}  {ct.std():>10.4f}  {"100.0%":>8}\n')
            f.write(f'  {"Intra-chain CN":<20}  {ci.mean():>10.4f}  {ci.std():>10.4f}  {pct_i:>7.1f}%\n')
            f.write(f'  {"Inter-chain CN":<20}  {ce.mean():>10.4f}  {ce.std():>10.4f}  {pct_e:>7.1f}%\n')
            f.write(f'  {thin}\n')
            if ce.mean() > 0:
                f.write(f'  Intra / Inter ratio  :  {ci.mean()/ce.mean():.2f} : 1  '
                        f'  ({"self-healing dominant" if ci.mean()/ce.mean() > 1 else "inter-chain dominant"})\n')
            f.write(f'\n')

            # ── Bond Distances sub-section ──
            f.write(f'  [ BOND DISTANCES ]\n')
            f.write(f'  {thin}\n')
            f.write(f'  {"Type":<14}  {"Mean (Ang)":>12}  {"Std Dev":>10}  {"Min":>8}  {"Max":>8}  {"Count":>10}\n')
            f.write(f'  {"-"*14}  {"-"*12}  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*10}\n')
            if len(di) > 0:
                f.write(f'  {"Intra-chain":<14}  {di.mean():>12.4f}  {di.std():>10.4f}  '
                        f'{di.min():>8.4f}  {di.max():>8.4f}  {len(di):>10,}\n')
            else:
                f.write(f'  {"Intra-chain":<14}  {"N/A":>12}  {"N/A":>10}  {"N/A":>8}  {"N/A":>8}  {0:>10,}\n')
            if len(de) > 0:
                f.write(f'  {"Inter-chain":<14}  {de.mean():>12.4f}  {de.std():>10.4f}  '
                        f'{de.min():>8.4f}  {de.max():>8.4f}  {len(de):>10,}\n')
            else:
                f.write(f'  {"Inter-chain":<14}  {"N/A":>12}  {"N/A":>10}  {"N/A":>8}  {"N/A":>8}  {0:>10,}\n')
            f.write(f'  {thin}\n\n')

            # ── Temporal stats ──
            f.write(f'  [ TEMPORAL STABILITY  ({n_frames} frames) ]\n')
            f.write(f'  {thin}\n')
            f.write(f'  {"Property":<26}  {"Min":>8}  {"Max":>8}  {"Mean":>8}  {"Std":>8}\n')
            f.write(f'  {"-"*26}  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*8}\n')
            f.write(f'  {"CN Total (per frame)":<26}  {ct.min():>8.4f}  {ct.max():>8.4f}  '
                    f'{ct.mean():>8.4f}  {ct.std():>8.4f}\n')
            f.write(f'  {"CN Intra (per frame)":<26}  {ci.min():>8.4f}  {ci.max():>8.4f}  '
                    f'{ci.mean():>8.4f}  {ci.std():>8.4f}\n')
            f.write(f'  {"CN Inter (per frame)":<26}  {ce.min():>8.4f}  {ce.max():>8.4f}  '
                    f'{ce.mean():>8.4f}  {ce.std():>8.4f}\n')
            f.write(f'  {thin}\n\n')

        # ── CROSS-PAIR OVERVIEW ──
        if len(pairs) > 1:
            f.write(f'  {sep}\n')
            f.write(f'  CROSS-PAIR OVERVIEW\n')
            f.write(f'  {sep}\n\n')
            f.write(f'  {"Pair":<10}  {"Total CN":>10}  {"Intra CN":>10}  '
                    f'{"Inter CN":>10}  {"Intra%":>8}  {"Inter%":>8}  {"I/E Ratio":>10}\n')
            f.write(f'  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}  {"-"*8}  {"-"*10}\n')
            for a, b in pairs:
                pk = (a,b)
                ct = np.array(pair_cn_total[pk]); ci = np.array(pair_cn_intra[pk]); ce = np.array(pair_cn_inter[pk])
                ratio = f'{ci.mean()/ce.mean():.2f}:1' if ce.mean()>0 else 'N/A'
                f.write(f'  {a+"-"+b:<10}  {ct.mean():>10.4f}  {ci.mean():>10.4f}  '
                        f'{ce.mean():>10.4f}  {ci.mean()/ct.mean()*100 if ct.mean()>0 else 0:>7.1f}%  '
                        f'{ce.mean()/ct.mean()*100 if ct.mean()>0 else 0:>7.1f}%  {ratio:>10}\n')
            f.write(f'\n')



#  PLOT: per-pair clean 3D
def plot_3d_clean(box, pos, symbols, chain_ids, pairs,
                  cn_cut, out_prefix, frame_num, frame_cn_stats=None,
                  donor_map=None, angle_cut=150.0):
    a, b = pairs[0]
    types, type_names = build_type_map(symbols)
    if a not in type_names or b not in type_names:
        print(f"Cannot create 3D clean: {a} or {b} not found"); return
    ida, idb = type_names.index(a), type_names.index(b)
    sym_arr = np.array(symbols)
    A_pos = pos[types==ida]; B_pos = pos[types==idb]
    A_ch  = chain_ids[types==ida]; B_ch  = chain_ids[types==idb]
    if donor_map is not None:
        A_global_indices = np.where(types==ida)[0]
        B_global_indices = np.where(types==idb)[0]
        intra, inter, _ = get_hbonds_for_pair_v5(
            box, A_pos, B_pos, A_ch, B_ch, cn_cut,
            donor_map, pos, sym_arr, angle_cut, A_global_indices, B_global_indices)
    else:
        intra, inter = get_hbonds_for_pair(box, A_pos, B_pos, A_ch, B_ch, cn_cut)

    nl    = freud.locality.AABBQuery(box,B_pos).query(A_pos,{"r_max":cn_cut}).toNeighborList()
    selA = list({i for i,j in intra} | {i for i,j in inter})
    if not selA: print(f"No {a}-{b} bonds found"); return
    fig = plt.figure(figsize=(13,9)); ax = fig.add_subplot(111,projection='3d')
    for i in selA:
        ax.scatter(*A_pos[i], c='red', s=350, edgecolors='darkred', lw=3, alpha=0.95, zorder=10)
    pB_intra, pB_inter = set(), set()
    for i,j in intra:
        Bp = A_pos[i]+mic_vector(box,B_pos[j],A_pos[i])
        if j not in pB_intra and j not in pB_inter:
            ax.scatter(*Bp, c='limegreen', s=200, edgecolors='darkgreen', lw=2.5, alpha=0.9)
            pB_intra.add(j)
        ax.plot([A_pos[i,0],Bp[0]],[A_pos[i,1],Bp[1]],[A_pos[i,2],Bp[2]],'g-',lw=3.5,alpha=0.85)
    for i,j in inter:
        Bp = A_pos[i]+mic_vector(box,B_pos[j],A_pos[i])
        if j not in pB_intra and j not in pB_inter:
            ax.scatter(*Bp, c='cyan', s=200, edgecolors='darkblue', lw=2.5, alpha=0.9)
            pB_inter.add(j)
        ax.plot([A_pos[i,0],Bp[0]],[A_pos[i,1],Bp[1]],[A_pos[i,2],Bp[2]],'b--',lw=3.5,alpha=0.85)
    for i in selA:
        ax.text(*A_pos[i]+[0,0,0.6],f'{A_ch[i]}',fontsize=8,color='black',ha='center',va='bottom',
                bbox=dict(boxstyle='round,pad=0.3',facecolor='white',edgecolor='red',alpha=0.7,lw=1))
    ax.set_xlabel('X (A)',fontsize=13,fontweight='bold'); ax.set_ylabel('Y (A)',fontsize=13,fontweight='bold')
    ax.set_zlabel('Z (A)',fontsize=13,fontweight='bold')
    ax.set_title(f'H-Bond Network: {a}-{b} | Frame {frame_num} | Cutoff {cn_cut:.2f} A | Angle >={angle_cut:.0f} deg',
                 fontsize=14,fontweight='bold')
    leg=[Patch(facecolor='red',edgecolor='darkred',label=f'{a} donors'),
         Patch(facecolor='limegreen',edgecolor='darkgreen',label=f'{b} intra-chain'),
         Patch(facecolor='cyan',edgecolor='darkblue',label=f'{b} inter-chain'),
         Line2D([0],[0],color='green',lw=3,linestyle='-',label='Intra bond'),
         Line2D([0],[0],color='blue',lw=3,linestyle='--',label='Inter bond')]
    ax.legend(handles=leg,fontsize=10,loc='upper left',framealpha=0.95)
    uch = np.unique(A_ch[selA])
    cn_txt = f'CN intra={frame_cn_stats[0]:.3f} inter={frame_cn_stats[1]:.3f}' if frame_cn_stats else ''
    ax.text2D(0.02,0.02,
              f'{len(selA)} {a} donors | {len(intra)} intra | {len(inter)} inter\n'
              f'{len(uch)} chains | {cn_txt}',
              transform=ax.transAxes,fontsize=9,verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.5',facecolor='lightyellow',edgecolor='black',alpha=0.85))
    ax.view_init(elev=25,azim=40); plt.tight_layout()
    plt.savefig(f'{out_prefix}_3D_bonds.png',dpi=300,bbox_inches='tight'); plt.close()
    print(f"  Clean 3D PNG : {out_prefix}_3D_bonds.png  ({len(intra)} intra + {len(inter)} inter)")

    # ── PLOTLY INTERACTIVE HTML ──
    fig_p = go.Figure()

    # Donor atoms (A) with chain labels
    fig_p.add_trace(go.Scatter3d(
        x=A_pos[selA, 0], y=A_pos[selA, 1], z=A_pos[selA, 2],
        mode='markers+text',
        marker=dict(size=12, color='red', line=dict(color='darkred', width=2)),
        text=[f'Chain {A_ch[i]}' for i in selA],
        textposition='top center', textfont=dict(size=9, color='black'),
        name=f'{a} donors',
        hovertemplate=(f'{a} donor<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>'
                       f'z=%{{z:.2f}}<br>Chain: %{{text}}<extra></extra>')))

    # Intra acceptor atoms
    intra_B_idx = list(pB_intra)
    if intra_B_idx:
        fig_p.add_trace(go.Scatter3d(
            x=B_pos[intra_B_idx, 0], y=B_pos[intra_B_idx, 1], z=B_pos[intra_B_idx, 2],
            mode='markers',
            marker=dict(size=7, color='limegreen', line=dict(color='darkgreen', width=1)),
            name=f'{b} intra-chain',
            hovertemplate=(f'{b} (intra)<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>')))

    # Inter acceptor atoms (MIC-corrected positions)
    inter_B_mic = {}
    for i, j in inter:
        if j not in inter_B_mic:
            inter_B_mic[j] = A_pos[i] + mic_vector(box, B_pos[j], A_pos[i])
    if inter_B_mic:
        mic_arr = np.array(list(inter_B_mic.values()))
        fig_p.add_trace(go.Scatter3d(
            x=mic_arr[:, 0], y=mic_arr[:, 1], z=mic_arr[:, 2],
            mode='markers',
            marker=dict(size=7, color='cyan', line=dict(color='darkblue', width=1)),
            name=f'{b} inter-chain',
            hovertemplate=(f'{b} (inter)<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>')))

    # Dummy legend-only traces for bond lines
    fig_p.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
        line=dict(color='green', width=5), name=f'Intra H-bond ({len(intra)})',
        legendgroup='bonds', showlegend=True))
    fig_p.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
        line=dict(color='blue', width=5, dash='dash'), name=f'Inter H-bond ({len(inter)})',
        legendgroup='bonds', showlegend=True))

    # Actual bond lines
    for i, j in intra:
        Bp = A_pos[i] + mic_vector(box, B_pos[j], A_pos[i])
        fig_p.add_trace(go.Scatter3d(
            x=[A_pos[i,0], Bp[0], None], y=[A_pos[i,1], Bp[1], None],
            z=[A_pos[i,2], Bp[2], None], mode='lines',
            line=dict(color='green', width=5),
            legendgroup='bonds', showlegend=False, hoverinfo='skip'))
    for i, j in inter:
        Bp = A_pos[i] + mic_vector(box, B_pos[j], A_pos[i])
        fig_p.add_trace(go.Scatter3d(
            x=[A_pos[i,0], Bp[0], None], y=[A_pos[i,1], Bp[1], None],
            z=[A_pos[i,2], Bp[2], None], mode='lines',
            line=dict(color='blue', width=5, dash='dash'),
            legendgroup='bonds', showlegend=False, hoverinfo='skip'))

    cn_txt_p = (f'CN intra={frame_cn_stats[0]:.3f} | inter={frame_cn_stats[1]:.3f}'
                if frame_cn_stats else '')
    fig_p.update_layout(
        title=dict(
            text=(f'H-Bond Network: {a}-{b} | Frame {frame_num} | Cutoff {cn_cut:.2f} A | Angle >={angle_cut:.0f} deg<br>'
                  f'{len(selA)} {a} donors | {len(intra)} intra | {len(inter)} inter | '
                  f'{len(uch)} chains<br>{cn_txt_p}'),
            font=dict(size=13)),
        scene=dict(
            xaxis=dict(title='X (A)', backgroundcolor='rgb(245,245,245)'),
            yaxis=dict(title='Y (A)', backgroundcolor='rgb(245,245,245)'),
            zaxis=dict(title='Z (A)', backgroundcolor='rgb(235,235,235)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=1.01,
                    bgcolor='rgba(255,255,240,0.95)', bordercolor='black', borderwidth=1,
                    font=dict(size=11), tracegroupgap=4),
        margin=dict(l=0, r=220, t=130, b=0),
        width=1200, height=860, paper_bgcolor='white')

    html_out = f'{out_prefix}_3D_bonds_interactive.html'
    pio.write_html(fig_p, html_out, include_plotlyjs='cdn')
    print(f"  Clean HTML   : {html_out}")



#  PLOT: per-pair backbone (HTML + PNG)  
def plot_3d_backbone(box, pos, symbols, chain_ids, pairs,
                     cn_cut, out_prefix, frame_num,
                     frame_cn_stats=None, max_chains=3,
                     specific_chains=None, bond_cut=1.8,
                     donor_map=None, angle_cut=150.0):
    a, b = pairs[0]
    types, type_names = build_type_map(symbols)
    sym_arr = np.array(symbols)
    if a not in type_names or b not in type_names:
        print(f"Cannot create backbone: {a} or {b} not found"); return
    ida, idb = type_names.index(a), type_names.index(b)
    A_pos = pos[types==ida]; B_pos = pos[types==idb]
    A_ch  = chain_ids[types==ida]; B_ch  = chain_ids[types==idb]

    # V5: use angle-filtered bond finder if donor_map is available
    if donor_map is not None:
        A_global_indices = np.where(types==ida)[0]
        B_global_indices = np.where(types==idb)[0]
        intra, inter, _ = get_hbonds_for_pair_v5(
            box, A_pos, B_pos, A_ch, B_ch, cn_cut,
            donor_map, pos, sym_arr, angle_cut, A_global_indices, B_global_indices)
    else:
        intra, inter = get_hbonds_for_pair(box, A_pos, B_pos, A_ch, B_ch, cn_cut)

    if not inter: print(f"No inter-chain bonds for {a}-{b} — backbone skipped"); return

    if specific_chains is not None:
        top = [int(c) for c in specific_chains]
    else:
        score = Counter()
        for i,j in inter: score[int(A_ch[i])]+=1; score[int(B_ch[j])]+=1
        tot  = len(score); actual = min(max_chains, tot)
        if tot < max_chains:
            print(f"  [backbone {a}-{b}] WARNING: only {tot} chains have inter bonds "
                  f"(requested {max_chains})")
        top = [c for c,_ in score.most_common(actual)]
        print(f"  [backbone {a}-{b}] chains: {top}")

    top_set = set(top)
    cmap    = {c: CHAIN_PALETTE[i%len(CHAIN_PALETTE)] for i,c in enumerate(top)}

    sel_mask = np.isin(chain_ids, list(top_set))
    sp, ss, sc = pos[sel_mask], sym_arr[sel_mask], chain_ids[sel_mask]

    nl_bb = freud.locality.AABBQuery(box,sp).query(sp,{"r_max":bond_cut,"exclude_ii":True}).toNeighborList()
    bb_bonds = []
    for k in range(len(nl_bb)):
        i,j,d = nl_bb.query_point_indices[k], nl_bb.point_indices[k], nl_bb.distances[k]
        if d <= PAIR_CUTOFFS.get(frozenset([ss[i],ss[j]]),0.00) and i<j: bb_bonds.append((i,j))

    intra_sel = [(i,j) for i,j in intra if int(A_ch[i]) in top_set]
    inter_sel = [(i,j) for i,j in inter if int(A_ch[i]) in top_set or int(B_ch[j]) in top_set]

    mode_lbl = f"Specified: {top}" if specific_chains else f"Auto top-{len(top)}: {top}"
    cn_ann   = (f'CN intra={frame_cn_stats[0]:.3f} | inter={frame_cn_stats[1]:.3f}'
                if frame_cn_stats else '')

    # ── MATPLOTLIB PNG ──
    fig = plt.figure(figsize=(15,10)); ax = fig.add_subplot(111,projection='3d')
    for idx in range(len(sp)):
        p,s,c = sp[idx], ss[idx], sc[idx]
        if   s=='H': ax.scatter(*p, c='silver',   s=12,  alpha=0.30, zorder=1)
        elif s=='C': ax.scatter(*p, c=cmap.get(c,'gray'), s=70, alpha=0.60, zorder=2)
        elif s=='O': ax.scatter(*p, c='red',       s=150, alpha=0.95, edgecolors='darkred',  lw=1.8, zorder=5)
        elif s=='N': ax.scatter(*p, c='royalblue', s=120, alpha=0.90, edgecolors='darkblue', lw=1.5, zorder=4)
    for i,j in bb_bonds:
        pj = sp[i]+mic_vector(box,sp[j],sp[i])
        ax.plot([sp[i,0],pj[0]],[sp[i,1],pj[1]],[sp[i,2],pj[2]],'-',color='gray',lw=0.9,alpha=0.30)
    for i,j in intra_sel:
        Bp = A_pos[i]+mic_vector(box,B_pos[j],A_pos[i])
        ax.plot([A_pos[i,0],Bp[0]],[A_pos[i,1],Bp[1]],[A_pos[i,2],Bp[2]],'g-',lw=2.5,alpha=0.80)
    for i,j in inter_sel:
        Bp = A_pos[i]+mic_vector(box,B_pos[j],A_pos[i])
        ax.plot([A_pos[i,0],Bp[0]],[A_pos[i,1],Bp[1]],[A_pos[i,2],Bp[2]],'b--',lw=3.0,alpha=0.90)
    Asel = np.isin(A_ch, list(top_set))
    for p,cid in zip(A_pos[Asel], A_ch[Asel]):
        ax.text(*p+[0,0,0.6],f'{cid}',fontsize=7,color='black',ha='center',va='bottom',
                bbox=dict(boxstyle='round,pad=0.2',facecolor='white',edgecolor='red',alpha=0.65,lw=0.8))
    ax.set_xlabel('X (A)',fontsize=12,fontweight='bold')
    ax.set_ylabel('Y (A)',fontsize=12,fontweight='bold')
    ax.set_zlabel('Z (A)',fontsize=12,fontweight='bold')
    ax.set_title(f'Backbone + H-Bond: {a}-{b} | Frame {frame_num} | Angle >={angle_cut:.0f} deg\n{mode_lbl}',
                 fontsize=12,fontweight='bold',pad=12)
    leg=[Patch(facecolor='red',edgecolor='darkred',label='O atoms'),
         Patch(facecolor='royalblue',edgecolor='darkblue',label='N atoms'),
         Patch(facecolor='silver',edgecolor='gray',lw=1.5,label='H atoms'),
         Line2D([0],[0],color='gray',lw=1.5,label='Covalent backbone'),
         Line2D([0],[0],color='green',lw=2.5,linestyle='-',
                label=f'Intra H-bond ({len(intra_sel)})'),
         Line2D([0],[0],color='blue',lw=2.5,linestyle='--',
                label=f'Inter H-bond ({len(inter_sel)})')]
    for c in top: leg.append(Patch(facecolor=cmap[c], label=f'Chain {c} (C)'))
    ax.legend(handles=leg, fontsize=8.5, loc='upper left', framealpha=0.95,
              edgecolor='black', fancybox=True)
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.text2D(0.01,0.01,
              f'Chains: {top}\nAtoms: {len(sp)} | Cov bonds: {len(bb_bonds)}\n'
              f'Intra: {len(intra_sel)} | Inter: {len(inter_sel)}\n{cn_ann}',
              transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.5',facecolor='lightyellow',edgecolor='black',alpha=0.85,lw=1.2))
    ax.view_init(elev=25,azim=40); plt.tight_layout()
    png_out = f'{out_prefix}_3D_backbone.png'
    plt.savefig(png_out,dpi=300,bbox_inches='tight'); plt.close()
    print(f"  Backbone PNG : {png_out}")

    # ── PLOTLY HTML ──
    fig_p = go.Figure()
    for lbl,mask,col,sz,opcty in [
        ('O atoms', ss=='O', 'red',      10, 1.0),
        ('N atoms', ss=='N', 'royalblue', 8, 1.0),
        ('H atoms', ss=='H', 'silver',    3, 0.45)]:
        if np.any(mask):
            fig_p.add_trace(go.Scatter3d(
                x=sp[mask,0],y=sp[mask,1],z=sp[mask,2], mode='markers',
                marker=dict(size=sz,color=col,opacity=opcty),
                name=lbl, legendgroup='atoms',
                hovertemplate=f'{lbl}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>'))
    for c in top:
        cm_ = (ss=='C')&(sc==c)
        if np.any(cm_):
            fig_p.add_trace(go.Scatter3d(
                x=sp[cm_,0],y=sp[cm_,1],z=sp[cm_,2], mode='markers',
                marker=dict(size=5,color=cmap[c]),
                name=f'Chain {c} (C)', legendgroup=f'ch_{c}',
                hovertemplate=f'C chain {c}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>'))
    # dummy legend lines
    for nm,col,dash_ in [('Covalent backbone','gray','solid'),
                          (f'Intra H-bond ({len(intra_sel)})','green','solid'),
                          (f'Inter H-bond ({len(inter_sel)})','blue','dash')]:
        fig_p.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode='lines',
            line=dict(color=col,width=4,dash=dash_),
            name=nm, legendgroup='bonds', showlegend=True))
    for i,j in bb_bonds:
        pj = sp[i]+mic_vector(box,sp[j],sp[i])
        fig_p.add_trace(go.Scatter3d(
            x=[sp[i,0],pj[0],None],y=[sp[i,1],pj[1],None],z=[sp[i,2],pj[2],None],
            mode='lines',line=dict(color='gray',width=2),
            legendgroup='bonds',showlegend=False,hoverinfo='skip'))
    for i,j in intra_sel:
        Bp=A_pos[i]+mic_vector(box,B_pos[j],A_pos[i])
        fig_p.add_trace(go.Scatter3d(
            x=[A_pos[i,0],Bp[0],None],y=[A_pos[i,1],Bp[1],None],z=[A_pos[i,2],Bp[2],None],
            mode='lines',line=dict(color='green',width=5),
            legendgroup='bonds',showlegend=False,hoverinfo='skip'))
    for i,j in inter_sel:
        Bp=A_pos[i]+mic_vector(box,B_pos[j],A_pos[i])
        fig_p.add_trace(go.Scatter3d(
            x=[A_pos[i,0],Bp[0],None],y=[A_pos[i,1],Bp[1],None],z=[A_pos[i,2],Bp[2],None],
            mode='lines',line=dict(color='blue',width=5,dash='dash'),
            legendgroup='bonds',showlegend=False,hoverinfo='skip'))
    fig_p.update_layout(
        title=dict(text=(f'Backbone + H-Bond: {a}-{b} | Frame {frame_num} | Angle >={angle_cut:.0f} deg<br>'
                         f'{mode_lbl} | {len(inter_sel)} inter | {len(intra_sel)} intra | {cn_ann}'),
                   font=dict(size=13)),
        scene=dict(xaxis=dict(title='X (A)',backgroundcolor='rgb(245,245,245)'),
                   yaxis=dict(title='Y (A)',backgroundcolor='rgb(245,245,245)'),
                   zaxis=dict(title='Z (A)',backgroundcolor='rgb(235,235,235)'),
                   camera=dict(eye=dict(x=1.5,y=1.5,z=1.2))),
        legend=dict(yanchor='top',y=0.99,xanchor='left',x=1.01,
                    bgcolor='rgba(255,255,240,0.95)',bordercolor='black',borderwidth=1,
                    font=dict(size=11),tracegroupgap=4),
        margin=dict(l=0,r=220,t=120,b=0), width=1200,height=860,paper_bgcolor='white')
    html_out = f'{out_prefix}_3D_backbone_interactive.html'
    pio.write_html(fig_p, html_out, include_plotlyjs='cdn')
    print(f"  Backbone HTML: {html_out}")


#  PLOT: COMBINED backbone (HTML + PNG) — ALL PAIRS TOGETHER
def plot_3d_backbone_combined(box, pos, symbols, chain_ids, pairs,
                              cn_cut, out_prefix, frame_num,
                              frame_cn_stats_dict=None,
                              max_chains=3, specific_chains=None, bond_cut=1.8,
                              donor_map=None, angle_cut=150.0):
    types, type_names = build_type_map(symbols)
    sym_arr = np.array(symbols)

    pair_bonds = {}
    all_inter_combined = []
    for a,b in pairs:
        if a not in type_names or b not in type_names: continue
        ida,idb = type_names.index(a), type_names.index(b)
        Ap = pos[types==ida]; Bp = pos[types==idb]
        Ac = chain_ids[types==ida]; Bc = chain_ids[types==idb]
        # V5: use angle-filtered bond finder if donor_map is available
        if donor_map is not None:
            A_global_indices = np.where(types==ida)[0]
            B_global_indices = np.where(types==idb)[0]
            intra, inter, _ = get_hbonds_for_pair_v5(
                box, Ap, Bp, Ac, Bc, cn_cut,
                donor_map, pos, sym_arr, angle_cut, A_global_indices, B_global_indices)
        else:
            intra, inter = get_hbonds_for_pair(box, Ap, Bp, Ac, Bc, cn_cut)
        pair_bonds[(a,b)] = dict(intra=intra,inter=inter,A_pos=Ap,B_pos=Bp,A_ch=Ac,B_ch=Bc)
        all_inter_combined.extend([(int(Ac[i]),int(Bc[j])) for i,j in inter])

    if not pair_bonds or not all_inter_combined:
        print("No inter-chain bonds across any pair — combined backbone skipped"); return

    if specific_chains is not None:
        top = [int(c) for c in specific_chains]
        print(f"[combined] Specified chains: {top}")
    else:
        score = Counter()
        for c1,c2 in all_inter_combined: score[c1]+=1; score[c2]+=1
        tot = len(score); actual = min(max_chains, tot)
        if tot < max_chains:
            print(f"[combined] WARNING: only {tot} chains have inter bonds (requested {max_chains})")
        top = [c for c,_ in score.most_common(actual)]
        print(f"[combined] Top chains (all pairs combined): {top}")

    top_set = set(top)
    cmap    = {c: CHAIN_PALETTE[i%len(CHAIN_PALETTE)] for i,c in enumerate(top)}

    sel_mask = np.isin(chain_ids, list(top_set))
    sp, ss, sc = pos[sel_mask], sym_arr[sel_mask], chain_ids[sel_mask]

    nl_bb = freud.locality.AABBQuery(box,sp).query(sp,{"r_max":bond_cut,"exclude_ii":True}).toNeighborList()
    bb_bonds = []
    for k in range(len(nl_bb)):
        i,j,d = nl_bb.query_point_indices[k], nl_bb.point_indices[k], nl_bb.distances[k]
        if d <= PAIR_CUTOFFS.get(frozenset([ss[i],ss[j]]),0.00) and i<j: bb_bonds.append((i,j))

    pair_bonds_sel = {}
    for (a,b),bd in pair_bonds.items():
        pair_bonds_sel[(a,b)] = dict(
            intra=[(i,j) for i,j in bd['intra'] if int(bd['A_ch'][i]) in top_set],
            inter=[(i,j) for i,j in bd['inter'] if int(bd['A_ch'][i]) in top_set
                   or int(bd['B_ch'][j]) in top_set])

    pairs_lbl  = " + ".join(f'{a}-{b}' for a,b in pairs)
    mode_lbl   = f"Specified: {top}" if specific_chains else f"Auto top-{len(top)}: {top}"
    cn_lines   = []
    if frame_cn_stats_dict:
        for (a,b),(ci,ce) in frame_cn_stats_dict.items():
            cn_lines.append(f'{a}-{b}: intra={ci:.3f} inter={ce:.3f}')
    cn_ann = '  |  '.join(cn_lines)

    # ── MATPLOTLIB PNG ──
    fig = plt.figure(figsize=(17,11)); ax = fig.add_subplot(111,projection='3d')
    for idx in range(len(sp)):
        p,s,c = sp[idx],ss[idx],sc[idx]
        if   s=='H': ax.scatter(*p,c='silver',   s=12, alpha=0.30,zorder=1)
        elif s=='C': ax.scatter(*p,c=cmap.get(c,'gray'),s=70,alpha=0.60,zorder=2)
        elif s=='O': ax.scatter(*p,c='red',       s=150,alpha=0.95,edgecolors='darkred', lw=1.8,zorder=5)
        elif s=='N': ax.scatter(*p,c='royalblue', s=120,alpha=0.90,edgecolors='darkblue',lw=1.5,zorder=4)
    for i,j in bb_bonds:
        pj=sp[i]+mic_vector(box,sp[j],sp[i])
        ax.plot([sp[i,0],pj[0]],[sp[i,1],pj[1]],[sp[i,2],pj[2]],'-',color='gray',lw=0.9,alpha=0.30)
    for pidx,(a,b) in enumerate(pairs):
        if (a,b) not in pair_bonds: continue
        bd,bsel = pair_bonds[(a,b)], pair_bonds_sel[(a,b)]
        pc = PAIR_COLORS[pidx%len(PAIR_COLORS)]
        Ap,Bp = bd['A_pos'],bd['B_pos']
        for i,j in bsel['intra']:
            Bpl=Ap[i]+mic_vector(box,Bp[j],Ap[i])
            ax.plot([Ap[i,0],Bpl[0]],[Ap[i,1],Bpl[1]],[Ap[i,2],Bpl[2]],
                    color=pc['intra'],lw=2.5,linestyle=pc['intra_ls'],alpha=0.80)
        for i,j in bsel['inter']:
            Bpl=Ap[i]+mic_vector(box,Bp[j],Ap[i])
            ax.plot([Ap[i,0],Bpl[0]],[Ap[i,1],Bpl[1]],[Ap[i,2],Bpl[2]],
                    color=pc['inter'],lw=3.0,linestyle=pc['inter_ls'],alpha=0.90)
    for pidx,(a,b) in enumerate(pairs):
        if (a,b) not in pair_bonds: continue
        bd = pair_bonds[(a,b)]; Ap,Ac = bd['A_pos'],bd['A_ch']
        for p,cid in zip(Ap[np.isin(Ac,list(top_set))], Ac[np.isin(Ac,list(top_set))]):
            ax.text(*p+[0,0,0.55],f'{cid}',fontsize=7,color='black',ha='center',va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2',facecolor='white',edgecolor='red',alpha=0.65,lw=0.8))
    ax.set_xlabel('X (A)',fontsize=12,fontweight='bold')
    ax.set_ylabel('Y (A)',fontsize=12,fontweight='bold')
    ax.set_zlabel('Z (A)',fontsize=12,fontweight='bold')
    ax.set_title(f'Combined Backbone + H-Bond: {pairs_lbl} | Angle >={angle_cut:.0f} deg\nFrame {frame_num} | {mode_lbl}\n{cn_ann}',
                 fontsize=11,fontweight='bold',pad=12)
    leg=[Patch(facecolor='red',edgecolor='darkred',label='O atoms'),
         Patch(facecolor='royalblue',edgecolor='darkblue',label='N atoms'),
         Patch(facecolor='silver',edgecolor='gray',lw=1.5,label='H atoms'),
         Line2D([0],[0],color='gray',lw=1.5,label='Covalent backbone')]
    for pidx,(a,b) in enumerate(pairs):
        bsel = pair_bonds_sel.get((a,b),{})
        pc   = PAIR_COLORS[pidx%len(PAIR_COLORS)]
        leg.append(Line2D([0],[0],color=pc['intra'],lw=2.5,linestyle=pc['intra_ls'],
                          label=f'{a}-{b} intra ({len(bsel.get("intra",[]))} bonds)'))
        leg.append(Line2D([0],[0],color=pc['inter'],lw=2.5,linestyle=pc['inter_ls'],
                          label=f'{a}-{b} inter ({len(bsel.get("inter",[]))} bonds)'))
    for c in top: leg.append(Patch(facecolor=cmap[c],label=f'Chain {c} (C)'))
    ax.legend(handles=leg,fontsize=8.5,loc='upper left',framealpha=0.95,edgecolor='black',fancybox=True)
    ax.grid(True,alpha=0.2,linestyle=':')
    ti = sum(len(pair_bonds_sel.get((a,b),{}).get('intra',[])) for a,b in pairs)
    te = sum(len(pair_bonds_sel.get((a,b),{}).get('inter',[])) for a,b in pairs)
    ax.text2D(0.01,0.01,f'Pairs: {pairs_lbl}\nChains: {top}\nAtoms: {len(sp)}\n'
              f'Cov bonds: {len(bb_bonds)}\nTotal intra: {ti} | Total inter: {te}',
              transform=ax.transAxes,fontsize=8,verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.5',facecolor='lightyellow',edgecolor='black',alpha=0.85,lw=1.2))
    ax.view_init(elev=25,azim=40); plt.tight_layout()
    png_out = f'{out_prefix}_combined_backbone.png'
    plt.savefig(png_out,dpi=300,bbox_inches='tight'); plt.close()
    print(f"  Combined PNG : {png_out}")

    # ── PLOTLY HTML ──
    fig_p = go.Figure()
    for lbl,mask,col,sz,opc in [('O atoms',ss=='O','red',10,1.0),
                                  ('N atoms',ss=='N','royalblue',8,1.0),
                                  ('H atoms',ss=='H','silver',3,0.45)]:
        if np.any(mask):
            fig_p.add_trace(go.Scatter3d(x=sp[mask,0],y=sp[mask,1],z=sp[mask,2],
                mode='markers',marker=dict(size=sz,color=col,opacity=opc),
                name=lbl,legendgroup='atoms',
                hovertemplate=f'{lbl}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>'))
    for c in top:
        cm_ = (ss=='C')&(sc==c)
        if np.any(cm_):
            fig_p.add_trace(go.Scatter3d(x=sp[cm_,0],y=sp[cm_,1],z=sp[cm_,2],
                mode='markers',marker=dict(size=5,color=cmap[c]),
                name=f'Chain {c} (C)',legendgroup=f'ch_{c}',
                hovertemplate=f'C chain {c}<br>x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<extra></extra>'))
    fig_p.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode='lines',
        line=dict(color='gray',width=2),name='Covalent backbone',legendgroup='bonds',showlegend=True))
    for pidx,(a,b) in enumerate(pairs):
        if (a,b) not in pair_bonds: continue
        pc   = PAIR_COLORS[pidx%len(PAIR_COLORS)]
        bsel = pair_bonds_sel.get((a,b),{})
        fig_p.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode='lines',
            line=dict(color=pc['intra'],width=5),
            name=f'{a}-{b} intra ({len(bsel.get("intra",[]))} bonds)',
            legendgroup=f'b_{a}{b}',showlegend=True))
        fig_p.add_trace(go.Scatter3d(x=[None],y=[None],z=[None],mode='lines',
            line=dict(color=pc['inter'],width=5,dash='dash'),
            name=f'{a}-{b} inter ({len(bsel.get("inter",[]))} bonds)',
            legendgroup=f'b_{a}{b}',showlegend=True))
    for i,j in bb_bonds:
        pj=sp[i]+mic_vector(box,sp[j],sp[i])
        fig_p.add_trace(go.Scatter3d(x=[sp[i,0],pj[0],None],y=[sp[i,1],pj[1],None],
            z=[sp[i,2],pj[2],None],mode='lines',line=dict(color='gray',width=2),
            legendgroup='bonds',showlegend=False,hoverinfo='skip'))
    for pidx,(a,b) in enumerate(pairs):
        if (a,b) not in pair_bonds: continue
        bd,bsel = pair_bonds[(a,b)],pair_bonds_sel[(a,b)]
        pc = PAIR_COLORS[pidx%len(PAIR_COLORS)]
        Ap,Bp = bd['A_pos'],bd['B_pos']
        for i,j in bsel['intra']:
            Bpl=Ap[i]+mic_vector(box,Bp[j],Ap[i])
            fig_p.add_trace(go.Scatter3d(x=[Ap[i,0],Bpl[0],None],y=[Ap[i,1],Bpl[1],None],
                z=[Ap[i,2],Bpl[2],None],mode='lines',line=dict(color=pc['intra'],width=5),
                legendgroup=f'b_{a}{b}',showlegend=False,hoverinfo='skip'))
        for i,j in bsel['inter']:
            Bpl=Ap[i]+mic_vector(box,Bp[j],Ap[i])
            fig_p.add_trace(go.Scatter3d(x=[Ap[i,0],Bpl[0],None],y=[Ap[i,1],Bpl[1],None],
                z=[Ap[i,2],Bpl[2],None],mode='lines',line=dict(color=pc['inter'],width=5,dash='dash'),
                legendgroup=f'b_{a}{b}',showlegend=False,hoverinfo='skip'))
    fig_p.update_layout(
        title=dict(text=(f'Combined Backbone: {pairs_lbl} | Frame {frame_num} | Angle >={angle_cut:.0f} deg<br>'
                         f'{mode_lbl}<br>Total inter: {te} | Total intra: {ti}<br>{cn_ann}'),
                   font=dict(size=13)),
        scene=dict(xaxis=dict(title='X (A)',backgroundcolor='rgb(245,245,245)'),
                   yaxis=dict(title='Y (A)',backgroundcolor='rgb(245,245,245)'),
                   zaxis=dict(title='Z (A)',backgroundcolor='rgb(235,235,235)'),
                   camera=dict(eye=dict(x=1.5,y=1.5,z=1.2))),
        legend=dict(yanchor='top',y=0.99,xanchor='left',x=1.01,
                    bgcolor='rgba(255,255,240,0.95)',bordercolor='black',borderwidth=1,
                    font=dict(size=11),tracegroupgap=4),
        margin=dict(l=0,r=230,t=160,b=0),width=1250,height=900,paper_bgcolor='white')
    html_out = f'{out_prefix}_combined_backbone_interactive.html'
    pio.write_html(fig_p, html_out, include_plotlyjs='cdn')
    print(f"  Combined HTML: {html_out}")


#  TIME SERIES
def plot_time_series(pair_cn_intra, pair_cn_inter, start, stride, out_prefix):
    n = len(pair_cn_intra)
    fig, axes = plt.subplots(n, 1, figsize=(11, 4.5*n))
    if n==1: axes=[axes]
    for (pk,ax) in zip(pair_cn_intra.keys(), axes):
        ci=np.array(pair_cn_intra[pk]); ce=np.array(pair_cn_inter[pk]); ct=ci+ce
        frames=start+np.arange(len(ci))*stride; step=max(1,len(frames)//20)
        ax.plot(frames,ci,'g-',lw=2.5,label='Intra',alpha=0.85)
        ax.plot(frames[::step],ci[::step],'go',ms=5)
        ax.plot(frames,ce,'b-',lw=2.5,label='Inter',alpha=0.85)
        ax.plot(frames[::step],ce[::step],'bs',ms=5)
        ax.plot(frames,ct,'r:',lw=2,label='Total',alpha=0.7)
        ax.axhline(ci.mean(),color='green',ls='--',lw=1.5,alpha=0.5)
        ax.axhline(ce.mean(),color='blue', ls='--',lw=1.5,alpha=0.5)
        ax.set_xlabel('Frame',fontsize=12,fontweight='bold')
        ax.set_ylabel('CN',fontsize=12,fontweight='bold')
        ax.set_title(f'Time Evolution: {pk[0]}-{pk[1]}',fontsize=13,fontweight='bold')
        ax.legend(fontsize=11,loc='best',framealpha=0.95); ax.grid(True,alpha=0.3)
        ax.text(0.98,0.97,f'Intra: {ci.mean():.3f}+/-{ci.std():.3f}\nInter: {ce.mean():.3f}+/-{ce.std():.3f}',
                transform=ax.transAxes,fontsize=10,va='top',ha='right',
                bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.8))
    plt.tight_layout()
    plt.savefig(f'{out_prefix}_time_series.png',dpi=300,bbox_inches='tight'); plt.close()
    print(f"  Time series  : {out_prefix}_time_series.png")


#  ANGLE DIAGNOSTIC  (V5 addition)

def collect_angles_for_pair(box, A_pos, B_pos, A_ch, B_ch, cn_cut,
                             donor_map, pos, sym_arr,
                             A_global_indices, B_global_indices):
    """
    Collect D-H···A angles (degrees) for N-H and O-H donors only
    (C-H excluded, covalent self-bond excluded), with no angle filter,
    so you can see the real H-bond geometry and choose --angle-cut from data.

    Returns
    -------
    angles_intra : np.ndarray  — angles for intra-chain pairs
    angles_inter : np.ndarray  — angles for inter-chain pairs
    n_no_donor   : int         — H atoms skipped (no donor or wrong donor type)
    """
    nl = freud.locality.AABBQuery(box, B_pos).query(A_pos, {"r_max": cn_cut}).toNeighborList()
    angles_intra, angles_inter = [], []
    n_no_donor = 0

    for i in range(len(A_pos)):
        for j in nl.point_indices[nl.query_point_indices == i]:
            g_j = int(B_global_indices[j]) if B_global_indices is not None else j
            if g_j not in donor_map:
                n_no_donor += 1
                continue
            D_global = donor_map[g_j]

            # Skip non-donors — derived automatically from PAIR_CUTOFFS
            if sym_arr[D_global] not in VALID_DONORS:
                continue

            # Skip the covalent bond itself: when acceptor A is the same atom as D
            g_i = int(A_global_indices[i]) if A_global_indices is not None else i
            if D_global == g_i:
                continue

            angle = calc_dha_angle(box, pos[D_global], B_pos[j], A_pos[i])
            if A_ch[i] == B_ch[j]:
                angles_intra.append(angle)
            else:
                angles_inter.append(angle)

    return np.array(angles_intra), np.array(angles_inter), n_no_donor


def plot_angle_diagnostic(all_angles, pairs, angle_cut, out_prefix):
    """
    For each pair, plot the D-H···A angle distribution split by intra/inter,
    with vertical lines at common cutoffs (110, 120, 130, 150 deg) so you can
    see exactly how many bonds each threshold keeps.

    Also prints a compact ASCII table to the terminal.
    """
    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(12, 5*n))
    if n == 1: axes = [axes]

    print("\n" + "="*72)
    print("  ANGLE DISTRIBUTION DIAGNOSTIC  (no angle filter applied)")
    print("="*72)
    print(f"  {'Pair':<8}  {'Type':<7}  {'Total':>7}  "
          f"{'>=110':>7}  {'>=120':>7}  {'>=130':>7}  {'>=150':>7}  {'>=160':>7}")
    print(f"  {'-'*8}  {'-'*7}  {'-'*7}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

    thresholds = [110, 120, 130, 150, 160]
    cutoff_colors = {110:'#d62728', 120:'#ff7f0e', 130:'#9467bd',
                     150:'#2ca02c', 160:'#1f77b4'}

    for (pk, ax) in zip(pairs, axes):
        a_intra = all_angles[pk]['intra']
        a_inter = all_angles[pk]['inter']
        pair_str = f"{pk[0]}-{pk[1]}"

        for arr, label, color in [
            (a_intra, 'intra', 'green'),
            (a_inter, 'inter', 'royalblue'),
        ]:
            if len(arr) == 0:
                print(f"  {pair_str:<8}  {label:<7}  {'0':>7}  " + "  ".join(['N/A']*len(thresholds)))
                continue

            # Histogram
            counts, bin_edges = np.histogram(arr, bins=90, range=(0, 180))
            bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
            ax.plot(bin_centers, counts, color=color, lw=2,
                    label=f'{label} (n={len(arr)})', alpha=0.85)
            ax.fill_between(bin_centers, counts, alpha=0.15, color=color)

            # Print table row
            counts_thr = [np.sum(arr >= t) for t in thresholds]
            pcts       = [f"{np.sum(arr>=t)/len(arr)*100:.0f}%" for t in thresholds]
            row_vals   = "  ".join(f"{c:>4}({p:>3})" for c,p in zip(counts_thr,pcts))
            print(f"  {pair_str:<8}  {label:<7}  {len(arr):>7}  {row_vals}")

        # Vertical lines for each threshold
        for t in thresholds:
            ax.axvline(t, color=cutoff_colors[t], ls='--', lw=1.5, alpha=0.8,
                       label=f'>={t}°')

        # Highlight the currently used cutoff
        ax.axvline(angle_cut, color='black', ls='-', lw=2.5, alpha=0.9,
                   label=f'current cut ({angle_cut:.0f}°)')

        ax.set_xlabel('D–H···A angle (degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Angle distribution: {pk[0]}···{pk[1]}  '
                     f'(all bonds within distance cutoff, no angle filter)',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(0, 180)
        ax.legend(fontsize=9, loc='upper left', framealpha=0.95, ncol=2)
        ax.grid(True, alpha=0.3)

    print("="*72)
    print("  Values shown as: count(percentage passing that threshold)")
    print(f"  Current --angle-cut = {angle_cut:.0f} deg")
    print("="*72 + "\n")

    plt.tight_layout()
    out = f"{out_prefix}_angle_distribution.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Angle dist.  : {out}")


def write_angle_report(path, all_angles, pairs, angle_cut, n_frames):
    """
    Write a detailed angle diagnostic text report: per-pair, per-type
    (intra/inter), percentiles, and per-threshold counts.
    """
    W = 72
    sep  = '=' * W
    thin = '·' * W
    thresholds = [100, 110, 120, 130, 140, 150, 155, 160, 165, 170]

    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'\n{"#"*W}\n')
        f.write(f'#{"D-H···A ANGLE DISTRIBUTION REPORT  (V5)":^{W-2}}#\n')
        f.write(f'#{"No angle filter applied — raw geometry of all pairs in cutoff":^{W-2}}#\n')
        f.write(f'{"#"*W}\n\n')
        f.write(f'  Frames analysed : {n_frames}\n')
        f.write(f'  Current --angle-cut : {angle_cut:.1f} deg\n\n')

        for pk in pairs:
            a_intra = all_angles[pk]['intra']
            a_inter = all_angles[pk]['inter']
            pair_str = f"{pk[0]}-{pk[1]}"

            f.write(f'  {sep}\n')
            f.write(f'  PAIR : {pair_str}\n')
            f.write(f'  {sep}\n\n')

            for arr, label in [(a_intra, 'Intra-chain'), (a_inter, 'Inter-chain')]:
                f.write(f'  [ {label}  —  {len(arr)} total bonds ]\n')
                f.write(f'  {thin}\n')
                if len(arr) == 0:
                    f.write(f'  No bonds found.\n\n')
                    continue
                f.write(f'  {"Mean":<18}: {arr.mean():.2f} deg\n')
                f.write(f'  {"Std dev":<18}: {arr.std():.2f} deg\n')
                f.write(f'  {"Min":<18}: {arr.min():.2f} deg\n')
                f.write(f'  {"Max":<18}: {arr.max():.2f} deg\n')
                for p in [10, 25, 50, 75, 90, 95]:
                    f.write(f'  {"Percentile "+str(p)+"%":<18}: {np.percentile(arr,p):.2f} deg\n')
                f.write(f'\n  Threshold survival (bonds passing >= cutoff):\n')
                f.write(f'  {"Cutoff":>8}  {"Count":>8}  {"Percent":>8}  {"Bar"}\n')
                f.write(f'  {"-"*8}  {"-"*8}  {"-"*8}  {"-"*30}\n')
                for t in thresholds:
                    n_pass = int(np.sum(arr >= t))
                    pct    = n_pass / len(arr) * 100
                    bar    = '█' * int(pct / 2)
                    marker = ' <-- current' if abs(t - angle_cut) < 0.5 else ''
                    f.write(f'  {t:>7}°  {n_pass:>8}  {pct:>7.1f}%  {bar}{marker}\n')
                f.write(f'\n')

        f.write(f'  {sep}\n')
        f.write(f'  HOW TO USE THIS REPORT\n')
        f.write(f'  {sep}\n\n')
        f.write(f'  Look at the "Intra-chain" survival table for your pair of\n')
        f.write(f'  interest. The cutoff where intra-chain count matches your\n')
        f.write(f'  OVITO/V3 reference CN is the angle threshold to use.\n\n')
        f.write(f'  Rule of thumb:\n')
        f.write(f'    >= 150 deg : strict (VMD default, linear bonds only)\n')
        f.write(f'    >= 130 deg : moderate (common in polymer H-bond literature)\n')
        f.write(f'    >= 110 deg : loose (includes bent/weak bonds)\n')
        f.write(f'    no filter  : use V3 or set a very low angle-cut\n\n')


#  MAIN
def main():
    parser = argparse.ArgumentParser(description="H-bond Analysis: Intra vs Inter-chain (V5 — distance + angle criterion)")
    parser.add_argument("xyz_file")
    parser.add_argument("start", type=int)
    parser.add_argument("--stop",       type=int,   default=None)
    parser.add_argument("--stride",     type=int,   default=1)
    parser.add_argument("--pairs",      nargs="+",  required=True)
    parser.add_argument("--rmax",       type=float, required=True)
    parser.add_argument("--nbins",      type=int,   default=300)
    parser.add_argument("--cn-cut",     type=float, required=True)
    parser.add_argument("--angle-cut",  type=float, default=150.0,
                        help="Minimum D-H...A angle in degrees (default: 150.0). "
                             "Use 150 for strong bonds (N-H, O-H donors, VMD default). "
                             "Use 110 to also include weak C-H...O bonds. "
                             "Only used when --mode=geometric.")
    parser.add_argument("--mode", default="geometric",
                        choices=["geometric", "ovito", "vmd"],
                        help="Analysis mode (default: geometric). "
                             "ovito    : distance only, no filters — matches OVITO/V3 exactly. "
                             "geometric: H···A distance + donor filter + angle (--angle-cut). "
                             "vmd      : D···A distance + donor filter + angle fixed at 150 deg "
                             "           — matches VMD H-bond plugin exactly. "
                             "           Use --cn-cut 3.5 with vmd mode.")
    parser.add_argument("--avg",        action="store_true")
    parser.add_argument("--angle-diag", action="store_true",
                        help="Collect and plot the full D-H...A angle distribution "
                             "(no angle filter applied) so you can choose --angle-cut "
                             "from data. Writes *_angle_distribution.png and "
                             "*_angle_report.txt. Use this on a short run first.")
    parser.add_argument('--debug-frame', type=int, default=None,
                        help="Print bond details for one specific frame")
    parser.add_argument("--out-prefix", default="analysis")
    parser.add_argument("--bond-cut",   type=float, default=1.8)
    parser.add_argument("--backbone-chains", nargs="+", type=int, default=None)
    parser.add_argument("--max-backbone-chains", type=int, default=3)
    # V4 cluster arguments
    parser.add_argument("--cluster",      action="store_true",
                        help="Toggle ON OVITO-style cluster analysis + 3D cluster plots")
    parser.add_argument("--cluster-pairs", nargs="+", default=None,
                        help="Pairs for clustering e.g. Be-F (default: same as --pairs)")
    parser.add_argument("--cluster-cut",   type=float, default=None,
                        help="Cutoff for cluster bonds (default: --cn-cut)")
    parser.add_argument("--cluster-atom",  type=str,   default=None,
                        help="Atom type to count per cluster e.g. Be")
    parser.add_argument("--prob-plot",     action="store_true",
                        help="Plot P(n) vs n cluster size probability")
    parser.add_argument("--aba-angle", action="store_true",
                        help="Compute and plot Be-F-Be angle distribution (requires --cluster). "
                             "Uses --cluster-atom as Be and the other atom in --cluster-pairs as F.")
    parser.add_argument("--corner-edge",   action="store_true",
                        help="Compute corner vs edge sharing between BeF4 tetrahedra (requires --cluster). "
                             "Writes summary, CSV, time series PNG, and interactive 3D HTML.")
    args = parser.parse_args()

    pairs = parse_pairs(args.pairs)

    if args.cluster:
        cluster_pairs = parse_pairs(args.cluster_pairs) if args.cluster_pairs else pairs
        cluster_cut   = args.cluster_cut if args.cluster_cut else args.cn_cut
        cluster_atom  = args.cluster_atom if args.cluster_atom else cluster_pairs[0][0]
    else:
        cluster_pairs = []; cluster_cut = args.cn_cut; cluster_atom = None

    pair_cn_total = {}; pair_cn_intra = {}; pair_cn_inter = {}; pair_cn_all = {}
    all_distances={};  rdf_acc={};       rdf_r={}
    cluster_size_lists  = defaultdict(list)
    cluster_n_per_frame = defaultdict(list)
    # V8 new accumulators
    aba_angles_all  = []          # list of per-frame angle lists
    corner_per_frame    = []
    edge_per_frame      = []
    face_per_frame      = []
    # V9: tagged angle accumulators (geometry-first, no hardcoded thresholds)
    edge_angles_tagged   = []     # angles through shared F of edge-sharing pairs
    corner_angles_tagged = []     # angles through shared F of corner-sharing pairs
    # V5 angle diagnostic accumulator
    all_angles = {pk: {'intra': [], 'inter': []} for pk in pairs}
    n_frames=0

    print("\n"+"="*72)
    print("  HYDROGEN BOND ANALYSIS  :  INTRA vs INTER-CHAIN  (V5)")
    print("="*72)
    print(f"  File    : {args.xyz_file}")
    print(f"  Pairs   : {', '.join(f'{a}-{b}' for a,b in pairs)}")
    if args.mode == "ovito":
        print(f"  Mode    : OVITO  (distance only — matches OVITO/V3, no angle/donor filter)")
        print(f"  H-bond  : H···A <= {args.cn_cut} A  |  Covalent: {args.bond_cut} A")
    elif args.mode == "vmd":
        _max_dh  = max(DONOR_COVALENT_CUTOFFS.values()) if DONOR_COVALENT_CUTOFFS else 1.01
        _h_a_eq  = round(args.cn_cut - _max_dh, 2)   # what H···A this D···A corresponds to
        print(f"  Mode    : VMD  (D···A distance + donor filter + angle = 150 deg fixed)")
        print(f"  H-bond  : D···A <= {args.cn_cut} A  |  Angle >= 150 deg  |  Covalent: {args.bond_cut} A")
        print(f"  Donors  : {VALID_DONORS}  (auto from PAIR_CUTOFFS)")
        print(f"  Largest D-H bond : {_max_dh} A")
        print(f"  Your D···A {args.cn_cut} A  =  H···A {_h_a_eq} A  (D···A minus D-H bond length)")
        print(f"  Tip: to match geometric --cn-cut X, use vmd --cn-cut = X + {_max_dh}")
    else:
        print(f"  Mode    : geometric  (H···A distance + donor filter + angle criterion)")
        print(f"  H-bond  : H···A <= {args.cn_cut} A  |  Angle >= {args.angle_cut} deg  |  Covalent: {args.bond_cut} A")
        print(f"  Donors  : {VALID_DONORS}  (auto from PAIR_CUTOFFS)")
    print(f"  Angle diag: {'ON' if args.angle_diag else 'off'}")
    print(f"  Backbone: {'specified='+str(args.backbone_chains) if args.backbone_chains else 'auto top-'+str(args.max_backbone_chains)}")
    if args.cluster:
        print(f"  Cluster      : ON  pair={cluster_pairs}  cut={cluster_cut}  atom={cluster_atom}")
    else:
        print(f"  Cluster      : OFF")
    print()

    for iframe,(lat,symbols,pos) in enumerate(read_xyz_frames(args.xyz_file)):
        if iframe < args.start: continue
        if args.stop and iframe >= args.stop: break
        if (iframe-args.start)%args.stride != 0: continue

        box       = freud.box.Box.from_matrix(lat)
        types, tn = build_type_map(symbols)
        chain_ids = identify_chains(box, pos, bond_cutoff=args.bond_cut, symbols=symbols)

        # build donor map for geometric and vmd modes
        if args.mode in ("geometric", "vmd"):
            donor_map = build_donor_map(box, pos, symbols)
            sym_arr   = np.array(symbols)
            # vmd mode uses D-A distance — build reverse map and donor pos arrays
            if args.mode == "vmd":
                global _reverse_donor_map
                _reverse_donor_map = {v: k for k, v in donor_map.items()}
                donor_pos_map = {}
                for elem in VALID_DONORS:
                    if elem in tn:
                        mask = types == tn.index(elem)
                        donor_pos_map[elem] = (
                            pos[mask],
                            chain_ids[mask],
                            np.where(mask)[0]
                        )
                n_donors = sum(len(v[2]) for v in donor_pos_map.values())
                n_acceptors = sum(1 for s in symbols if s == pairs[0][0])
                print(f"  N donors    (frame {iframe}): {n_donors}")
                print(f"  N acceptors (frame {iframe}): {n_acceptors}")
                print(f"  [VMD proof] CN denominator uses {n_acceptors} acceptors (frame {iframe})")
            else:
                donor_pos_map = None
        else:
            donor_map     = None
            sym_arr       = None
            donor_pos_map = None

        for a,b in pairs:
            if a not in tn or b not in tn: continue
            A=pos[types==tn.index(a)]; B=pos[types==tn.index(b)]
            cA=chain_ids[types==tn.index(a)]; cB=chain_ids[types==tn.index(b)]

            if args.mode == "ovito":
                # ── OVITO mode: pure distance neighbour count, identical to V3 ──
                ci,ce,di,de = coordination_number_intra_inter_detailed(
                    box, A, B, cA, cB, args.cn_cut,
                    same_type=(a==b))
            elif args.mode == "vmd":
                # ── VMD mode: D···A distance, donor filter, angle fixed at 150 ──
                A_global_indices = np.where(types==tn.index(a))[0]
                B_global_indices = np.where(types==tn.index(b))[0]
                ci,ce,ct_all,di,de = coordination_number_intra_inter_detailed_v5_DA(
                    box, A, B, cA, cB, args.cn_cut,
                    donor_map, pos, sym_arr, 150.0,
                    A_global_indices, B_global_indices,
                    donor_pos_map, same_type=(a==b))
                
            else:
                # ── geometric mode: H···A distance, donor filter, --angle-cut ──
                A_global_indices = np.where(types==tn.index(a))[0]
                B_global_indices = np.where(types==tn.index(b))[0]
                ci,ce,di,de = coordination_number_intra_inter_detailed_v5(
                    box, A, B, cA, cB, args.cn_cut,
                    donor_map, pos, sym_arr, args.angle_cut,
                    A_global_indices, B_global_indices,
                    same_type=(a==b))

                # ========== VISUAL DEBUG ==========
                if hasattr(args, 'debug_frame') and args.debug_frame is not None and iframe == args.debug_frame:
                    intra_dbg, inter_dbg, skip_dbg = get_hbonds_for_pair_v5(
                        box, A, B, cA, cB, args.cn_cut,
                        donor_map, pos, sym_arr, args.angle_cut,
                        A_global_indices, B_global_indices)
                    print(f'\n🔍 FRAME {iframe} DEBUG ({a}-{b})')
                    print(f'  Intra: {len(intra_dbg)}  Inter: {len(inter_dbg)}  Skipped: {skip_dbg}')
                    for label, bonds in [('INTRA', intra_dbg[:3]), ('INTER', inter_dbg[:3])]:
                        for k, (i, j) in enumerate(bonds):
                            g_j = int(B_global_indices[j])
                            g_i = int(A_global_indices[i])
                            D_global = donor_map.get(g_j, -1)
                            dist = np.linalg.norm(mic_vector(box, A[i], B[j]))
                            angle = calc_dha_angle(box, pos[D_global], B[j], A[i]) if D_global != -1 else -1
                            print(f'  {label} #{k+1}: chain{chain_ids[g_i]}-idx{g_i} ... chain{chain_ids[g_j]}-idx{g_j} r={dist:.3f}A angle={angle:.1f}deg')

            pk=(a,b)
            pair_cn_intra.setdefault(pk,[]).append(ci)
            pair_cn_inter.setdefault(pk,[]).append(ce)
            pair_cn_total.setdefault(pk,[]).append(ci+ce)
            all_distances.setdefault(pk,{'intra':[],'inter':[]})
            all_distances[pk]['intra'].append(di); all_distances[pk]['inter'].append(de)
            if args.avg:
                rdf=compute_rdf_group(box,A,B,args.rmax,args.nbins)
                if pk not in rdf_acc:
                    rdf_acc[pk]=np.zeros_like(rdf.rdf); rdf_r[pk]=rdf.bin_centers.copy()
                rdf_acc[pk]+=rdf.rdf

            # angle diagnostic — geometric mode only
            if args.angle_diag and args.mode == "geometric":
                A_global_indices = np.where(types==tn.index(a))[0]
                B_global_indices = np.where(types==tn.index(b))[0]
                ai, ae, _ = collect_angles_for_pair(
                    box, A, B, cA, cB, args.cn_cut,
                    donor_map, pos, sym_arr,
                    A_global_indices, B_global_indices)
                all_angles[pk]['intra'].append(ai)
                all_angles[pk]['inter'].append(ae)

        # Cluster analysis per frame (from V4)
        if args.cluster:
            for cpair in cluster_pairs:
                n_cl,sizes,_,la,lb = analyse_clusters_frame(
                    box,pos,symbols,cpair,cluster_cut,cluster_atom)
                cluster_size_lists[cpair].append(sizes)
                cluster_n_per_frame[cpair].append(n_cl)
                if iframe == args.start:
                    print("\n  [ Cluster 3D plot: "+cpair[0]+"-"+cpair[1]+" frame "+str(iframe)+" ]")
                    plot_3d_clusters(box,pos,symbols,cpair,cluster_cut,
                                     cluster_atom,args.out_prefix,iframe)

            # V9: Be-F-Be angle distribution (tagged by sharing type)
            if args.aba_angle or args.corner_edge:
                cpair = cluster_pairs[0]
                be_at = cluster_atom; f_at = cpair[1] if cpair[0]==cluster_atom else cpair[0]
                # plain angles for --aba-angle plot
                frame_angles = analyse_aba_angles_frame(
                    box, pos, symbols, be_at, f_at, cluster_cut)
                aba_angles_all.append(frame_angles)
                # tagged angles for data-derived region detection
                f_ea, f_ca, _ = analyse_aba_angles_tagged_frame(
                    box, pos, symbols, be_at, f_at, cluster_cut)
                edge_angles_tagged.extend(f_ea)
                corner_angles_tagged.extend(f_ca)

            # V8: Corner / Edge sharing
            if args.corner_edge:
                cpair = cluster_pairs[0]
                be_at = cluster_atom; f_at = cpair[1] if cpair[0]==cluster_atom else cpair[0]
                n_co, n_ed, n_fa, co_pairs, ed_pairs, _ = \
                    analyse_corner_edge_sharing_frame(
                        box, pos, symbols, be_at, f_at, cluster_cut)
                corner_per_frame.append(n_co)
                edge_per_frame.append(n_ed)
                face_per_frame.append(n_fa)
                if iframe == args.start:
                    print(f"\n  [ Corner/Edge 3D: {be_at}-{f_at} frame {iframe} ]")
                    print(f"    Corner-sharing pairs: {n_co} | Edge-sharing: {n_ed} | Face-sharing: {n_fa}")
                    plot_3d_corner_edge_interactive(
                        box, pos, symbols, be_at, f_at, cluster_cut, args.out_prefix, iframe)

        if iframe==args.start:
            # Chain info file
            ci_file=f"{args.out_prefix}_chain_info.txt"
            with open(ci_file,"w") as cf:
                cf.write("="*60+"\n CHAIN IDENTIFICATION\n"+"="*60+"\n\n")
                cf.write(f"  Mode            : {args.mode}\n")
                cf.write(f"  Covalent cutoff : {args.bond_cut} A\n")
                cf.write(f"  Total atoms     : {len(pos)}\n")
                cf.write(f"  Number of chains: {chain_ids.max()+1}\n")
                if args.mode == "geometric":
                    cf.write(f"  Donor map size  : {len(donor_map)} H atoms with known donors\n")
                cf.write("\n")
                csz=np.bincount(chain_ids)
                for cid in range(min(20,chain_ids.max()+1)):
                    cf.write(f"  Chain {cid:>4d}: {csz[cid]:>5d} atoms\n")
                if chain_ids.max()>=20:
                    cf.write(f"  ... and {chain_ids.max()+1-20} more chains\n")
            print(f"  Chain info   : {ci_file}\n")
            if args.mode == "geometric":
                print(f"  Donor map    : {len(donor_map)} H atoms mapped to donors\n")

            # plot functions use donor_map=None to auto-select V3 path (ovito mode)
            # vmd mode uses same plot path as geometric (donor_map populated)
            # but with angle fixed at 150
            dm    = donor_map                                    # None in ovito
            acut  = 150.0 if args.mode == "vmd" else args.angle_cut
            plot_cut = (args.cn_cut - _max_dh) if args.mode == "vmd" else args.cn_cut

            print("  [ Per-pair clean 3D plots ]")
            for a_p,b_p in pairs:
                pk=(a_p,b_p)
                cs=(pair_cn_intra[pk][0], pair_cn_inter[pk][0]) if pk in pair_cn_intra else None
                plot_3d_clean(box,pos,symbols,chain_ids,
                              [(a_p,b_p)], plot_cut,
                              f"{args.out_prefix}_{a_p}{b_p}",
                              iframe, frame_cn_stats=cs,
                              donor_map=dm, angle_cut=acut)

            print("\n  [ Per-pair backbone plots (separate HTML + PNG per pair) ]")
            for a_p,b_p in pairs:
                pk=(a_p,b_p)
                cs=(pair_cn_intra[pk][0], pair_cn_inter[pk][0]) if pk in pair_cn_intra else None
                plot_3d_backbone(box,pos,symbols,chain_ids,
                                 [(a_p,b_p)], plot_cut,
                                 f"{args.out_prefix}_{a_p}{b_p}",
                                 iframe, frame_cn_stats=cs,
                                 max_chains=args.max_backbone_chains,
                                 specific_chains=args.backbone_chains,
                                 bond_cut=args.bond_cut,
                                 donor_map=dm, angle_cut=acut)

            print("\n  [ Combined backbone plot (all pairs in ONE HTML + PNG) ]")
            cn_stats_dict={
                (a_p,b_p):(pair_cn_intra[(a_p,b_p)][0], pair_cn_inter[(a_p,b_p)][0])
                for a_p,b_p in pairs if (a_p,b_p) in pair_cn_intra
            }
            plot_3d_backbone_combined(
                box,pos,symbols,chain_ids,
                pairs, plot_cut,
                args.out_prefix, iframe,
                frame_cn_stats_dict=cn_stats_dict,
                max_chains=args.max_backbone_chains,
                specific_chains=args.backbone_chains,
                bond_cut=args.bond_cut,
                donor_map=dm, angle_cut=acut)
            print()

        n_frames+=1
        if iframe%10==0: print(f"  Frame {iframe}...")

    print(f"\n  Processed {n_frames} frames\n")

    for pk in all_distances:
        all_distances[pk]['intra']=np.concatenate(all_distances[pk]['intra']) if any(len(x)>0 for x in all_distances[pk]['intra']) else np.array([])
        all_distances[pk]['inter']=np.concatenate(all_distances[pk]['inter']) if any(len(x)>0 for x in all_distances[pk]['inter']) else np.array([])

    # CSV
    csv_f=f"{args.out_prefix}_coordination_per_frame.csv"
    with open(csv_f,"w") as f:
        f.write("frame,pair,CN_total,CN_intra,CN_inter\n")
        for pk in pair_cn_total:
            for i in range(len(pair_cn_total[pk])):
                f.write(f"{args.start+i*args.stride},{pk[0]}-{pk[1]},"
                        f"{pair_cn_total[pk][i]:.6f},"
                        f"{pair_cn_intra[pk][i]:.6f},"
                        f"{pair_cn_inter[pk][i]:.6f}\n")
    print(f"  CSV          : {csv_f}")

    #pair_cn_total[pairs[0]] = [ci + ce]
    total_cn_last_frame = ci + ce 

    # Summary
    sum_f=f"{args.out_prefix}_summary.txt"
    write_summary(sum_f, pairs, n_frames, args,
                  pair_cn_total, pair_cn_intra, pair_cn_inter, all_distances)
    print(f"  Summary      : {sum_f}")

    if args.avg:
        for pk,gs in rdf_acc.items():
            np.savetxt(f"{args.out_prefix}_rdf_{pk[0]}-{pk[1]}.dat",
                       np.column_stack((rdf_r[pk],gs/n_frames)),header="r g(r)")
        print("  RDF files saved")

    print("\n  Generating time series...")
    plot_time_series(pair_cn_intra, pair_cn_inter, args.start, args.stride, args.out_prefix)

    # V5 angle diagnostic output
    if args.angle_diag:
        print("\n  Generating angle distribution diagnostic...")
        # Concatenate angle arrays across all frames
        for pk in pairs:
            all_angles[pk]['intra'] = (
                np.concatenate(all_angles[pk]['intra'])
                if any(len(x) > 0 for x in all_angles[pk]['intra'])
                else np.array([]))
            all_angles[pk]['inter'] = (
                np.concatenate(all_angles[pk]['inter'])
                if any(len(x) > 0 for x in all_angles[pk]['inter'])
                else np.array([]))
        plot_angle_diagnostic(all_angles, pairs, args.angle_cut, args.out_prefix)
        ang_rep = f"{args.out_prefix}_angle_report.txt"
        write_angle_report(ang_rep, all_angles, pairs, args.angle_cut, n_frames)
        print(f"  Angle report : {ang_rep}")

    # Cluster post-loop outputs (from V4)
    if args.cluster:
        print("\n  [ Cluster Analysis Outputs ]")
        for cpair in cluster_pairs:
            pair_str = cpair[0]+"-"+cpair[1]
            write_cluster_summary(args.out_prefix+"_cluster_"+pair_str+"_summary.txt",
                                  cluster_size_lists[cpair],cluster_n_per_frame[cpair],
                                  cpair,cluster_cut,cluster_atom,n_frames)
            csv_cl = args.out_prefix+"_cluster_"+pair_str+"_per_frame.csv"
            with open(csv_cl,"w") as f:
                f.write("frame,n_clusters,sizes\n")
                for fi,(nc,sizes) in enumerate(zip(cluster_n_per_frame[cpair],
                                                   cluster_size_lists[cpair])):
                    f.write(str(args.start+fi*args.stride)+","+str(nc)+","+
                            "|".join(str(s) for s in sizes)+"\n")
            print("  Cluster CSV: "+csv_cl)
            if args.prob_plot:
                plot_cluster_probability(cluster_size_lists[cpair],
                                         cluster_atom,cpair,args.out_prefix)

        # V9: Be-F-Be angle distribution output
        if args.aba_angle and aba_angles_all:
            cpair = cluster_pairs[0]
            be_at = cluster_atom; f_at = cpair[1] if cpair[0]==cluster_atom else cpair[0]
            print(f"\n  [ Be-F-Be Angle Distribution ]")
            plot_aba_angle_distribution(
                aba_angles_all, be_at, f_at, args.out_prefix, n_frames, cluster_cut)

        # V9: Corner / Edge sharing output
        if args.corner_edge and corner_per_frame:
            cpair = cluster_pairs[0]
            be_at = cluster_atom; f_at = cpair[1] if cpair[0]==cluster_atom else cpair[0]
            print(f"\n  [ Corner/Edge Sharing Outputs ]")
            sum_ce = args.out_prefix + f"_{be_at}{f_at}{be_at}_corner_edge_summary.txt"
            write_corner_edge_summary(sum_ce, corner_per_frame, edge_per_frame, face_per_frame,
                                      be_at, f_at, cluster_cut, n_frames,
                                      edge_angles_all=edge_angles_tagged,
                                      corner_angles_all=corner_angles_tagged)
            csv_ce = args.out_prefix + f"_{be_at}{f_at}{be_at}_corner_edge_per_frame.csv"
            with open(csv_ce, 'w') as f:
                f.write("frame,corner_pairs,edge_pairs,face_pairs\n")
                for fi, (co, ed, fa) in enumerate(zip(corner_per_frame, edge_per_frame, face_per_frame)):
                    f.write(f"{args.start+fi*args.stride},{co},{ed},{fa}\n")
            print(f"  Corner/Edge CSV: {csv_ce}")
            plot_corner_edge_time_series(corner_per_frame, edge_per_frame, face_per_frame,
                                         be_at, f_at, args.out_prefix, args.start, args.stride)
            # angle distribution with data-derived regions (no hardcoded thresholds)
            plot_corner_edge_angle_distribution(
                aba_angles_all, be_at, f_at, args.out_prefix, n_frames, cluster_cut,
                edge_angles_all=edge_angles_tagged,
                corner_angles_all=corner_angles_tagged)

    print("\n"+"="*72)
    print("  V9 ANALYSIS COMPLETE  —  Output files:")
    print("="*72)
    for a_p,b_p in pairs:
        print(f"  {args.out_prefix}_{a_p}{b_p}_3D_bonds.png")
        print(f"  {args.out_prefix}_{a_p}{b_p}_3D_bonds_interactive.html")
        print(f"  {args.out_prefix}_{a_p}{b_p}_3D_backbone.png")
        print(f"  {args.out_prefix}_{a_p}{b_p}_3D_backbone_interactive.html")
    print(f"  {args.out_prefix}_combined_backbone.png")
    print(f"  {args.out_prefix}_combined_backbone_interactive.html")
    print(f"  {args.out_prefix}_time_series.png")
    print(f"  {args.out_prefix}_coordination_per_frame.csv")
    print(f"  {args.out_prefix}_summary.txt")
    if args.angle_diag:
        print(f"  {args.out_prefix}_angle_distribution.png")
        print(f"  {args.out_prefix}_angle_report.txt")
    if args.cluster:
        for cpair in cluster_pairs:
            ps = cpair[0]+"-"+cpair[1]
            print(f"  {args.out_prefix}_cluster_{ps}_3D.png")
            print(f"  {args.out_prefix}_cluster_{ps}_3D_interactive.html")
            print(f"  {args.out_prefix}_cluster_{ps}_summary.txt")
            print(f"  {args.out_prefix}_cluster_{ps}_per_frame.csv")
            if args.prob_plot:
                print(f"  {args.out_prefix}_cluster_prob_{cluster_atom}.png")
        if args.aba_angle:
            cpair = cluster_pairs[0]
            be_at = cluster_atom; f_at = cpair[1] if cpair[0]==cluster_atom else cpair[0]
            print(f"  {args.out_prefix}_{be_at}{f_at}{be_at}_angle_distribution.png")
        if args.corner_edge:
            cpair = cluster_pairs[0]
            be_at = cluster_atom; f_at = cpair[1] if cpair[0]==cluster_atom else cpair[0]
            print(f"  {args.out_prefix}_{be_at}{f_at}{be_at}_corner_edge_summary.txt")
            print(f"  {args.out_prefix}_{be_at}{f_at}{be_at}_corner_edge_per_frame.csv")
            print(f"  {args.out_prefix}_{be_at}{f_at}{be_at}_corner_edge_time_series.png")
            print(f"  {args.out_prefix}_{be_at}{f_at}{be_at}_corner_edge_angle_dist.png")
            print(f"  {args.out_prefix}_{be_at}{f_at}{be_at}_corner_edge_3D_interactive.html")
    print("="*72+"\n")


def compute_rdf_group(box, A, B, r_max, n_bins):
    rdf = freud.density.RDF(bins=n_bins, r_max=r_max)
    rdf.compute(system=(box,B), query_points=A, reset=True)
    return rdf

def parse_pairs(tokens):
    return [(t.split('-')[0].strip(), t.split('-')[1].strip()) for t in tokens if '-' in t]


if __name__ == "__main__":
    main()

# Example commands
#
# MODE 1: OVITO — distance only, (OVITO STYLE)
# python V7_cord_inter_intra.py file.xyz 0 --stop 100 --bond-cut 1.8 --pairs O-H N-H --rmax 12.0 --nbins 300 --cn-cut 2.24 --mode ovito --avg --max-backbone-chains 15
#
# MODE 2: GEOMETRIC — H···A distance + donor filter + your angle choice
# python V7_cord_inter_intra.py file.xyz 0 --stop 100 --bond-cut 1.8 --pairs O-H N-H --rmax 12.0 --nbins 300 --cn-cut 2.24 --mode geometric --angle-cut 130 --avg --max-backbone-chains 15
#
# MODE 3: VMD — D···A distance + donor filter + angle fixed 150 deg 
# python V7_cord_inter_intra.py file.xyz 0 --stop 100 --bond-cut 1.8 --pairs O-H N-H --rmax 12.0 --nbins 300 --cn-cut 3.5 --mode vmd --avg --max-backbone-chains 15
#
# ── CLUSTER ONLY (molten salt Be-F)
# NOTE: --cluster-cut is AUTOMATIC — it defaults to --cn-cut if not given.
#       You only need --cluster-cut if you want a DIFFERENT cutoff than --cn-cut (rare).
#       So for Be-F at 2.35 A, just set --cn-cut 2.35 and --cluster-cut is handled automatically.
#       --cluster-pairs defaults to --pairs automatically if not given.
#       --cluster-cut  defaults to --cn-cut  automatically if not given.
#       --cluster-atom is the only cluster argument you MUST provide (which atom to count per cluster).
#       So the minimal cluster command is just: add --cluster and --cluster-atom to your normal command.
# python V8_cord_inter_intra.py file.xyz 0 --stop 100 --bond-cut 1.8 --pairs Be-F --rmax 12.0 --nbins 300 --cn-cut 2.35 --mode ovito --avg --cluster --cluster-atom Be --prob-plot
#
# ── CLUSTER + Be-F-Be ANGLE DISTRIBUTION (V8 NEW) ──
# --aba-angle: plots Be-F-Be angle histogram, marks edge-sharing (~90 deg) and corner-sharing (>130 deg) peaks
# python V8_cord_inter_intra.py file.xyz 0 --stop 100 --bond-cut 1.8 --pairs Be-F --rmax 12.0 --nbins 300 --cn-cut 2.35 --mode ovito --avg --cluster --cluster-atom Be --prob-plot --aba-angle
#
# ── CLUSTER + CORNER/EDGE SHARING ANALYSIS (V8 NEW) ──
# --corner-edge: counts corner-sharing (1 shared F) vs edge-sharing (2 shared F) BeF4 tetrahedra pairs
#               writes summary txt, per-frame CSV, time series PNG, and interactive 3D HTML
# python V8_cord_inter_intra.py file.xyz 0 --stop 100 --bond-cut 1.8 --pairs Be-F --rmax 12.0 --nbins 300 --cn-cut 2.35 --mode ovito --avg --cluster --cluster-atom Be --prob-plot --aba-angle --corner-edge
#
# ── FULL MOLTEN SALT ANALYSIS (all features) ──
# python V8_cord_inter_intra.py FlibeCsF_510C.xyz 0 --stop 100 --bond-cut 1.8 --pairs Be-F --rmax 12.0 --nbins 300 --cn-cut 2.35 --mode ovito --avg --cluster --cluster-atom Be --prob-plot --aba-angle --corner-edge
# python V7_cord_inter_intra.py file.xyz 0 --stop 100 --bond-cut 1.8 --pairs Be-F # calculate CN for BOTH Li-F and Be-F --rmax 12.0 --nbins 300 --cn-cut 2.35 --mode ovito --avg --cluster --cluster-pairs Be-F # but build clusters ONLY from Be-F bonds --cluster-cut 2.35 --cluster-atom Be --prob-plot # but build clusters ONLY from Be-F bonds 
# python V7_cord_inter_intra.py FlibeCsF_510C_27x_train1_nvt_equi.xyz 0 --stop 100 --bond-cut 1.8 --pairs Be-F --rmax 12.0 --nbins 300 --cn-cut 2.35 --mode ovito --avg --cluster --cluster-pairs Be-F --cluster-cut 2.35 --cluster-atom Be --prob-plot 

#
# ── GEOMETRIC + CLUSTER (H-bond polymer + cluster) ──
# NOTE: same rule — --cluster-cut is automatic, no need to repeat --cn-cut value.
# python V7_cord_inter_intra.py file.xyz 0 --stop 100 --bond-cut 1.8 --pairs O-H N-H --rmax 12.0 --nbins 300 --cn-cut 2.24 --mode geometric --angle-cut 130 --avg --backbone-chains 2 30 45 --cluster --cluster-pairs O-H --cluster-atom O --prob-plot
# python V7_cord_inter_intra.py file.xyz 0 --stop 100 --bond-cut 1.8 --pairs O-H N-H --rmax 12.0 --nbins 300 --cn-cut 2.24 --mode geometric --angle-cut 130 --avg --backbone-chains 2 30 45 --cluster --cluster-pairs Be-F --cluster-cut 2.35 --cluster-atom Be --prob-plot
#
# ── ANGLE DIAGNOSTIC: run first on 10 frames to choose --angle-cut (geometric mode only) ──
# python V7_cord_inter_intra.py file.xyz 0 --stop 10 --bond-cut 1.8 --pairs O-H N-H --rmax 12.0 --nbins 300 --cn-cut 2.24 --mode geometric --angle-cut 150 --angle-diag
# Then read *_angle_report.txt and *_angle_distribution.png to pick your cutoff.
#
# ── Specific backbone chains ──
# python V7_cord_inter_intra.py file.xyz 0 --stop 100 --bond-cut 1.8 --pairs O-H N-H --rmax 12.0 --nbins 300 --cn-cut 2.24 --mode geometric --angle-cut 130 --avg --backbone-chains 2 30 45
