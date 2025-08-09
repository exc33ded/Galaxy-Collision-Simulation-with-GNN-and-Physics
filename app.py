import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from scipy.spatial import cKDTree
from matplotlib.collections import LineCollection

st.set_page_config(layout="wide")
st.title("Galaxy collision — Physics (left) vs GNN prediction (right)")

# ------------------------------
# Model definition (must match training)
# ------------------------------
class EdgeMP(MessagePassing):
    def __init__(self, in_channels, hidden_channels):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels*2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU()
        )
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
    def message(self, x_i, x_j):
        m = torch.cat([x_i, x_j], dim=-1)
        return self.mlp(m)
    def update(self, aggr_out):
        return aggr_out

class GalaxyGNN(nn.Module):
    def __init__(self, node_in=7, hidden=128, num_layers=3):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.layers = nn.ModuleList([EdgeMP(hidden, hidden) for _ in range(num_layers)])
        self.dec = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 6)
        )
    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        h = self.enc(x)
        for layer in self.layers:
            m = layer(h, edge_index)
            h = h + m
        out = self.dec(h)
        return out[:, :3], out[:, 3:6]

# ------------------------------
# Vectorized N-body (fast on CPU)
# ------------------------------
G = 1.0

def compute_accelerations_vectorized(positions, masses, softening=0.1):
    """
    positions: (N,3) numpy
    masses: (N,) numpy
    returns acc: (N,3)
    """
    # pairwise displacement vectors: r_j - r_i => (N, N, 3)
    r = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N,N,3)
    dist2 = np.sum(r*r, axis=2) + softening**2                       # (N,N)
    inv_dist3 = dist2 ** -1.5
    np.fill_diagonal(inv_dist3, 0.0)                                # zero self
    coef = G * masses[np.newaxis, :] * inv_dist3                    # (N,N)
    acc = (r * coef[:, :, None]).sum(axis=1)                        # (N,3)
    return acc.astype(np.float32)

def integrate_leapfrog_vectorized(pos0, vel0, masses, dt, steps, softening=0.1):
    pos = pos0.copy()
    vel = vel0.copy()
    pos_hist = [pos.copy()]
    vel_hist = [vel.copy()]
    a = compute_accelerations_vectorized(pos, masses, softening)
    vel_half = vel + 0.5 * a * dt
    for _ in range(steps):
        pos = pos + vel_half * dt
        a = compute_accelerations_vectorized(pos, masses, softening)
        vel_half = vel_half + a * dt
        vel = vel_half - 0.5 * a * dt
        pos_hist.append(pos.copy())
        vel_hist.append(vel.copy())
    return np.stack(pos_hist), np.stack(vel_hist)

# ------------------------------
# Graph builder (k-NN)
# ------------------------------
def build_knn_edge_index(positions, k=8):
    N = positions.shape[0]
    tree = cKDTree(positions)
    kq = min(k+1, N)
    _, idxs = tree.query(positions, k=kq)
    rows = np.repeat(np.arange(N), idxs.shape[1]-1)
    cols = idxs[:,1:].reshape(-1)
    edge_index = np.vstack([rows, cols])
    return torch.tensor(edge_index, dtype=torch.long)

# ------------------------------
# UI controls (more conservative defaults)
# ------------------------------
col1, col2 = st.columns([1,1])
with col1:
    st.header("Simulation parameters")
    n1 = st.number_input("Particles in galaxy A", min_value=5, max_value=200, value=40, step=5)
    n2 = st.number_input("Particles in galaxy B", min_value=5, max_value=200, value=40, step=5)
    sep = st.slider("Initial separation", 1.0, 20.0, 8.0, step=0.5)
    vrel = st.slider("Relative velocity (magnitude)", 0.0, 3.0, 0.8, step=0.1)
    mass1 = st.slider("Mass galaxy A", 0.1, 5.0, 1.0, step=0.1)
    mass2 = st.slider("Mass galaxy B", 0.1, 5.0, 1.0, step=0.1)
    scale1 = st.slider("Scale radius A", 0.2, 3.0, 1.0, step=0.1)
    scale2 = st.slider("Scale radius B", 0.2, 3.0, 1.0, step=0.1)
    dt = st.number_input("Integrator dt (physics)", min_value=0.002, max_value=0.1, value=0.03, step=0.002)
    steps = st.slider("Number of timesteps to simulate", 5, 200, 60, step=5)
    k_nn = st.number_input("k for k-NN graph (GNN edges)", min_value=1, max_value=32, value=6)
    trail_len = st.slider("Trail length (frames)", 1, 40, 12)
    edge_update_interval = st.number_input("Edge update interval (frames)", 1, 10, 2)
    speed = st.number_input("Animation pause (seconds)", min_value=0.0, max_value=0.5, value=0.06, step=0.01)

with col2:
    st.header("Model / run")
    model_file = st.text_input("Checkpoint file (local path)", value="gnn_galaxy.pt")
    load_button = st.button("Load model")
    start_button = st.button("Run Physics + GNN")
    st.write("Tips: keep particle counts <= 100 per galaxy for smooth CPU experience.")

# ------------------------------
# Model load (cached)
# ------------------------------
@st.cache_resource
def load_model(path):
    device = torch.device("cpu")
    m = GalaxyGNN(node_in=7, hidden=128, num_layers=3).to(device)
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    m.load_state_dict(sd)
    m.eval()
    return m

model = None
if load_button:
    try:
        model = load_model(model_file)
        st.success(f"Loaded model from {model_file}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
def plummer_sphere(n, scale_radius=1.0, total_mass=1.0, rng=None):
    """Return (pos, vel, masses) for n particles in a Plummer-like sphere."""
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    # approximate Plummer radial sampling
    r = scale_radius * (u ** (-2.0/3.0) - 1.0) ** (-0.5)
    cost = rng.uniform(-1.0, 1.0, n)
    sint = np.sqrt(1 - cost**2)
    phi = rng.uniform(0, 2*np.pi, n)
    x = r * sint * np.cos(phi)
    y = r * sint * np.sin(phi)
    z = r * cost
    pos = np.vstack([x, y, z]).T
    # small random velocities (bound-ish)
    vscale = np.sqrt(G * total_mass / (scale_radius + 1e-8))
    vel = rng.normal(0, 0.25 * vscale, size=(n, 3))
    masses = np.full(n, total_mass / n, dtype=np.float32)
    return pos.astype(np.float32), vel.astype(np.float32), masses

def build_two_galaxies(N1, N2, separation, vrel, mass1, mass2, scale1, scale2, rng_seed=42):
    rng = np.random.default_rng(rng_seed)

    # Galaxy 1: placed to the left
    pos1 = rng.normal(scale=scale1, size=(N1, 3)) - np.array([separation / 2, 0, 0])
    vel1 = rng.normal(scale=0.05, size=(N1, 3)) + np.array([vrel / 2, 0, 0])

    # Galaxy 2: placed to the right
    pos2 = rng.normal(scale=scale2, size=(N2, 3)) + np.array([separation / 2, 0, 0])
    vel2 = rng.normal(scale=0.05, size=(N2, 3)) - np.array([vrel / 2, 0, 0])

    # Mass arrays
    masses1 = np.full(N1, mass1)
    masses2 = np.full(N2, mass2)

    # Combine into single arrays
    pos0 = np.vstack([pos1, pos2])
    vel0 = np.vstack([vel1, vel2])
    masses = np.hstack([masses1, masses2])

    return pos0, vel0, masses


# ------------------------------
# Run simulation (physics + GNN rollout) and animate
# ------------------------------
if start_button:
    # auto-load model if not yet loaded
    if model is None:
        try:
            model = load_model(model_file)
            st.success(f"Loaded model from {model_file}")
        except Exception as e:
            st.error(f"Cannot proceed without model loaded: {e}")
            st.stop()

    # generate initial condition
    N1, N2 = int(n1), int(n2)
    pos0, vel0, masses = build_two_galaxies(N1, N2, sep, vrel, mass1, mass2, scale1, scale2, rng_seed=1234)
    N = pos0.shape[0]

    st.info("Computing physics reference trajectory...")
    phys_pos_traj, phys_vel_traj = integrate_leapfrog_vectorized(pos0, vel0, masses, dt=dt, steps=steps, softening=0.1)

    # prepare initial data for GNN rollout
    x_init = np.hstack([pos0, vel0, masses.reshape(-1,1)]).astype(np.float32)
    x_torch = torch.tensor(x_init, dtype=torch.float)
    edge_index = build_knn_edge_index(pos0, k=k_nn)
    data = Data(x=x_torch, edge_index=edge_index)

    data = data.to(torch.device("cpu"))

    # GNN autoregressive rollout (rebuild edges every edge_update_interval)
    st.info("Running GNN rollout (autoregressive; vectorized-friendly)...")
    pred_pos_traj = [pos0.copy()]
    pred_vel_traj = [vel0.copy()]
    curr_pos = pos0.copy()
    curr_vel = vel0.copy()

    for t in range(steps):
        if (t % edge_update_interval) == 0:
            edge_index = build_knn_edge_index(curr_pos, k=k_nn)
        # build input tensor for model
        x_step = np.hstack([curr_pos, curr_vel, masses.reshape(-1,1)]).astype(np.float32)
        data_step = Data(x=torch.tensor(x_step), edge_index=edge_index)
        data_step = data_step.to(torch.device("cpu"))
        with torch.no_grad():
            pos_pred_t, vel_pred_t = model(data_step)
        pos_pred_np = pos_pred_t.cpu().numpy()
        vel_pred_np = vel_pred_t.cpu().numpy()
        pred_pos_traj.append(pos_pred_np.copy())
        pred_vel_traj.append(vel_pred_np.copy())
        curr_pos = pos_pred_np
        curr_vel = vel_pred_np

    pred_pos_traj = np.stack(pred_pos_traj)
    pred_vel_traj = np.stack(pred_vel_traj)

    # ------------------------------
    # Prepare plotting (create scatter objects once & update)
    # ------------------------------
    left_col, right_col = st.columns(2)
    ph = left_col.empty()
    gh = right_col.empty()

    # initial plots (create figure and scatter artists)
    fig1, ax1 = plt.subplots(figsize=(5,5))
    fig2, ax2 = plt.subplots(figsize=(5,5))
    ax1.set_facecolor('black'); ax2.set_facecolor('black')
    ax1.tick_params(colors='white'); ax2.tick_params(colors='white')
    ax1.set_aspect('equal'); ax2.set_aspect('equal')

    colors = np.array(['C0'] * N)
    colors[N1:] = 'C1'

    sc1 = ax1.scatter(phys_pos_traj[0,:,0], phys_pos_traj[0,:,1], s=6, c=colors)
    sc2 = ax2.scatter(pred_pos_traj[0,:,0], pred_pos_traj[0,:,1], s=6, c=colors)

    # prepare trail line collections (one per galaxy for speed)
    def build_trail_lines(traj, t, trail_len, indices):
        # returns list of line segments [(x0,y0),(x1,y1)] for all nodes in indices over last interval
        segs = []
        for i in indices:
            start = max(0, t-trail_len)
            for tt in range(start, t):
                segs.append(np.vstack([traj[tt,i,:2], traj[tt+1,i,:2]]))
        return segs

    # empty initial line collections
    lc1 = LineCollection([], linewidths=0.6, colors='white', alpha=0.6)
    lc2 = LineCollection([], linewidths=0.6, colors='white', alpha=0.6)
    ax1.add_collection(lc1)
    ax2.add_collection(lc2)

    # animation loop
    for t in range(0, steps+1):
        p_phys = phys_pos_traj[t]
        p_pred = pred_pos_traj[t]

        # update scatter positions
        sc1.set_offsets(p_phys[:,:2])
        sc2.set_offsets(p_pred[:,:2])

        # update trails by building line segments (vectorized-ish)
        segs1 = build_trail_lines(phys_pos_traj, t, trail_len, range(N))
        segs2 = build_trail_lines(pred_pos_traj, t, trail_len, range(N))
        lc1.set_segments(segs1)
        lc2.set_segments(segs2)

        ax1.set_xlim(np.min([p_phys[:,0].min(), p_pred[:,0].min()]) - 1, np.max([p_phys[:,0].max(), p_pred[:,0].max()]) + 1)
        ax1.set_ylim(np.min([p_phys[:,1].min(), p_pred[:,1].min()]) - 1, np.max([p_phys[:,1].max(), p_pred[:,1].max()]) + 1)
        ax2.set_xlim(ax1.get_xlim()); ax2.set_ylim(ax1.get_ylim())

        ax1.set_title(f"Physics (t={t})", color='white')
        ax2.set_title(f"GNN (t={t})", color='white')

        ph.pyplot(fig1)
        gh.pyplot(fig2)

        # small pause (controls perceived frame rate)
        time.sleep(float(speed))

    st.success("Animation finished. (Optimized for CPU)")

