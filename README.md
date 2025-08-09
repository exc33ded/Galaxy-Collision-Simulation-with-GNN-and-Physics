# 🌌 Galaxy Collision Simulation (GNN + Physics)

This project simulates **galaxy collisions** using a combination of:
- **Graph Neural Networks (GNNs)** for learned particle motion predictions
- **Newtonian physics** for gravitational interactions
- **Streamlit** for real-time, interactive 3D visualization

The result is a dynamic simulation of two galaxies approaching, colliding, and interacting based on realistic astrophysics.

---

## ✨ Features
- Physics-accurate **N-body gravitational simulation**
- **GNN integration** for learned motion dynamics
- Real-time **3D visualization** using Plotly
- Interactive controls via **Streamlit** sidebar
- Adjustable:
  - Particle counts per galaxy
  - Initial separation
  - Relative velocity
  - Galaxy masses and sizes

---

## 📦 Installation

1. **Clone this repository**
```bash
git clone https://github.com/exc33ded/Galaxy-Collision-Simulation-with-GNN-and-Physics.git
cd galaxy-collision
```

2. **Create Virtual Environement**
```bash
python -m venv base
.\base\Scripts\Activate.ps1
```


3. **Install dependencies**
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
torch==2.3.1
torch-geometric==2.6.1
matplotlib==3.9.2
numpy==1.26.4
streamlit==1.37.1
scipy==1.14.0
networkx==3.3
tqdm==4.66.4
```

---

## 🚀 Usage

Run the simulation:
```bash
streamlit run app.py
```

In the browser interface:
1. Set simulation parameters in the sidebar  
2. Click **Start Simulation**  
3. Watch two galaxies collide in real time 🌠

---

## ⚙️ How It Works

1. **Initialization**  
   Two galaxies are generated with:
   - Positions sampled from a distribution (disk-like)
   - Initial velocities for head-on approach or offset pass

2. **Physics Updates**  
   At each timestep:
   - Compute gravitational forces between all particles  
   - Update velocities and positions (Newton’s laws)  
   - (Optional) Pass through **GalaxyGNN** for ML-driven motion correction

3. **Visualization**  
   A Plotly 3D scatter plot updates in real time to show particle motion.

---

## 📂 Project Structure
```
.
├── app.py             # Main Streamlit application
├── gnn_model.ipynb    # Python notebook for traninig the GNN
├── gnn_galaxy.py      # Trained GNN model
├── requirements.txt
└── README.md
```

---

## 🖼 Example Output
You’ll see:
- Two clouds of points (stars) starting apart
- Gradual gravitational attraction  
- Distortion and tidal tails during interaction  
- Either merger or separation based on velocity & mass

---

## ⚠️ Performance Notes
- **CPU:** Works but slower — keep total particles < 2000 for smooth updates  
- **GPU:** Much faster, supports higher particle counts and longer simulations

---

## 📜 License
MIT License — free to use, modify, and share.
