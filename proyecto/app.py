"""
app.py  —  Particle Navigation RL · Dashboard Streamlit
────────────────────────────────────────────────────────
Lanzar con:
    streamlit run app.py

Requisitos:
    pip install streamlit stable-baselines3 gymnasium imageio[pillow] matplotlib numpy pandas
"""

import os, sys, threading, io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from particle_env import ParticlePotentialEnv, default_potential

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Particle RL Navigator",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"]          { font-family: 'IBM Plex Sans', sans-serif; }
.stApp                              { background: #0a0c12; }
section[data-testid="stSidebar"]    { background: #0f1119 !important; border-right: 1px solid #1e2535; }

.rl-header {
    font-family: 'IBM Plex Mono', monospace; font-size: 1.9rem; font-weight: 600;
    color: #00e5ff; letter-spacing: -0.03em; margin-bottom: 0;
    text-shadow: 0 0 40px rgba(0,229,255,0.3);
}
.rl-sub {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #3d4f6b;
    letter-spacing: 0.15em; text-transform: uppercase; margin-top: 2px;
}
.badge-running { background:#0d2b1a; color:#00e5a0; border:1px solid #00e5a0; border-radius:4px; padding:2px 10px; font-size:.75rem; font-family:monospace; }
.badge-done    { background:#0d1f2b; color:#00aaff; border:1px solid #00aaff; border-radius:4px; padding:2px 10px; font-size:.75rem; font-family:monospace; }
.badge-idle    { background:#1a1a1a; color:#555;    border:1px solid #333;    border-radius:4px; padding:2px 10px; font-size:.75rem; font-family:monospace; }
.warn-box      { background:#1e1506; border:1px solid #f0b429; border-radius:6px; padding:10px 14px; font-family:monospace; font-size:.8rem; color:#f0b429; }

.stButton > button {
    background: transparent !important; border: 1px solid #00e5ff !important;
    color: #00e5ff !important; font-family: 'IBM Plex Mono', monospace !important;
    font-size: .8rem !important; border-radius: 4px !important; transition: all .2s !important;
}
.stButton > button:hover { background: rgba(0,229,255,.08) !important; box-shadow: 0 0 16px rgba(0,229,255,.2) !important; }

.stSlider label, .stSelectbox label, .stNumberInput label, .stTextArea label {
    font-family: 'IBM Plex Mono', monospace !important; font-size: .7rem !important;
    color: #3d4f6b !important; text-transform: uppercase; letter-spacing: .1em;
}
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #1e2535; gap: 0; }
.stTabs [data-baseweb="tab"]      { font-family:'IBM Plex Mono',monospace !important; font-size:.75rem !important; color:#3d4f6b !important; padding:8px 20px !important; }
.stTabs [aria-selected="true"]    { color:#00e5ff !important; border-bottom:2px solid #00e5ff !important; background:transparent !important; }
.stProgress > div > div           { background-color: #00e5ff !important; }
div[data-testid="stExpander"]     { border:1px solid #1e2535 !important; border-radius:6px !important; background:#13161e !important; }

/* code editor area */
textarea { background: #0f1119 !important; color: #a8ff78 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: .82rem !important; border: 1px solid #1e2535 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  PALETA
# ═══════════════════════════════════════════════════════════════════════════

BG, PANEL, GR = "#0a0c12", "#13161e", "#1e2535"
C1, C2, C3, C4 = "#00e5ff", "#ff6b6b", "#a8ff78", "#ffd166"
MUTED, WHITE   = "#3d4f6b", "#e8eaf0"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GR,
    "axes.labelcolor": WHITE, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": WHITE, "grid.color": GR, "grid.linewidth": .5, "font.family": "monospace",
})

# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

def ss(k, v):
    if k not in st.session_state: st.session_state[k] = v

ss("training_active", False)
ss("training_done",   False)
ss("log_lines",       [])
ss("potential_code",  
"""# Edita V(x, y) aquí.  Usa numpy (np) libremente.
# Valores altos = zonas costosas / barreras.

# --- Preset: dos gaussianas (por defecto) ---
V1 = 3.0 * np.exp(-((x - 0.3)**2 + (y - 0.5)**2) / 0.02)
V2 = 2.0 * np.exp(-((x - 0.7)**2 + (y - 0.5)**2) / 0.02)
return float(V1 + V2)
""")
ss("potential_error", None)
ss("start", (0.1, 0.1))
ss("goal",  (0.9, 0.9))


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE potential_fn DESDE CÓDIGO
# ═══════════════════════════════════════════════════════════════════════════

def build_potential_fn(code: str):
    """
    Compila el código del editor como cuerpo de una función V(x, y).
    Devuelve (fn, error_str).
    """
    full = "import numpy as np\ndef _V(x, y):\n"
    for line in code.splitlines():
        full += "    " + line + "\n"
    try:
        ns = {}
        exec(compile(full, "<editor>", "exec"), ns)
        fn = ns["_V"]
        # test rápido
        result = fn(0.5, 0.5)
        float(result)
        return fn, None
    except Exception as e:
        return default_potential, str(e)


def get_potential_fn():
    fn, err = build_potential_fn(st.session_state["potential_code"])
    st.session_state["potential_error"] = err
    return fn


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def smooth(v, w=0.85):
    s, last = [], float(v[0])
    for x in v:
        last = last * w + (1 - w) * x
        s.append(last)
    return np.array(s)


def load_monitor(log_dir):
    frames = []
    for p in Path(log_dir).rglob("*.monitor.csv"):
        try:
            frames.append(pd.read_csv(p, skiprows=1))
        except Exception:
            pass
    if not frames: return None
    df = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    df["episode"] = df.index + 1
    return df


def load_eval(log_dir):
    p = Path(log_dir) / "evaluations.npz"
    if not p.exists(): return None
    d = np.load(p)
    return pd.DataFrame({
        "timestep":    d["timesteps"],
        "mean_reward": d["results"].mean(axis=1),
        "std_reward":  d["results"].std(axis=1),
    })


def load_tb(log_dir):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {}
    scalars = {}
    for ef in Path(log_dir).rglob("events.out.tfevents.*"):
        ea = EventAccumulator(str(ef)); ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            evts = ea.Scalars(tag)
            scalars[tag] = pd.DataFrame({"step":  [e.step  for e in evts],
                                          "value": [e.value for e in evts]})
    return scalars


def model_exists(path):
    return os.path.isfile(path) and os.path.getsize(path) > 1000


@st.cache_data(show_spinner=False)
def build_V_grid(code: str, n=220):
    """Cache del grid de potencial. Se invalida al cambiar el código."""
    fn, _ = build_potential_fn(code)
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    XX, YY = np.meshgrid(xs, ys)
    return xs, ys, np.vectorize(fn)(XX, YY)


def rollout(model, potential_fn, start, goal, max_steps, deterministic=True):
    env = ParticlePotentialEnv(potential_fn=potential_fn, start=start,
                                goal=goal, max_steps=max_steps)
    obs, _ = env.reset()
    traj, total_r = [env.pos.copy()], 0.0
    done = False
    while not done:
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = env.action_space.sample()
        obs, r, term, trunc, _ = env.step(action)
        traj.append(env.pos.copy())
        total_r += r
        done = term or trunc
    return np.array(traj), total_r


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURAS
# ═══════════════════════════════════════════════════════════════════════════

def fig_potential_preview(code, start, goal):
    xs, ys, V = build_V_grid(code)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=BG)
    ax.contourf(xs, ys, V, levels=40, cmap="inferno", alpha=0.9)
    ax.contour (xs, ys, V, levels=10, colors="white",  lw=0.4, alpha=0.3)
    ax.plot(*start, "o", color=C3, ms=10, zorder=5, label="inicio")
    ax.plot(*goal,  "*", color=C4, ms=14, zorder=5, label="objetivo")
    # línea directa
    ax.plot([start[0], goal[0]], [start[1], goal[1]],
            "--", color="white", lw=0.8, alpha=0.3, label="camino directo")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
    ax.legend(fontsize=8, framealpha=0.15, loc="upper left")
    ax.set_title("Vista previa del entorno", color=WHITE, fontsize=10, pad=8)
    fig.tight_layout()
    return fig


def fig_metrics(log_dir, smooth_w):
    monitor = load_monitor(log_dir)
    eval_df = load_eval(log_dir)
    tb      = load_tb(log_dir)

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35,
                            left=0.07, right=0.97, top=0.92, bottom=0.08)
    fig.suptitle("Training Metrics", fontsize=13, color=WHITE, y=0.97)

    def _panel(spec, df, col, xcol, color, title, xlabel, ylabel):
        ax = fig.add_subplot(spec)
        if df is not None and col in df.columns:
            v = df[col].values
            ax.plot(df[xcol], v, color=color, alpha=.2, lw=.6)
            ax.plot(df[xcol], smooth(v, smooth_w), color=color, lw=1.8)
        else:
            ax.text(0.5, 0.5, "Sin datos\n(entrena primero)", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes, fontsize=9)
        ax.set_title(title, color=WHITE, fontsize=9)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True)

    _panel(gs[0,0], monitor, "r", "episode", C1, "Recompensa / episodio",  "Episodio", "Reward")
    _panel(gs[0,1], monitor, "l", "episode", C2, "Longitud de episodio",   "Episodio", "Pasos")

    ax3 = fig.add_subplot(gs[0,2])
    if eval_df is not None:
        ts, mu, sd = eval_df["timestep"].values, eval_df["mean_reward"].values, eval_df["std_reward"].values
        ax3.fill_between(ts, mu-sd, mu+sd, color=C3, alpha=.15)
        ax3.plot(ts, mu, color=C3, lw=2, marker="o", ms=4)
    else:
        ax3.text(0.5, 0.5, "Sin evaluations.npz", ha="center", va="center",
                 color=MUTED, transform=ax3.transAxes, fontsize=9)
    ax3.set_title("Eval periódica", color=WHITE, fontsize=9)
    ax3.set_xlabel("Timesteps"); ax3.set_ylabel("Mean reward"); ax3.grid(True)

    for spec, tag, color, title in [
        (gs[1,0], "train/actor_loss",  C1, "Actor loss"),
        (gs[1,1], "train/critic_loss", C2, "Critic loss"),
        (gs[1,2], "train/ent_coef",    C4, "Entropy coef"),
    ]:
        ax = fig.add_subplot(spec)
        df = tb.get(tag)
        if df is not None and len(df):
            v = df["value"].values
            ax.plot(df["step"], v, color=color, alpha=.2, lw=.6)
            ax.plot(df["step"], smooth(v, smooth_w), color=color, lw=1.8)
        else:
            ax.text(0.5, 0.5, f"Sin datos TB\n({tag})", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes, fontsize=9)
        ax.set_title(title, color=WHITE, fontsize=9); ax.set_xlabel("Timesteps"); ax.grid(True)
    return fig


def fig_trajectories(trajs, rewards, start, goal, code, compare_trajs=None):
    xs, ys, V = build_V_grid(code)
    ncols = 2 if compare_trajs else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5*ncols, 5.5), facecolor=BG, squeeze=False)
    for ax in axes[0]:
        ax.contourf(xs, ys, V, levels=40, cmap="inferno", alpha=0.85)
        ax.contour (xs, ys, V, levels=10, colors="white", lw=0.3, alpha=0.25)
        ax.plot(*start, "o", color=C3, ms=9, zorder=5)
        ax.plot(*goal,  "*", color=C4, ms=13, zorder=5)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
    for i, (traj, r) in enumerate(zip(trajs, rewards)):
        alpha = 0.3 + 0.4*(i/max(len(trajs)-1, 1))
        axes[0][0].plot(traj[:,0], traj[:,1], color=C1, alpha=alpha, lw=1.5)
    axes[0][0].set_title(f"Política SAC  |  R̄={np.mean(rewards):.1f}", color=WHITE, fontsize=9)
    if compare_trajs:
        for traj in compare_trajs:
            axes[0][1].plot(traj[:,0], traj[:,1], color=C2, alpha=0.4, lw=1.2)
        axes[0][1].set_title("Política aleatoria", color=C2, fontsize=9)
    fig.tight_layout(pad=1.2)
    return fig


def make_gif(trajs, rewards, start, goal, code, fps=20) -> bytes:
    xs, ys, V = build_V_grid(code)
    PAUSE = int(fps * 0.6)
    schedule = []
    for ep, traj in enumerate(trajs):
        for t in range(len(traj)):       schedule.append((ep, t))
        for _ in range(PAUSE):           schedule.append((ep, len(traj)-1))

    fig, ax = plt.subplots(figsize=(5, 5), facecolor=BG)
    ax.contourf(xs, ys, V, levels=40, cmap="inferno", alpha=0.88)
    ax.contour (xs, ys, V, levels=10, colors="white", lw=0.3, alpha=0.2)
    ax.plot(*start, "o", color=C3, ms=9, zorder=5)
    ax.plot(*goal,  "*", color=C4, ms=13, zorder=5)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal"); ax.set_facecolor(PANEL)

    ghosts   = [ax.plot([],[],color=C1, alpha=0.12+0.08*i, lw=0.9)[0] for i in range(len(trajs))]
    live_l,  = ax.plot([],[], color=C1, lw=2.2, zorder=6)
    dot,     = ax.plot([],[], "o", color=C1, ms=7, zorder=7)
    title_t  = ax.set_title("", color=WHITE, fontsize=9, pad=6)

    def update(idx):
        ep, t = schedule[idx]
        seg   = trajs[ep][:t+1]
        for i, g in enumerate(ghosts):
            g.set_data(trajs[i][:,0], trajs[i][:,1]) if i < ep else g.set_data([],[])
        live_l.set_data(seg[:,0], seg[:,1])
        dot.set_data([seg[-1,0]], [seg[-1,1]])
        title_t.set_text(f"Ep {ep+1}/{len(trajs)}  R={rewards[ep]:.1f}  paso {t}")
        return ghosts + [live_l, dot, title_t]

    anim = FuncAnimation(fig, update, frames=len(schedule), interval=1000//fps, blit=True)
    buf  = io.BytesIO()
    anim.save(buf, writer=PillowWriter(fps=fps), dpi=110, savefig_kwargs={"facecolor": BG})
    plt.close(fig); buf.seek(0)
    return buf.read()


def fig_heatmap(trajs_agent, start, goal, code, trajs_rand=None):
    all_a   = np.vstack(trajs_agent)
    xs, ys, V = build_V_grid(code, 150)
    ncols = 2 if trajs_rand else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5*ncols, 5), facecolor=BG, squeeze=False)

    def draw(ax, pos, color, title):
        h, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=60, range=[[0,1],[0,1]])
        ax.imshow(h.T, origin="lower", extent=[0,1,0,1],
                  cmap="plasma" if color==C1 else "Reds",
                  aspect="equal", interpolation="gaussian")
        ax.plot(*start, "o", color=C3, ms=9, zorder=5)
        ax.plot(*goal,  "*", color=C4, ms=13, zorder=5)
        ax.set_title(title, color=color, fontsize=9); ax.set_facecolor(PANEL)

    draw(axes[0][0], all_a, C1, f"Densidad SAC ({len(trajs_agent)} eps)")
    if trajs_rand:
        draw(axes[0][1], np.vstack(trajs_rand), C2, f"Densidad aleatoria ({len(trajs_rand)} eps)")
    fig.tight_layout(pad=1.2)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO EN HILO
# ═══════════════════════════════════════════════════════════════════════════

def run_training(timesteps, lr, buf, batch, n_envs, log_dir, model_path,
                 potential_code, start, goal, max_steps):
    try:
        from particle_env import train as _train
        fn, err = build_potential_fn(potential_code)
        if err:
            st.session_state["log_lines"].append(f"❌  Error en potencial: {err}")
            st.session_state["training_active"] = False
            return

        st.session_state["log_lines"].append("🔧  Compilando entorno…")
        _train(
            total_timesteps=timesteps,
            log_dir=log_dir,
            model_path=model_path,
            potential_fn=fn,
            start=start,
            goal=goal,
            max_steps=max_steps,
            lr=lr,
            buffer_size=buf,
            batch_size=batch,
            n_envs=n_envs,
        )
        st.session_state["log_lines"].append(f"✅  Guardado en {model_path}.zip")
        st.session_state["training_done"]   = True
        st.session_state["training_active"] = False
    except Exception as e:
        st.session_state["log_lines"].append(f"❌  {e}")
        st.session_state["training_active"] = False


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR  —  Rutas, hiperparámetros y visualización
# ═══════════════════════════════════════════════════════════════════════════

PRESETS = {
    "Dos gaussianas (defecto)": """\
V1 = 3.0 * np.exp(-((x - 0.3)**2 + (y - 0.5)**2) / 0.02)
V2 = 2.0 * np.exp(-((x - 0.7)**2 + (y - 0.5)**2) / 0.02)
return float(V1 + V2)
""",
    "Barrera vertical": """\
return float(5.0 * np.exp(-((x - 0.5)**2) / 0.005))
""",
    "Canal horizontal": """\
# Barrera arriba y abajo, canal libre en y=0.5
V = 4.0 * np.exp(-(y - 0.2)**2 / 0.01) + 4.0 * np.exp(-(y - 0.8)**2 / 0.01)
return float(V)
""",
    "Laberinto 2×2": """\
walls = [
    (0.33, 0.2, 0.008, 0.015),   # (cx, cy, sx, sy)
    (0.33, 0.8, 0.008, 0.015),
    (0.66, 0.5, 0.008, 0.015),
]
V = sum(6.0 * np.exp(-((x-cx)**2/sx + (y-cy)**2/sy)) for cx,cy,sx,sy in walls)
return float(V)
""",
    "Pozo central (atractor)": """\
# Zona de coste negativo en el centro: el agente tiende a pasar por ahí
V = -2.0 * np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.03)
return float(V)
""",
    "Cero (espacio libre)": """\
return 0.0
""",
}

with st.sidebar:
    st.markdown('<p class="rl-header">⚛ Particle RL</p>', unsafe_allow_html=True)
    st.markdown('<p class="rl-sub">Navigator · SAC · Gymnasium</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown("#### 🗂 Rutas")
    log_dir    = st.text_input("Log dir",    value="./logs/",               label_visibility="collapsed")
    model_path = st.text_input("Model path", value="./models/particle_sac", label_visibility="collapsed")
    st.caption(f"`{log_dir}`  ·  `{model_path}.zip`")

    st.divider()
    st.markdown("#### ⚙️ Hiperparámetros SAC")
    timesteps = st.select_slider("Timesteps",
        options=[50_000,100_000,200_000,300_000,500_000,1_000_000], value=200_000)
    lr    = st.select_slider("Learning rate",
        options=[1e-4,3e-4,1e-3], value=3e-4, format_func=lambda x: f"{x:.0e}")
    buf   = st.select_slider("Buffer size",   options=[50_000,100_000,200_000], value=100_000)
    batch = st.select_slider("Batch size",    options=[64,128,256,512],         value=256)
    n_envs= st.select_slider("Envs paralelos",options=[1,2,4,8],               value=4)

    st.divider()
    st.markdown("#### 🎬 Visualización")
    smooth_w    = st.slider("Suavizado EMA",      0.0, 0.99, 0.85, 0.01)
    n_ep_anim   = st.slider("Episodios animación", 1, 8, 3)
    fps_gif     = st.slider("FPS gif",            10, 40, 20)
    show_random = st.checkbox("Comparar con política aleatoria", value=False)

    st.divider()
    if st.session_state["training_active"]:
        st.markdown('<span class="badge-running">● ENTRENANDO</span>', unsafe_allow_html=True)
    elif st.session_state["training_done"] or model_exists(model_path + ".zip"):
        st.markdown('<span class="badge-done">● MODELO LISTO</span>',  unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-idle">○ SIN MODELO</span>',    unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CABECERA
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<h1 class="rl-header">Particle Navigation · RL Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="rl-sub">Soft Actor–Critic · Continuous Control · Custom Gymnasium Env</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════

tab_env, tab_train, tab_metrics, tab_policy = st.tabs([
    "🗺 Entorno", "🚀 Entrenamiento", "📈 Métricas", "🎬 Política"
])


# ───────────────────────────────────────────────────────────────────────────
# TAB 1  —  ENTORNO  (configura potencial, start, goal, max_steps)
# ───────────────────────────────────────────────────────────────────────────

with tab_env:
    col_edit, col_preview = st.columns([1, 1], gap="large")

    with col_edit:
        st.markdown("#### 🧮 Función de potencial")

        # Preset selector
        preset_name = st.selectbox("Cargar preset", list(PRESETS.keys()), index=0)
        if st.button("↓ Cargar en editor", use_container_width=True):
            st.session_state["potential_code"] = PRESETS[preset_name]
            build_V_grid.clear()   # invalida cache
            st.rerun()

        # Editor de código
        new_code = st.text_area(
            "Cuerpo de V(x, y)  —  numpy disponible como `np`",
            value=st.session_state["potential_code"],
            height=200,
            key="code_editor",
        )
        if new_code != st.session_state["potential_code"]:
            st.session_state["potential_code"] = new_code
            build_V_grid.clear()

        # Feedback de errores
        _, err = build_potential_fn(st.session_state["potential_code"])
        if err:
            st.markdown(f'<div class="warn-box">⚠ Error en V(x,y):<br><code>{err}</code></div>',
                        unsafe_allow_html=True)
        else:
            st.success("V(x, y) válida ✓", icon="✅")

        st.divider()
        st.markdown("#### 📍 Puntos de inicio y objetivo")

        c_s, c_g = st.columns(2)
        with c_s:
            sx = st.slider("Inicio  x", 0.02, 0.98, float(st.session_state["start"][0]), 0.01, key="sx")
            sy = st.slider("Inicio  y", 0.02, 0.98, float(st.session_state["start"][1]), 0.01, key="sy")
        with c_g:
            gx = st.slider("Objetivo  x", 0.02, 0.98, float(st.session_state["goal"][0]), 0.01, key="gx")
            gy = st.slider("Objetivo  y", 0.02, 0.98, float(st.session_state["goal"][1]), 0.01, key="gy")

        st.session_state["start"] = (sx, sy)
        st.session_state["goal"]  = (gx, gy)

        st.divider()
        st.markdown("#### ⏱ Duración del episodio")
        max_steps = st.slider("Max steps por episodio", 100, 2000, 500, 50)

    with col_preview:
        st.markdown("#### Vista previa")
        st.pyplot(
            fig_potential_preview(
                st.session_state["potential_code"],
                st.session_state["start"],
                st.session_state["goal"],
            ),
            use_container_width=True,
        )
        st.caption("🟢 inicio   ⭐ objetivo   ╌ camino directo")

        # Valor puntual interactivo
        st.divider()
        st.markdown("#### Inspeccionar V(x, y)")
        ic1, ic2 = st.columns(2)
        with ic1: ix = st.number_input("x", 0.0, 1.0, 0.5, 0.01)
        with ic2: iy = st.number_input("y", 0.0, 1.0, 0.5, 0.01)
        fn_now, _ = build_potential_fn(st.session_state["potential_code"])
        st.metric("V(x, y)", f"{fn_now(ix, iy):.4f}")


# ───────────────────────────────────────────────────────────────────────────
# TAB 2  —  ENTRENAMIENTO
# ───────────────────────────────────────────────────────────────────────────

with tab_train:
    c_btn, c_log = st.columns([1, 2], gap="large")

    with c_btn:
        st.markdown("#### Control")

        # Resumen de la config actual del entorno
        _, pot_err = build_potential_fn(st.session_state["potential_code"])
        if pot_err:
            st.markdown('<div class="warn-box">⚠ El potencial tiene errores — corrígelo en la pestaña Entorno antes de entrenar.</div>',
                        unsafe_allow_html=True)
        else:
            with st.expander("Config activa del entorno", expanded=False):
                st.json({
                    "start":      list(st.session_state["start"]),
                    "goal":       list(st.session_state["goal"]),
                    "max_steps":  max_steps,
                    "timesteps":  timesteps,
                    "lr":         lr,
                    "buffer":     buf,
                    "batch":      batch,
                    "n_envs":     n_envs,
                    "log_dir":    log_dir,
                    "model_path": model_path + ".zip",
                })

        if model_exists(model_path + ".zip"):
            st.success(f"Modelo en `{model_path}.zip`", icon="✅")

        can_train = not st.session_state["training_active"] and pot_err is None
        btn_train = st.button(
            "▶ Iniciar entrenamiento" if not st.session_state["training_active"] else "⏳ Entrenando…",
            disabled=not can_train,
            use_container_width=True,
        )

        if btn_train:
            st.session_state["training_active"] = True
            st.session_state["training_done"]   = False
            st.session_state["log_lines"]       = ["🔧  Iniciando…"]
            # Captura valores actuales para pasarlos al hilo
            _code      = st.session_state["potential_code"]
            _start     = st.session_state["start"]
            _goal      = st.session_state["goal"]
            _msteps    = max_steps
            threading.Thread(
                target=run_training,
                args=(timesteps, lr, buf, batch, n_envs, log_dir, model_path,
                      _code, _start, _goal, _msteps),
                daemon=True,
            ).start()
            st.rerun()

        if st.session_state["training_active"]:
            st.info("Entrenando en segundo plano.", icon="ℹ️")
            if st.button("🔄 Refrescar", use_container_width=True):
                st.rerun()

    with c_log:
        st.markdown("#### Log")
        st.code("\n".join(st.session_state["log_lines"]) or "— Sin actividad —", language="bash")

        st.divider()
        st.markdown("#### Estructura de archivos")
        st.code(f"""
{log_dir}
├── env_0.monitor.csv
├── evaluations.npz
└── SAC_1/events.out.tfevents.*

{os.path.dirname(model_path)}/
├── particle_sac.zip
└── best/best_model.zip
""", language="bash")


# ───────────────────────────────────────────────────────────────────────────
# TAB 3  —  MÉTRICAS
# ───────────────────────────────────────────────────────────────────────────

with tab_metrics:
    monitor = load_monitor(log_dir)
    eval_df = load_eval(log_dir)

    if monitor is None and eval_df is None:
        st.warning(f"No hay logs en `{log_dir}`. Entrena primero.", icon="⚠️")
    else:
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Episodios", f"{len(monitor):,}" if monitor is not None else "—")
        with k2: st.metric("Reward medio", f"{monitor['r'].mean():.1f}" if monitor is not None else "—")
        with k3: st.metric("Mejor eval", f"{eval_df['mean_reward'].max():.1f}" if eval_df is not None else "—")
        with k4: st.metric("Timesteps eval", f"{eval_df['timestep'].max():,}" if eval_df is not None else "—")

        st.divider()
        st.pyplot(fig_metrics(log_dir, smooth_w), use_container_width=True)

        st.divider()
        dl1, dl2 = st.columns(2)
        with dl1:
            buf_png = io.BytesIO()
            fig_metrics(log_dir, smooth_w).savefig(buf_png, format="png",
                dpi=150, bbox_inches="tight", facecolor=BG)
            st.download_button("⬇ PNG métricas", data=buf_png.getvalue(),
                               file_name="training_metrics.png", mime="image/png",
                               use_container_width=True)
        with dl2:
            if monitor is not None:
                st.download_button("⬇ CSV episodios",
                                   data=monitor.to_csv(index=False).encode(),
                                   file_name="episodes.csv", mime="text/csv",
                                   use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────
# TAB 4  —  POLÍTICA
# ───────────────────────────────────────────────────────────────────────────

with tab_policy:
    if not model_exists(model_path + ".zip"):
        st.warning(f"No hay modelo en `{model_path}.zip`. Entrena primero.", icon="⚠️")
    else:
        from stable_baselines3 import SAC

        @st.cache_resource(show_spinner="Cargando modelo…")
        def load_model(path):
            return SAC.load(path)

        model = load_model(model_path + ".zip")

        # Parámetros del entorno actuales para rollouts
        _fn    = get_potential_fn()
        _start = st.session_state["start"]
        _goal  = st.session_state["goal"]
        _code  = st.session_state["potential_code"]

        subtab_traj, subtab_anim, subtab_heat = st.tabs([
            "📍 Trayectorias", "🎞 Animación GIF", "🔥 Heatmap"
        ])

        # ── Trayectorias estáticas ──────────────────────────────────────
        with subtab_traj:
            if st.button("▶ Generar trayectorias", use_container_width=False):
                with st.spinner("Ejecutando episodios…"):
                    trajs, rewards = [], []
                    for _ in range(n_ep_anim):
                        t, r = rollout(model, _fn, _start, _goal, max_steps)
                        trajs.append(t); rewards.append(r)
                    rand_trajs = []
                    if show_random:
                        for _ in range(n_ep_anim):
                            t, _ = rollout(None, _fn, _start, _goal, max_steps)
                            rand_trajs.append(t)
                    st.session_state.update({"trajs": trajs, "rewards": rewards,
                                              "rand_trajs": rand_trajs or None})

            if "trajs" in st.session_state:
                trajs   = st.session_state["trajs"]
                rewards = st.session_state["rewards"]
                kk = st.columns(min(len(trajs), 4))
                for i, (r, t) in enumerate(zip(rewards, trajs)):
                    with kk[i % len(kk)]: st.metric(f"Ep {i+1}", f"R={r:.1f}", f"{len(t)} pasos")
                st.pyplot(fig_trajectories(trajs, rewards, _start, _goal, _code,
                                           st.session_state.get("rand_trajs")),
                          use_container_width=True)

        # ── Animación GIF ───────────────────────────────────────────────
        with subtab_anim:
            if st.button("🎬 Generar GIF", use_container_width=False):
                with st.spinner(f"Renderizando {n_ep_anim} eps a {fps_gif} fps…"):
                    trajs, rewards = [], []
                    for _ in range(n_ep_anim):
                        t, r = rollout(model, _fn, _start, _goal, max_steps)
                        trajs.append(t); rewards.append(r)
                    st.session_state["gif_bytes"] = make_gif(
                        trajs, rewards, _start, _goal, _code, fps=fps_gif)

            if "gif_bytes" in st.session_state:
                st.image(st.session_state["gif_bytes"], caption="Política SAC aprendida")
                st.download_button("⬇ Descargar GIF",
                                   data=st.session_state["gif_bytes"],
                                   file_name="policy_animation.gif",
                                   mime="image/gif", use_container_width=True)

        # ── Heatmap ─────────────────────────────────────────────────────
        with subtab_heat:
            n_heat = st.slider("Episodios para heatmap", 10, 100, 30, step=10)
            if st.button("🔥 Generar heatmap", use_container_width=False):
                with st.spinner(f"Ejecutando {n_heat} episodios…"):
                    ta, tr = [], []
                    for _ in range(n_heat):
                        t, _ = rollout(model, _fn, _start, _goal, max_steps)
                        ta.append(t)
                    if show_random:
                        for _ in range(n_heat):
                            t, _ = rollout(None, _fn, _start, _goal, max_steps)
                            tr.append(t)
                    st.session_state["heat_fig"] = fig_heatmap(
                        ta, _start, _goal, _code, tr if show_random else None)

            if "heat_fig" in st.session_state:
                st.pyplot(st.session_state["heat_fig"], use_container_width=True)