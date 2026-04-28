import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from typing import Callable


# ---------------------------------------------------------------------------
# 1. POTENCIAL POR DEFECTO
# ---------------------------------------------------------------------------

def default_potential(x: float, y: float) -> float:
    """Dos barreras gaussianas — paisaje de referencia."""
    # V1 = 3.0 * np.exp(-((x - 0.3)**2 + (y - 0.5)**2) / 0.02)
    # V2 = 2.0 * np.exp(-((x - 0.7)**2 + (y - 0.5)**2) / 0.02)
    # return float(V1 + V2)
    return 0.0

# Alias para compatibilidad con scripts anteriores
potential = default_potential


# ---------------------------------------------------------------------------
# 2. ENTORNO GYMNASIUM
# ---------------------------------------------------------------------------

class ParticlePotentialEnv(gym.Env):
    """
    Entorno 2D continuo para navegación bajo un potencial configurable.

    Todos los parámetros se pasan en __init__, por lo que el mismo código
    de entrenamiento puede usarse con cualquier configuración sin tocar el
    archivo.

    Args:
        potential_fn : V(x,y) → float.  None → usa default_potential.
        start        : posición inicial (x0, y0)  en [0,1]²
        goal         : posición objetivo (xg, yg) en [0,1]²
        max_steps    : pasos máximos por episodio
        dt           : paso de integración de Euler
        max_speed    : velocidad máxima
        max_accel    : rango de acción (aceleración)
        goal_radius  : distancia de éxito al objetivo
        render_mode  : "human" | None
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        potential_fn: Callable[[float, float], float] | None = None,
        start:        tuple[float, float] = (0.1, 0.1),
        goal:         tuple[float, float] = (0.9, 0.9),
        max_steps:    int   = 500,
        dt:           float = 0.05,
        max_speed:    float = 1.0,
        max_accel:    float = 1.0,
        goal_radius:  float = 0.05,
        render_mode:  str | None = None,
    ):
        super().__init__()

        self.potential_fn = potential_fn if potential_fn is not None else default_potential
        self.start        = np.array(start, dtype=np.float32)
        self.goal         = np.array(goal,  dtype=np.float32)
        self.max_steps    = max_steps
        self.dt           = dt
        self.max_speed    = max_speed
        self.max_accel    = max_accel
        self.goal_radius  = goal_radius
        self.render_mode  = render_mode

        obs_low  = np.array([-0.1, -0.1, -max_speed, -max_speed, -1.2, -1.2], dtype=np.float32)
        obs_high = np.array([ 1.1,  1.1,  max_speed,  max_speed,  1.2,  1.2], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.full(2, -max_accel, dtype=np.float32),
            high=np.full(2,  max_accel, dtype=np.float32),
        )

        self.pos   = self.start.copy()
        self.vel   = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self._fig  = None
        self._trajectory: list[np.ndarray] = []

    # ------------------------------------------------------------------
    def _get_obs(self):
        return np.concatenate([self.pos, self.vel, self.goal - self.pos]).astype(np.float32)

    def _get_info(self):
        return {
            "distance_to_goal": float(np.linalg.norm(self.goal - self.pos)),
            "potential":        float(self.potential_fn(*self.pos)),
            "steps":            self.steps,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos   = self.start.copy()
        self.vel   = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self._trajectory = [self.pos.copy()]
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_accel, self.max_accel)
        self.vel += action * self.dt
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed
        self.pos  += self.vel * self.dt
        self.steps += 1
        self._trajectory.append(self.pos.copy())

        V    = float(self.potential_fn(*self.pos))
        dist = float(np.linalg.norm(self.goal - self.pos))

        prev_dist = np.linalg.norm(self.goal - (self.pos - self.vel * self.dt))
        dist = np.linalg.norm(self.goal - self.pos)

        reward = (
            4.0 * (prev_dist - dist)   # progreso (MUY importante)
            - 0.002 * V
            - 0.01 * np.dot(action, action)
        )

        terminated    = dist < self.goal_radius
        if terminated:
            reward += 50.0

        out_of_bounds = bool(np.any(self.pos < -0.05) or np.any(self.pos > 1.05))
        if out_of_bounds:
            reward -= 20.0

        truncated = (self.steps >= self.max_steps) or out_of_bounds

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if not hasattr(self, "_V_grid"):
            xs = np.linspace(0, 1, 200)
            ys = np.linspace(0, 1, 200)
            XX, YY = np.meshgrid(xs, ys)
            self._V_grid = np.vectorize(self.potential_fn)(XX, YY)
        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))
            plt.ion()
        ax = self._ax; ax.clear()
        ax.contourf(np.linspace(0,1,200), np.linspace(0,1,200),
                    self._V_grid, levels=30, cmap="hot_r", alpha=0.7)
        if len(self._trajectory) > 1:
            t = np.array(self._trajectory)
            ax.plot(t[:,0], t[:,1], "c-", lw=1.5)
        ax.plot(*self.start,"go",ms=10); ax.plot(*self.goal,"b*",ms=12); ax.plot(*self.pos,"ro",ms=8)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(f"Paso {self.steps} | V={self.potential_fn(*self.pos):.2f}")
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig); self._fig = None


# ---------------------------------------------------------------------------
# 3. REGISTRO (parámetros por defecto)
# ---------------------------------------------------------------------------

gym.register(id="ParticlePotential-v0", entry_point=__name__ + ":ParticlePotentialEnv")


# ---------------------------------------------------------------------------
# 4. ENTRENAMIENTO (todos los parámetros configurables)
# ---------------------------------------------------------------------------

def train(
    total_timesteps: int   = 200_000,
    log_dir:         str   = "./logs/",
    model_path:      str   = "./models/particle_sac",
    potential_fn:    Callable | None = None,
    start:           tuple = (0.1, 0.1),
    goal:            tuple = (0.9, 0.9),
    max_steps:       int   = 500,
    lr:              float = 3e-4,
    buffer_size:     int   = 100_000,
    batch_size:      int   = 256,
    n_envs:          int   = 4,
):
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env  import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    import os

    os.makedirs(log_dir,                     exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    env_kwargs = dict(potential_fn=potential_fn, start=start, goal=goal, max_steps=max_steps)

    def make_env(rank):
        def _init():
            e = ParticlePotentialEnv(**env_kwargs)
            return Monitor(e, filename=os.path.join(log_dir, f"env_{rank}"))
        return _init

    env      = DummyVecEnv([make_env(i) for i in range(n_envs)])
    eval_env = Monitor(ParticlePotentialEnv(**env_kwargs))

    cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(model_path) + "/best/",
        log_path=log_dir,
        eval_freq=max(10_000 // n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    model = SAC("MlpPolicy", env, verbose=0,
                learning_rate=lr, buffer_size=buffer_size,
                batch_size=batch_size, tensorboard_log=log_dir, device= "cuda")

    model.learn(total_timesteps=total_timesteps, callback=cb)
    model.save(model_path)
    return model


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "check"], default="check")
    parser.add_argument("--steps", type=int, default=20_000)
    args = parser.parse_args()

    if args.mode == "check":
        from stable_baselines3.common.env_checker import check_env
        env = ParticlePotentialEnv()
        check_env(env, warn=True)
        print("✅  Entorno válido")
    elif args.mode == "train":
        train(total_timesteps=args.steps)