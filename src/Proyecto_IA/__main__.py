"""
Entorno personalizado: partícula navegando un espacio continuo bajo un potencial.
Objetivo: encontrar la trayectoria óptima entre dos puntos minimizando el coste
          acumulado definido por el potencial V(x, y).

Dependencias:
    pip install gymnasium stable-baselines3 numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# 1. POTENCIAL  (modifica esta función para cambiar el paisaje)
# ---------------------------------------------------------------------------

def potential(x: float, y: float) -> float:
    """
    Potencial V(x, y). Valores altos = zonas costosas / barrera.
    """
    # # Barrera gaussiana 1
    # V1 = 3.0 * np.exp(-((x - 0.3)**2 + (y - 0.5)**2) / 0.02)
    # # Barrera gaussiana 2
    # V2 = 2.0 * np.exp(-((x - 0.7)**2 + (y - 0.5)**2) / 0.02)

    V = x**2 - y**2 #potencial tipo silla de montar
    return float(V)


# ---------------------------------------------------------------------------
# 2. ENTORNO GYMNASIUM
# ---------------------------------------------------------------------------

class ParticlePotentialEnv(gym.Env):
    """
    Entorno 2D continuo para navegación de partícula bajo un potencial.

    Espacio de observación : [x, y, vx, vy, dx_goal, dy_goal]
    Espacio de acción       : [ax, ay]  (aceleración, continua)
    Recompensa              : -dt * V(x,y)  -  penalización por distancia  +  bonus al llegar
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        start: tuple[float, float] = (0.1, 0.1),
        goal: tuple[float, float]  = (0.9, 0.9),
        max_steps: int  = 500,
        dt: float       = 0.02,
        max_speed: float = 1.0,
        max_accel: float = 1.0,
        goal_radius: float = 0.05,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.start       = np.array(start, dtype=np.float32)
        self.goal        = np.array(goal,  dtype=np.float32)
        self.max_steps   = max_steps
        self.dt          = dt
        self.max_speed   = max_speed
        self.max_accel   = max_accel
        self.goal_radius = goal_radius
        self.render_mode = render_mode

        # ---- espacios ----
        # Observación: [x, y, vx, vy, dx_goal, dy_goal]
        obs_low  = np.array([-0.1, -0.1, -max_speed, -max_speed, -1.2, -1.2], dtype=np.float32)
        obs_high = np.array([ 1.1,  1.1,  max_speed,  max_speed,  1.2,  1.2], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Acción: [ax, ay] en [-max_accel, max_accel]
        self.action_space = spaces.Box(
            low  = np.full(2, -max_accel, dtype=np.float32),
            high = np.full(2,  max_accel, dtype=np.float32),
        )

        # Estado interno
        self.pos   = self.start.copy()
        self.vel   = np.zeros(2, dtype=np.float32)
        self.steps = 0

        # Para renderizado
        self._fig = None
        self._trajectory: list[np.ndarray] = []

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        delta = self.goal - self.pos
        return np.concatenate([self.pos, self.vel, delta]).astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "distance_to_goal": float(np.linalg.norm(self.goal - self.pos)),
            "potential": potential(*self.pos),
            "steps": self.steps,
        }

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos   = self.start.copy()
        self.vel   = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self._trajectory = [self.pos.copy()]
        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.clip(action, -self.max_accel, self.max_accel)

        # Integración de Euler
        self.vel += action * self.dt
        # Limitar velocidad
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed

        self.pos += self.vel * self.dt
        self.steps += 1
        self._trajectory.append(self.pos.copy())

        # ---- recompensa ----
        V   = potential(*self.pos)
        dist = float(np.linalg.norm(self.goal - self.pos))

        reward = (
            - self.dt * V          # coste por el potencial
            - 0.1  * dist          # penalización por distancia al objetivo
            - 0.01 * float(np.dot(action, action))  # penalización por esfuerzo
        )

        # Condición de llegada
        terminated = dist < self.goal_radius
        if terminated:
            reward += 50.0   # bonus grande al llegar

        # Fuera de límites
        out_of_bounds = bool(np.any(self.pos < -0.05) or np.any(self.pos > 1.05))
        if out_of_bounds:
            reward -= 20.0

        truncated = (self.steps >= self.max_steps) or out_of_bounds

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    def render(self):
        """Visualización en tiempo real o como imagen."""
        # Construir mapa de potencial (cacheado la primera vez)
        if not hasattr(self, "_V_grid"):
            xs = np.linspace(0, 1, 200)
            ys = np.linspace(0, 1, 200)
            XX, YY = np.meshgrid(xs, ys)
            self._V_grid = np.vectorize(potential)(XX, YY)

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(5, 5))
            plt.ion()

        ax = self._ax
        ax.clear()
        ax.contourf(np.linspace(0, 1, 200), np.linspace(0, 1, 200),
                    self._V_grid, levels=30, cmap="hot_r", alpha=0.7)
        ax.contour (np.linspace(0, 1, 200), np.linspace(0, 1, 200),
                    self._V_grid, levels=10, colors="white", linewidths=0.4, alpha=0.5)

        if len(self._trajectory) > 1:
            traj = np.array(self._trajectory)
            ax.plot(traj[:, 0], traj[:, 1], "c-", linewidth=1.5, label="trayectoria")

        ax.plot(*self.start, "go", markersize=10, label="inicio")
        ax.plot(*self.goal,  "b*", markersize=12, label="objetivo")
        ax.plot(*self.pos,   "ro", markersize=8)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"Paso {self.steps} | V={potential(*self.pos):.2f}")
        ax.legend(loc="upper left", fontsize=7)
        plt.pause(0.001)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None


# ---------------------------------------------------------------------------
# 3. REGISTRO EN GYMNASIUM  (opcional pero recomendable)
# ---------------------------------------------------------------------------

gym.register(
    id="ParticlePotential-v0",
    entry_point=__name__ + ":ParticlePotentialEnv",
)


# ---------------------------------------------------------------------------
# 4. ENTRENAMIENTO CON STABLE BASELINES 3
# ---------------------------------------------------------------------------

def train(total_timesteps: int = 200_000, log_dir: str = "./logs/"):
    from stable_baselines3 import SAC           # SAC es ideal para espacios continuos
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback

    # Entorno de entrenamiento (vectorizado para acelerar)
    env = make_vec_env("ParticlePotential-v0", n_envs=4)

    # Entorno de evaluación
    eval_env = ParticlePotentialEnv()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path=log_dir,
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # Modelo SAC  (también puedes probar TD3 o PPO)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        tensorboard_log=log_dir,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("./models/particle_sac")
    print("✅  Modelo guardado en ./models/particle_sac")
    return model


# ---------------------------------------------------------------------------
# 5. EVALUACIÓN / VISUALIZACIÓN DE LA POLÍTICA APRENDIDA
# ---------------------------------------------------------------------------

def evaluate(model_path: str = "./models/particle_sac", episodes: int = 3):
    from stable_baselines3 import SAC

    model = SAC.load(model_path)
    env   = ParticlePotentialEnv(render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episodio {ep+1}: recompensa={total_reward:.2f} | "
              f"dist_final={info['distance_to_goal']:.4f} | "
              f"pasos={info['steps']}")

    env.close()


# ---------------------------------------------------------------------------
# 6. PUNTO DE ENTRADA
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",  choices=["train", "eval", "check"], default="check")
    parser.add_argument("--steps", type=int, default=200_000)
    args = parser.parse_args()

    if args.mode == "check":
        # Comprobación rápida del entorno sin RL
        from stable_baselines3.common.env_checker import check_env
        env = ParticlePotentialEnv()
        check_env(env, warn=True)
        print("✅  Entorno válido para Stable Baselines 3")

        # Episodio aleatorio para ver que funciona
        obs, _ = env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break
        print("Obs final:", obs, "| Info:", info)

    elif args.mode == "train":
        train(total_timesteps=args.steps)

    elif args.mode == "eval":
        evaluate()