import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import RegularGridInterpolator

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

# [Aquí va tu clase ParticleContEnv exacta]
class ParticleContEnv(gym.Env):
    def __init__(self, grid, max_steps=1000, goal_radius=1.0, step_size=1.0, render_mode=None):
        super().__init__()
        self.grid = (grid - grid.min()) / (grid.max() - grid.min())
        self.H, self.W = self.grid.shape
        self.max_steps = max_steps
        self.goal_radius = goal_radius
        self.step_size = step_size
        self.render_mode = render_mode

        rows = np.arange(self.H)
        cols = np.arange(self.W)
        self._V_interp = RegularGridInterpolator((rows, cols), self.grid, method="linear", bounds_error=False, fill_value=1.0)

        gi, gj = np.gradient(self.grid)
        grad_mag = np.sqrt(gi**2 + gj**2)
        self._G_interp = RegularGridInterpolator((rows, cols), grad_mag, method="linear", bounds_error=False, fill_value=0.0)

        self.action_space = spaces.Box(low=np.array([-np.pi], dtype=np.float32), high=np.array([np.pi], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros(4, dtype=np.float32), high=np.ones(4, dtype=np.float32), dtype=np.float32)

    def _query(self, i, j):
        pt = np.array([[i, j]])
        return float(self._V_interp(pt)[0]), float(self._G_interp(pt)[0])

    def _obs(self):
        return np.array([self.i / (self.H - 1), self.j / (self.W - 1), self.goal_i / (self.H - 1), self.goal_j / (self.W - 1)], dtype=np.float32)

    def _clip(self, i, j):
        return float(np.clip(i, 0, self.H - 1)), float(np.clip(j, 0, self.W - 1))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = float(self.np_random.uniform(0, self.H - 1))
        self.j = float(self.np_random.uniform(0, self.W - 1))
        self.goal_i = float(self.np_random.uniform(0, self.H - 1))
        self.goal_j = float(self.np_random.uniform(0, self.W - 1))
        self.steps = 0
        return self._obs(), {}

    def step(self, action):
        theta = float(action[0])
        prev_i, prev_j = self.i, self.j

        self.i, self.j = self._clip(self.i + self.step_size * np.sin(theta), self.j + self.step_size * np.cos(theta))
        self.steps += 1

        prev_dist = np.linalg.norm([self.goal_i - prev_i, self.goal_j - prev_j])
        dist = np.linalg.norm([self.goal_i - self.i,  self.goal_j - self.j])

        V, G = self._query(self.i, self.j)

        move_vec = np.array([self.i - prev_i, self.j - prev_j])
        goal_vec = np.array([self.goal_i - prev_i, self.goal_j - prev_j])
        norm_g = np.linalg.norm(goal_vec)
        alignment = float(np.dot(move_vec, goal_vec) / (self.step_size * norm_g)) if norm_g > 1e-6 else 0.0

        reward = 3.0 * (prev_dist - dist) + 1.0 * alignment - 2.0 * V - 1.0 * G - 0.01

        terminated = dist < self.goal_radius
        if terminated: reward += 20.0

        truncated = self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"dist": dist, "V": V}


# ==========================================
# SCRIPT DE ENTRENAMIENTO
# ==========================================

if __name__ == "__main__":
    # 1. Crear un 'grid' sintético de prueba (un terreno con colinas y valles)
    grid_size = 50
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    # Una función que crea crestas y valles para que el agente los navegue
    dummy_grid = np.sin(X) + np.cos(Y) 

    # 2. Función para instanciar el entorno
    def make_env():
        # Reduzco max_steps a 200 para que los episodios terminen más rápido en entrenamiento
        return ParticleContEnv(grid=dummy_grid, max_steps=200, goal_radius=1.5, step_size=1.0)

    # 3. Vectorizar y Normalizar el entorno (CRUCIAL para PPO)
    # DummyVecEnv es necesario para Stable Baselines
    env = DummyVecEnv([make_env])
    
    # VecNormalize estabiliza las recompensas densas y reduce oscilaciones salvajes
    env = VecNormalize(
        env,
        norm_obs=False,       # Ya normalizas las obs en tu clase (dividiendo por H-1, W-1)
        norm_reward=True,     # ESTO ES CLAVE para estabilizar PPO con tus +20, -2, etc.
        clip_reward=10.0      # Evita picos de recompensa gigantes
    )

    # 4. Configurar el modelo PPO con hiperparámetros defensivos
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,       # Batch size generoso para suavizar gradientes en espacios continuos
        gamma=0.99,
        clip_range=0.2,       # Clipping de la política
        clip_range_vf=0.2,    # CLIP DEL CRÍTICO: Obliga a la Value Loss a no dar saltos bestiales
        ent_coef=0.01,        # Coeficiente de entropía: obliga al agente a explorar distintas direcciones
        verbose=1,
        tensorboard_log="./ppo_particle_tensorboard/"
    )

    # 5. (Opcional) Evaluar periódicamente durante el entrenamiento
    # Para evaluar, necesitamos otro entorno sin normalización de recompensa (para ver la real)
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
    
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./logs/best_model',
        log_path='./logs/results', 
        eval_freq=5000, # Evalúa cada 5000 steps
        deterministic=True, 
        render=False
    )

    # 6. Entrenar el modelo
    print("Iniciando el entrenamiento de PPO...")
    # 300,000 steps es un buen punto de partida para ver si la policy converge
    model.learn(total_timesteps=50000, callback=eval_callback, progress_bar=True)

    # 7. Guardar el modelo final y el normalizador
    model.save("ppo_particle_final")
    env.save("vec_normalize.pkl")
    print("Modelo guardado correctamente.")

    # 8. Test rápido para ver cómo se comporta
    print("Probando el agente entrenado...")
    obs = env.reset()
    for _ in range(20): # Prueba de 20 steps
        # Deterministic=True para que elija la mejor acción sin exploración aleatoria
        action, _states = model.predict(obs, deterministic=True) 
        obs, rewards, dones, infos = env.step(action)
        # Observa cómo baja la métrica 'dist'
        print(f"Distancia a meta: {infos[0]['dist']:.2f} | Recompensa normalizada: {rewards[0]:.2f}")
        if dones[0]:
            print("¡Meta alcanzada o episodio truncado!")
            obs = env.reset()