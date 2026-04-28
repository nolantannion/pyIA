import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from pathlib import Path
import os
import glob

def V(x,y):
    return 0

def grad(x,y):
    return [0,0]

# sampler
def sample_potential():
    return V, grad


def V_test(x, y):
    return 0.3*np.sin(4*x)*np.cos(4*y)

def grad_test(x, y):
    gx = 0.3*4*np.cos(4*x)*np.cos(4*y)
    gy = -0.3*4*np.sin(4*x)*np.sin(4*y)

    return np.array([gx, gy], dtype=np.float32)

def V_test1(x, y):
    return (
        0.4*np.sin(5*x) * np.cos(4*y)
        + 0.3*np.sin(2*x + y)
        + np.exp(-((x-0.2)**2 + (y-0.8)**2)/0.01)
        - 1.2*np.exp(-((x-0.7)**2 + (y-0.3)**2)/0.02)
    )

def grad_test1(x, y):
    gx = (
        0.4*5*np.cos(5*x)*np.cos(4*y)
        + 0.3*2*np.cos(2*x + y)
        - (2*(x-0.2)/0.01)*np.exp(-((x-0.2)**2 + (y-0.8)**2)/0.01)
        + 1.2*(2*(x-0.7)/0.02)*np.exp(-((x-0.7)**2 + (y-0.3)**2)/0.02)
    )

    gy = (
        -0.4*4*np.sin(5*x)*np.sin(4*y)
        + 0.3*np.cos(2*x + y)
        - (2*(y-0.8)/0.01)*np.exp(-((x-0.2)**2 + (y-0.8)**2)/0.01)
        + 1.2*(2*(y-0.3)/0.02)*np.exp(-((x-0.7)**2 + (y-0.3)**2)/0.02)
    )

    return np.array([gx, gy], dtype=np.float32)



class FastGradEnv(gym.Env):
    def __init__(self, max_steps=350, weights=None):
        super().__init__()

        # Parametros del sistema
        self.max_steps = max_steps
        self.dt = 0.02
        self.speed = 0.25

        self.energy = 0.0
        self.path_length = 0.0
        self.start_pos = np.zeros(2)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)


        # Lo que devuelve obs [x, y, dx_goal, dy_goal, V, gx, gy]:
        #   [0] x        - posición en x          (en [0, 1])
        #   [1] y        - posición en y          (en [0, 1])
        #   [2] dx_goal  - delta x hacia el goal  (en [-1, 1])
        #   [3] dy_goal  - delta y hacia el goal  (en [-1, 1])
        #   [4] V        - potencial local         (en [-2, 2])
        #   [5] gx       - gradiente en x          (en [-1, 1])
        #   [6] gy       - gradiente en y          (en [-1, 1])
        self.observation_space = spaces.Box(
            low=np.array( [0.0,  0.0,  -1.0, -1.0, -2.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0,   1.0,  1.0,  2.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32
        )


        self.weights = weights or {
            "progress": 5,
            "align": 1.2,
            "slope": 0.15,
            "potential": 0.6,
            "step": 0.02
        }

    # Funcion de reinicio, elige un nuevo potencial, posiciones y goal aleatorios
    # evita exactamente situarsse en el borde de la caja
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.V_fn, self.grad_fn = sample_potential()

        self.pos  = self.np_random.uniform(0.05, 0.95, 2).astype(np.float32)
        self.goal = self.np_random.uniform(0.05, 0.95, 2).astype(np.float32)

        # Almacenamos la posicion inicial
        self.start_pos = self.pos.copy()

        self.steps = 0


        self.energy      = 0.0
        self.path_length = 0.0

        return self._obs(), {}

    def _obs(self):
        V    = float(np.clip(self.V_fn(self.pos[0], self.pos[1]), -2, 2))
        grad = self.grad_fn(self.pos[0], self.pos[1])

        # Normalizar el vector dirección al goal para que quede en [-1, 1]
        delta     = self.goal - self.pos
        dist      = np.linalg.norm(delta) + 1e-8
        dir_goal  = (delta / dist).astype(np.float32)

        # Normalizar gradiente a [-1, 1] para respetar observation_space
        grad_norm = grad / (np.linalg.norm(grad) + 1e-8)

        return np.concatenate([
            self.pos,       # [0, 1]
            dir_goal,       # [-1, 1]
            [V],            # [-2, 2]
            grad_norm       # [-1, 1]
        ]).astype(np.float32)

    def step(self, action):
        # El agente elige un angulo y se mueve en esa direccion
        # es necesario action[0] porque esta definido como un vector
        theta     = float(action[0] * np.pi)
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)

        prev_pos  = self.pos.copy()

        # Actualizamos la posicion
        self.pos  = prev_pos + self.speed * direction * self.dt
        self.steps += 1

        # Distancia a la meta y distancia previa
        dist      = np.linalg.norm(self.goal - self.pos)
        prev_dist = np.linalg.norm(self.goal - prev_pos)

        # Calculamos el progreso
        r_progress = prev_dist - dist

        # Comprobar la direccion y alineamiento. Usamos prev porque ya hemos actualizado posiciones
        goal_dir   = (self.goal - prev_pos) / (prev_dist + 1e-8)
        r_align    = float(np.dot(direction, goal_dir))

        V    = float(np.clip(self.V_fn(self.pos[0], self.pos[1]), -2, 2))
        grad = self.grad_fn(self.pos[0], self.pos[1])

        r_slope = float(-np.dot(direction, grad))

        # Añadimos la energia y longitud del paso
        self.energy      += V
        self.path_length += self.speed * self.dt


        # Funcion de reward ajustada con los pesos proporcionados o por defecto
        w = self.weights
        reward = (
              w["progress"]  * r_progress
            + w["align"]     * r_align
            + w["slope"]     * r_slope
            - w["potential"] * V
            - w["step"]
        )

        # Comprobamos si esta fuera y penalizamos
        out_lim = np.any((self.pos < 0.0) | (self.pos > 1.0))
        if out_lim:
            reward -= 2.0

        # Comprobamos si llega al destino y recompensamos
        terminated = dist < 0.02
        if terminated:
            reward += 20.0

        # Comprobamos si alcanza el limite de pasos
        truncated = self.steps >= self.max_steps

        info = {}
        if terminated or truncated:
            direct_dist = np.linalg.norm(self.goal - self.start_pos)
            efficiency  = direct_dist / (self.path_length + 1e-8)
            info = {
                "energy":     self.energy,
                "success":    float(terminated),
                "efficiency": efficiency
            }

        return self._obs(), reward, terminated, truncated, info
    
# Super clase que hereda de la clase anterior pero usa un potencial fijo para el test
class TestEnv(FastGradEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.V_fn = V_test
        self.grad_fn = grad_test

        # Evita casos de inicio demasiado cercano
        while True:
            self.pos = self.np_random.uniform(0.01,0.99,2)
            self.goal = self.np_random.uniform(0.01,0.99,2)

            if np.linalg.norm(self.goal - self.pos) < 0.7:
                break

        self.start_pos = self.pos.copy()

        return self._obs(), {}


# Super clase que hereda de la clase anterior pero usa un potencial fijo para el test
class TestEnv1(FastGradEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.V_fn = V_test1
        self.grad_fn = grad_test1

        # Evita casos de inicio demasiado cercano
        while True:
            self.pos = self.np_random.uniform(0.01,0.99,2)
            self.goal = self.np_random.uniform(0.01,0.99,2)

            if np.linalg.norm(self.goal - self.pos) < 0.7:
                break

        self.start_pos = self.pos.copy()

        return self._obs(), {}
    

def evaluate(model, env, n_episodes=20, plot_hist=True, max_trials=500, seed = None):
    """
    Evalúa el modelo usando SOLO trayectorias exitosas.
    Ejecuta episodios hasta obtener n_episodes éxitos.

    INPUT:

    - model: modelo a evaluar
    - env: entorno para el modelo
    - n_episodes (opc): evaluaciones
    - plot_hist (opc): muestra un histograma de energias 
    - max_trials (opc): maximo numero de intentos para alcanzar los exitos totales
    - seed (opc): semilla para los numeros pseudoaleatorios

    RETURNS:

    Devuelve un diccionario que contiene:

    - success_rate: tasa de exito
    - energy_rl: energia promedio en la trayectoria del modelo
    - energy_linear: energia promedio en la trayectoria lineal
    - reward: recompensa final obtenida
    - efficiency: cociente longitud lineal entre trayectoria del modelo
    - length: longitud promedio de la trayectoria 
    - episodes_used: episodios exitosos
    - total_trials: episodios para llegar a n_episodes exitos
    """

    rewards, lengths, energies, effs = [], [], [], []
    linear_energies = []

    successes = 0
    trials = 0

    if seed == None:
        seed[0] = np.random.randint(0,10_000_000, 1)

    i_random = 0

    while successes < n_episodes and trials < max_trials:
        trials += 1

        obs, _ = env.reset(seed = seed + i_random)
        start = env.pos.copy()
        goal = env.goal.copy()

        traj = [start.copy()]
        total_r = 0

        done = False
        term = False

        while not done:
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            action = action[0]

            obs, r, term, trunc, _ = env.step(action)

            total_r += r
            traj.append(env.pos.copy())
            done = term or trunc

        if not term:
            i_random += 1
            continue  # ignorar episodio fallido

        # -------- solo éxitos --------
        successes += 1
        i_random += 1


        traj = np.array(traj)

        path_length = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        direct_dist = np.linalg.norm(start - goal)

        n_points = len(traj)
        xs = np.linspace(start[0], goal[0], n_points)
        ys = np.linspace(start[1], goal[1], n_points)

        energy_lin = 0.0
        for x, y in zip(xs, ys):
            energy_lin += env.V_fn(x, y)

        rewards.append(total_r)
        lengths.append(len(traj))
        energies.append(env.energy)
        linear_energies.append(energy_lin)
        effs.append(direct_dist / (path_length + 1e-8))

    # seguridad
    if successes < n_episodes:
        print(f"Warning: solo {successes} éxitos en {trials} intentos")

    rewards = np.array(rewards)
    lengths = np.array(lengths)
    energies = np.array(energies)
    linear_energies = np.array(linear_energies)
    effs = np.array(effs)

    if plot_hist and successes > 0:
        plt.figure()

        plt.hist(energies, bins=25, alpha=0.6, label="RL energy", color = 'r')
        plt.hist(linear_energies, bins=25, alpha=0.6, label="Linear energy", color = 'b')

        plt.axvline(np.mean(energies), linestyle="--", color = 'r', label="Mean RL")
        plt.axvline(np.mean(linear_energies), linestyle="--",color = 'b', label="Mean Linear")

        plt.xlabel("Energy")
        plt.ylabel("Frecuencia")
        plt.title("Distribución de energías (solo éxitos)")
        plt.legend()
        plt.show()

    return {
        "success_rate": successes / trials,
        "energy_rl": float(np.mean(energies)) if successes > 0 else np.nan,
        "energy_linear": float(np.mean(linear_energies)) if successes > 0 else np.nan,
        "reward": float(np.mean(rewards)) if successes > 0 else np.nan,
        "efficiency": float(np.mean(effs)) if successes > 0 else np.nan,
        "length": float(np.mean(lengths)) if successes > 0 else np.nan,
        "episodes_used": successes,
        "total_trials": trials
    }