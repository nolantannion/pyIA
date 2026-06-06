from stable_baselines3 import SAC

model = SAC.load("C:\\Users\\nol4n\\Documents\\pyIA\\modelos_logs\\models\\sac_v1\\sac_200000_steps.zip")

total_params = sum(p.numel() for p in model.policy.parameters())

print(f"Parámetros totales: {total_params:,}")