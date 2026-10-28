import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
STEP = 10                       # seconds between readings
N_MACHINES = 20
N_STEPS = 17280                 # 48 h of readings per machine

rows = []
for m in range(1, N_MACHINES + 1):
    life_scale = rng.uniform(0.7, 1.3)
    op_hours = 0.0
    last_maint = 0.0
    n_maint = 0
    health = 1.0
    for i in range(N_STEPS):
        hour = (i * STEP / 3600) % 24
        load = np.clip(0.6 + 0.3 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.05), 0.2, 1.0)

        since_maint = op_hours - last_maint
        wear = (0.00025 * life_scale) * (0.5 + load) * (1 + 0.02 * since_maint)
        health -= wear * (1 + rng.normal(0, 0.1))
        d = min(max(1 - health, 0.0), 1.0)

        temp  = 60 + 15 * load + 25 * d**2 + rng.normal(0, 1.0)
        vib   = 2.0 + 1.0 * load + 5 * d**2.5 + rng.normal(0, 0.15)
        pres  = 100 + 10 * load - 15 * d + rng.normal(0, 1.5 + 4 * d)
        rot   = 1500 * (0.7 + 0.3 * load) * (1 - 0.08 * d) + rng.normal(0, 10)
        power = 5 + 8 * load + 6 * d + rng.normal(0, 0.3)

        failed = health <= 0
        rows.append((m, i * STEP, temp, vib, pres, rot, power,
                     op_hours, n_maint, since_maint, health, failed))
        op_hours += STEP / 3600

        if failed:                                # failure -> maintenance -> restart
            n_maint += 1
            last_maint = op_hours
            health = 1.0

cols = ["machineID", "t_sec", "temp", "vib", "pressure", "rotation", "power",
        "op_hours", "n_maint", "hours_since_maint", "_health_hidden", "fail_event"]
df = pd.DataFrame(rows, columns=cols).round(3)
df.to_csv("factory_sim.csv", index=False)

print("Shape:", df.shape)
f = df.groupby("machineID").fail_event.sum()
print("Failures total:", int(f.sum()), "| per machine min/median/max:", f.agg(["min","median","max"]).tolist())
print(df[["temp","vib","pressure","rotation","power"]].describe().round(2))
print("Missing:", int(df.isna().sum().sum()))
