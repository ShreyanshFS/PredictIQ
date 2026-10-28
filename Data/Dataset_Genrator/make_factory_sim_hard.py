import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
STEP = 10
N_MACHINES = 20
N_STEPS = 17280   # ~48h nominal per machine

rows = []
for m in range(1, N_MACHINES + 1):
    # heterogeneous machines: different wear behavior, some noisier, some drift
    life_scale   = rng.uniform(0.5, 1.8)          # wider spread than before
    noise_mult   = rng.uniform(0.8, 3.0)           # some machines are just noisier
    drift_temp   = rng.uniform(-0.00015, 0.00025)  # slow sensor calibration drift/hour
    drift_pres   = rng.uniform(-0.0002, 0.0002)
    stuck_prob   = rng.uniform(0.0, 0.02)          # some machines have flaky sensors
    dropout_prob = rng.uniform(0.01, 0.05)         # missing readings
    shock_prob   = 0.00015                          # random sudden health hits (unrelated to wear)
    maint_miss_p = rng.uniform(0.0, 0.15)           # maintenance sometimes doesn't fully reset health
    maint_log_late = rng.uniform(0, 3)              # maint logged late by up to N readings

    t = 0
    op_hours = 0.0
    last_maint = 0.0
    n_maint = 0
    health = 1.0
    stuck_vals = {}   # sensor -> (value, remaining_steps)
    pending_maint_log = 0  # steps until a maintenance event actually gets logged

    for i in range(N_STEPS):
        # irregular timestamps: occasionally skip a reading (simulate dropped packets)
        step = STEP if rng.random() > 0.03 else STEP * rng.integers(2, 4)
        hour = (t / 3600) % 24
        load = np.clip(0.6 + 0.3*np.sin(2*np.pi*hour/24) + rng.normal(0, 0.08), 0.15, 1.05)

        since_maint = op_hours - last_maint
        wear = (0.00025 * life_scale) * (0.5 + load) * (1 + 0.02 * since_maint)
        health -= wear * (1 + rng.normal(0, 0.15))

        # random shock: sudden unexplained health hit (real machines fail from causes
        # not always visible in these 5 sensors - e.g. a bearing defect, bad batch of parts)
        if rng.random() < shock_prob:
            health -= rng.uniform(0.05, 0.25)

        d = min(max(1 - health, 0.0), 1.0)
        h_since = op_hours - last_maint

        # base signal (same physics as before) but noisier + drifting + occasional spikes
        temp  = 60 + 15*load + 25*d**2 + drift_temp*op_hours + rng.normal(0, 1.0*noise_mult)
        vib   = 2.0 + 1.0*load + 5*d**2.5 + rng.normal(0, 0.15*noise_mult)
        pres  = 100 + 10*load - 15*d + drift_pres*op_hours*100 + rng.normal(0, (1.5+4*d)*noise_mult)
        rot   = 1500*(0.7+0.3*load)*(1-0.08*d) + rng.normal(0, 10*noise_mult)
        power = 5 + 8*load + 6*d + rng.normal(0, 0.3*noise_mult)

        # occasional outlier spike (sensor glitch, not real)
        if rng.random() < 0.002:
            spike_sensor = rng.choice(["temp","vib","pressure","rotation","power"])
            spike_mult = rng.uniform(2, 5) * rng.choice([-1, 1])

        vals = {"temp":temp, "vib":vib, "pressure":pres, "rotation":rot, "power":power}

        # stuck sensor: a sensor freezes at its last value for a while
        for s in list(stuck_vals.keys()):
            val, remain = stuck_vals[s]
            if remain > 0:
                vals[s] = val
                stuck_vals[s] = (val, remain - 1)
            else:
                del stuck_vals[s]
        if rng.random() < stuck_prob:
            s = rng.choice(list(vals.keys()))
            stuck_vals[s] = (vals[s], rng.integers(20, 200))

        # random missing readings (sensor dropout -> NaN)
        for s in list(vals.keys()):
            if rng.random() < dropout_prob:
                vals[s] = np.nan

        failed = health <= 0
        n_maint_report = n_maint
        hours_since_maint_report = h_since

        # maintenance logging can be late or missing (real logs aren't perfect)
        if pending_maint_log > 0:
            pending_maint_log -= 1
        if pending_maint_log == 0 and failed:
            pass  # handled below after health reset

        rows.append((m, t, vals["temp"], vals["vib"], vals["pressure"], vals["rotation"], vals["power"],
                     op_hours, n_maint_report, hours_since_maint_report, health, failed))

        t += step
        op_hours += step / 3600

        if failed:
            n_maint += 1
            # maintenance sometimes doesn't fully fix the machine (partial repair)
            if rng.random() < maint_miss_p:
                health = rng.uniform(0.3, 0.7)   # imperfect reset
            else:
                health = 1.0
            last_maint = op_hours

cols = ["machineID","t_sec","temp","vib","pressure","rotation","power",
        "op_hours","n_maint","hours_since_maint","_health_hidden","fail_event"]
df = pd.DataFrame(rows, columns=cols)
df[["temp","vib","pressure","rotation","power"]] = df[["temp","vib","pressure","rotation","power"]].round(3)
df["op_hours"] = df["op_hours"].round(4)
df["hours_since_maint"] = df["hours_since_maint"].round(4)
df["_health_hidden"] = df["_health_hidden"].round(4)
df.to_csv("factory_sim_hard.csv", index=False)

print("Shape:", df.shape)
f = df.groupby("machineID").fail_event.sum()
print("Failures total:", int(f.sum()), "| per machine min/median/max:", f.agg(["min","median","max"]).tolist())
print("Missing values per column:\n", df[["temp","vib","pressure","rotation","power"]].isna().sum())
print("Time step gaps (unique, sample):", sorted(df.groupby("machineID")["t_sec"].diff().dropna().unique().tolist())[:10])
print(df[["temp","vib","pressure","rotation","power"]].describe().round(2))
