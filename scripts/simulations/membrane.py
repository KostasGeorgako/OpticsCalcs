import numpy as np
import pandas as pd

# -----------------------------
# Physical parameters
# -----------------------------
# Membrane properties: space blanket / Mylar ~20 µm thick
h = 20e-6          # thickness [m]
rho = 1400         # density [kg/m^3] (approx. Mylar)
T = 5            # membrane tension [Pa] (N/m^2), adjust as needed
c_water = 1480     # speed of sound in water [m/s]
mu = rho * h       # mass per unit area [kg/m^2]

# Target sound frequency
f_target = 20000   # 20 kHz
lambda_sound = c_water / f_target  # wavelength in water
k1 = 0.383         # fundamental mode coefficient for circular membrane

# Diameters (mm) and pressures (Pa)
diameters_mm = np.array([5, 10, 20, 50, 80])
pressures_Pa = np.array([1e-6, 1e-3, 1, 10])  # µPa, mPa, Pa, 10Pa
radii_m = diameters_mm * 1e-3 / 2

# -----------------------------
# Simulation
# -----------------------------
results = []

for d_mm, r in zip(diameters_mm, radii_m):

    # Fundamental resonance of a circular membrane
    f1 = (k1 / r) * np.sqrt(T / mu)  # Hz

    # Directivity parameter
    ka = 2 * np.pi * r / lambda_sound

    row = {
        'Diam (mm)': d_mm,
        'r (mm)': round(r * 1000, 1),
        'f1 (kHz)': round(f1 / 1000, 1),
        'ka@20kHz': round(ka, 2)
    }

    for P in pressures_Pa:
        # Max static deflection of a circular membrane under uniform pressure
        # Classical thin membrane theory: δ_max = (P * r^2) / (4 * T)
        delta = (P * r**2) / (4 * T)  # [m]

        # Approximate slope at edge: d(w)/dr ~ 2*δ / r (linear approx.)
        slope_rad = 2 * delta / r

        # Reflected laser angle: θ_out = 2 * slope
        theta_laser = 2 * slope_rad  # rad

        row.update({
            f'δ@{P}Pa (pm)': f'{delta*1e12:.10f}',
            f'θ@{P}Pa (nrad)': f'{theta_laser*1e9:.10f}',
            f'θ@{P}Pa (µdeg)': f'{theta_laser*1e6:.10f}'
        })

    results.append(row)

df = pd.DataFrame(results)
print(df.to_markdown(index=False, tablefmt="pipe"))
