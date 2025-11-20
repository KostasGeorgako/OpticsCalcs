import numpy as np

def deg_to_rad(theta):
    return theta * np.pi / 180

def rad_to_deg(theta):
    return theta * 180 / np.pi

delta = 1.65
Lb = 10
Ls = 7
h_gap = 5
w_gap = 10
hs = h_gap/2 + Ls/2
H = Ls + h_gap + Lb


heta_min = 0.5 **2
heta_max = 0.9 **2

def get_speaker_distance_bounds(theta, delta, Lb, heta_min, heta_max):
    factor = Lb / ( 2 * (np.tan(deg_to_rad(theta + delta)) - np.tan(deg_to_rad(theta))) )

    return [heta_min * factor, heta_max * factor]


for theta in np.arange(0.0, 45.0, 0.1):
    if np.tan(deg_to_rad(theta)) <= 0:
        continue

    bounds = get_speaker_distance_bounds(theta, delta, Lb, heta_min, heta_max)

    D = hs / np.tan(deg_to_rad(theta))

    if bounds[0] >= D + w_gap:
        print(f"theta={theta: .2f}, D={D: .2f}, x={bounds[0]: .2f}")
