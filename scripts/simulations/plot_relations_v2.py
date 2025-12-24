import numpy as np
import matplotlib.pyplot as plt

# --- Constants from your SimulationConfig --- #
L_s = 10.2   # speaker membrane diameter (cm)
W_s = 4.5    # speaker width (cm)
C_s = 5.0    # speaker redirection cone length (cm)
L_t = 20.0   # target diameter (cm)

H_l = 0.0    # height from base to laser pointer center (cm)
H_s = 6.8    # height from base to membrane center (cm)
H_s_top = 12.6  # height from base to speaker top (cm)
h_tolerance = 7.4
H_max = H_s_top + h_tolerance

heta_min = 1
heta_max = (L_t - 1) / 2

# --- Helper functions --- #
def deg_to_rad(theta):
    return theta * np.pi / 180

def rad_to_deg(theta):
    return theta * 180 / np.pi

def cot(theta):
    return 1 / np.tan(theta)

def tan(theta):
    return np.tan(theta)

def atan(theta):
    return np.arctan(theta)

# --- Coverage calculation --- #
def coverage(theta_deg, delta_deg=1.0, D_l_s=5.0, D_o_t=30.0, H_m=9.0):
    """
    Calculate coverage given theta (deg) and optional parameters.
    """
    theta = deg_to_rad(theta_deg)
    delta = deg_to_rad(delta_deg)

    y = H_m - H_s
    phi = np.pi/4 + theta/2

    # Mirror calculations
    mh1 = y * cot(theta) * tan(theta + delta) / (1 + tan(phi) * tan(theta + delta))
    dd1 = mh1 * tan(phi)
    dd2 = (dd1 - (y + mh1) * tan(delta)) / (1 - tan(delta))
    mh2 = dd2

    d_m_t = D_o_t + dd2
    cov = d_m_t * tan(delta) + mh2

    return cov

# --- Generate theta values and compute coverage --- #
theta_values = np.linspace(1, 80, 300)  # degrees
coverage_values = [coverage(theta) for theta in theta_values]

# --- Plot --- #
plt.figure(figsize=(10, 6))
plt.plot(theta_values, coverage_values, color='blue', lw=2)
plt.xlabel("Theta (degrees)")
plt.ylabel("Coverage (cm)")
plt.title("Coverage vs. Theta")
plt.grid(True)
plt.show()
