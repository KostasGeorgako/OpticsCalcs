import numpy as np
import random

def deg_to_rad(ang):
    return np.pi / 180 * ang

def rad_to_deg(ang):
    return 180 / np.pi * ang

max_refraction_delta_angle = 10

b = np.tan(deg_to_rad(max_refraction_delta_angle))
heta_boundaries = [ (b**2 - np.sqrt(b**4 + b**2) + 1/2),(b**2 + np.sqrt(b**4 + b**2) + 1/2) ]

print(f"For max refraction: {max_refraction_delta_angle} degrees")
print(f"--- Choose n values in the range ({0.0}, {heta_boundaries[0]:.2f}) or ({heta_boundaries[1]:.2f}, {1.0:.2f})")

heta_max = random.uniform(heta_boundaries[1], 1.0) if random.uniform(0, 1.0) >= 0.5 else random.uniform(0.0, heta_boundaries[0])
heta_min = random.uniform(heta_boundaries[0], heta_boundaries[1])

heta_min = 0.001
heta_max = 0.1

print(f"Randomly chosen n (min,max)=({heta_min:.2f},{heta_max: .2f})")


discriminator_max = 1 - 4*heta_max + 4*(heta_max**2) - 8*heta_max*(b**2)
discriminator_min = 1 - 4*heta_min + 4*(heta_min**2) - 8*heta_min*(b**2)

# if discriminators are negative, the respective inequalities hold for every value,
# , thus we can assume the wide range of 0.0 - 45.0
sols_max, sols_min = np.array([0.0, 45.0]), np.array([0.0, 45.0])
if discriminator_max >= 0:
    sols_max = (np.array([-np.sqrt(discriminator_max), +np.sqrt(discriminator_max)]) * (1/(4*heta_max*b)) + (1-2*heta_max) / (4*heta_max*b))

if discriminator_min >= 0.0:
    sols_min = (np.array([-np.sqrt(discriminator_min), +np.sqrt(discriminator_min)]) * (1/(4*heta_min*b)) + (1-2*heta_min) / (4*heta_min*b))

# angles are negative
sols_max = np.abs(sols_max)
sols_min = np.abs(sols_min)

sols_max = np.array([sols_max[1], sols_max[0]]) if sols_max[0] > sols_max[1] else sols_max
sols_min = np.array([sols_min[1], sols_min[0]]) if sols_min[0] > sols_min[1] else sols_min

# print(sols_max)
# print(sols_min)

laser_angles_max = rad_to_deg(np.atan(sols_max))
# laser_angles_max = laser_angles_max.clip(0.0, 45.0)

laser_angles_min = rad_to_deg(np.atan(sols_min))
# laser_angles_min = laser_angles_min.clip(0.0, 45.0)


print(f"Max laser angle in ({laser_angles_max[0]:.2f}, {laser_angles_max[1]:.2f})")
print(f"Min laser angle in ({laser_angles_min[0]:.2f}, {laser_angles_min[1]:.2f})")

min_laser_angle = max(laser_angles_min[0], laser_angles_max[0])
max_laser_angle = min(laser_angles_min[1], laser_angles_min[1])

print(f"Choose laser angle in intersection ({min_laser_angle: .2f}, {max_laser_angle: .2f})")

chosen_angle = random.uniform(min_laser_angle, max_laser_angle)

chosen_angle = 30.0
print(f"Randomly chosen laser angle θl={chosen_angle: .2f}")

print("--- Example target (distance) - (length) pairs")
for x in range(10, 100, 10):
    l = 2*x*np.tan(deg_to_rad(chosen_angle))
    print(f"x: {x: .2f}cm - l: {l: .2f}cm")

print("--- Example laser & speaker (distance) - (height) pairs")
for dlp in range(10, 100, 10):
    hs = dlp * np.tan(deg_to_rad(chosen_angle))
    print(f"dlp: {dlp: .2f}cm - hs: {hs: .2f}cm")

