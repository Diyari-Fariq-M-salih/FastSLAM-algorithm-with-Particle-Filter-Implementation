import numpy as np

def compute_potential(map_est, x, y):
    local = map_est[x-3:x+3, y-3:y+3]
    occ = np.mean(local > 0.6)     # repulsive
    unk = np.mean((local < -0.2))  # attractive
    return 5*occ - 3*unk

def potential_field_planner(map_est, x, y, theta, step=3):
    best_U = 1e9
    best_dx, best_dtheta = 0, 0

    for dtheta in [-20, -10, 0, 10, 20]:
        theta_new = theta + dtheta
        dx = step
        x_new = int(x + np.sin(theta_new/180*np.pi) * dx)
        y_new = int(y + np.cos(theta_new/180*np.pi) * dx)

        U = compute_potential(map_est, x_new, y_new)
        if U < best_U:
            best_U = U
            best_dx = dx
            best_dtheta = dtheta

    return best_dx, best_dtheta
