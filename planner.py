import numpy as np

def local_potential(map_est, x, y):
    x = int(x); y = int(y)
    if x < 3 or x >= 497 or y < 3 or y >= 497:
        return 9999

    local = map_est[x-3:x+3, y-3:y+3]
    occ = np.mean(local > 1.0)
    unk = np.mean(local < -1.0)

    return 5*occ - 3*unk   # repulsive from occ, attractive to unknown

def potential_field_planner(map_est, x, y, theta, step=3):
    bestU = 999999
    best_dx, best_dtheta = 0, 0

    for dtheta in [-30, -15, 0, 15, 30]:
        theta_new = theta + dtheta
        dx = step

        nx = x + np.sin(theta_new/180*np.pi) * dx
        ny = y + np.cos(theta_new/180*np.pi) * dx

        U = local_potential(map_est, nx, ny)
        if U < bestU:
            bestU = U
            best_dx = dx
            best_dtheta = dtheta

    return best_dx, best_dtheta
