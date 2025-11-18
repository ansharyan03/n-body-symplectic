#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 18:29:59 2025

@author: carolynshi
"""

import numpy as np
import matplotlib.pyplot as plt
#==============================================================
# q = position vector
# p = momentum vector   -> v = p/m
# m = mass array
# If we use AU< Solar Mass, yr:
# mass -> ito. 1 Solar Mass (1.989e33 g)
# length -> ito. 1 AU (1.496e11 m)
# time -> ito. 1 year (3.154e7 seconds)
# G = 4pi^2 (AU^3) / (SolMass * yr^2)
# unit conversion done in main function (or smth like that)
#==============================================================



#==============================================================
def acceleration(q, m, G):
    """Compute accelerations on all N bodies."""
    N = len(m)
    a = np.zeros_like(q)
    for i in range(N):
        for j in range(i+1, N):
            dq = q[j] - q[i]
            q2 = np.dot(dq, dq)     # separation distance sqaured. 
            q3 = q2 * np.sqrt(q2)   # separation distance cubed.
            fij = G * dq / q3       # Newton's gravitational force 
                                    # between i-th and j-th body OVER i-th mass.
            a[i] += m[j] * fij      # 
            a[j] -= m[i] * fij   # Newton's 3rd law
    return a

#==============================================================



#==============================================================
def total_energy(q, v, m, G):
    """Return total energy (kinetic + potential).
        Ideally we want total energy to be constant.
        Can append results into an array and plot 
        (E(t) - E(0)) / E(0) vs. time lastly. 
        Should be arounf 0."""
    T = 0.5 * np.sum(m[:, None] * v**2)     # Calculates Summation of mi*v^2 / (2mi)
    U = 0.0
    N = len(m)
    for i in range(N):
        for j in range(i+1, N):
            dq = np.linalg.norm(q[j] - q[i])    # calculates ||qi - qj||
            U -= G * m[i] * m[j] / dq
    return T + U

#==============================================================



#==============================================================
def total_angular_momentum(q, v, m):
    """Returns angular momentum.
        Ideally we want this to be constant too.
        Can append results into an array and plot 
        (L(t) - L(0)) / L(0) vs. time. 
        Trned should be arounf 0."""
    L = np.zeros(3)
    for i in range(len(m)):
        L += m[i] * np.cross(q[i], v[i])
    return L

#==============================================================
def symplectic_euler_step(q, v, m, dt, G):
    """Perform one integration step using symplectic euler integrator."""
    # symplectic euler 
    a = acceleration(q, m, G)
    v_new = v + 1 * dt * a           # take a full step in dq
    q_new = q + dt * v_new           # update position in a full step  
    a_new = acceleration(q_new, m, G)   # update dp in a full step
    

    return q_new, v_new

#==============================================================
def stormer_verlet_step(q, v, m, dt, G):
    """2nd order symplectic integrator:
        Explicit
        No initialization requirement (can just use q and v)
        Requires separable Hamiltonian H(p, q) = T(p) + V(q)
        requires smaller dt during close encounters"""
    a = acceleration(q, m, G)
    v_half = v + 0.5 * dt * a           # take a half step in dq
    q_new = q + v_half * dt             # update a full step in position
    a_new = acceleration(q_new, m, G)   # update acceleration (dp) via new position
    v_new = v_half + 0.5 * dt * a_new   # update dq to a full step 
    
    return q_new, v_new

#==============================================================
def implicit_midpoint_step(q, v, m, dt, G, tol=1e-12, max_iter=20):
    """Implicit midpoint integrator (implicit rk2):
       Requires initial conditions for q and v to start procedure
       Works with non-separable Hamiltonian (though not very relevant to this project)
       Needs one nonlinear solve at first step
       Less sensitive to extreme close encounters. Can work with large dt"""
    # initial guesses (explicit Euler)
    q_new = q + dt * v
    v_new = v + dt * acceleration(q, m, G)

    for _ in range(max_iter):

        q_mid = 0.5 * (q + q_new)
        v_mid = 0.5 * (v + v_new)

        a_mid = acceleration(q_mid, m, G)

        # new updates from midpoint
        q_new_next = q + dt * v_mid
        v_new_next = v + dt * a_mid

        # check convergence
        if (np.linalg.norm(q_new_next - q_new) < tol and
            np.linalg.norm(v_new_next - v_new) < tol):
            q_new, v_new = q_new_next, v_new_next
            break

        q_new, v_new = q_new_next, v_new_next

    return q_new, v_new
# Soles the implicit equaitons via iteration until convergence. This preserves symplecticity
# up to given tolerance.

#==============================================================
# adjusting CoM of system so we plot them in the center
def compute_com(q, m):
    return np.sum(q * m[:,None], axis=0) / np.sum(m)

def compute_com_velocity(v, m):
    return np.sum(v * m[:,None], axis=0) / np.sum(m)

def shift_to_com_frame(q, v, m):
    Rcom = compute_com(q, m)
    Vcom = compute_com_velocity(v, m)
    return q - Rcom, v - Vcom

#==============================================================
# # for plotting the bodies
def plot_orbits(q_hist, labels=None):
    """
    q_hist: array of shape (Nsteps, N_bodies, 3)
            positions already shifted to COM for plotting
    labels: optional list of names (len=N_bodies)
    """
    Nsteps, N_bodies, _ = q_hist.shape

    plt.figure(figsize=(7,7))

    for i in range(N_bodies):
        x = q_hist[:, i, 0]
        y = q_hist[:, i, 1]

        if labels is not None:
            plt.plot(x, y, label=labels[i])
        else:
            plt.plot(x, y, label=f"Body {i}")

    plt.axis("equal")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title("N-body Orbits (COM frame)")
    plt.legend()
    plt.grid(True)
    plt.show()




#==============================================================
def main():

    # --- Physical units ---
    G = 4 * np.pi**2   # AU^3 / (Msun * yr^2)

    # --- Define masses ---
    M_sun   = 1.0
    M_earth = 3.0e-6
    M_jupiter = 9.54e-4
    m = np.array([M_sun, M_earth, M_jupiter])


    # --- Define raw initial positions (heliocentric, for example) ---
    q = np.array([
        [0.0, 0.0, 0.0],   # Sun (arbitrary frame)
        [1.0, 0.0, 0.0],   # Earth
        [5, 0.0, 0.0],
    ])

    # --- Define raw initial velocities ---
    v = np.array([
        [0.0, 0.0, 0.0],   # Sun velocity (will be corrected)
        [0.0, 6.283, 0.0], # Earth ~ circular orbit
        [0.0, 3.5, 0.0]
    ])

    # --- shift entire system into COM frame (generic for ANY N) ---
    q, v = shift_to_com_frame(q, v, m)

    # --- Integration parameters ---
    dt = 0.00001
    Nsteps = 1000000

    q_hist = []
    E_hist = []
    L_hist = []

    # --- Integrate system ---
    for _ in range(Nsteps):

        q, v = stormer_verlet_step(q, v, m, dt, G)

        # for plotting ONLY, not for integration
        Rcom = compute_com(q, m)
        q_rel = q - Rcom

        q_hist.append(q_rel.copy())
        E_hist.append(total_energy(q, v, m, G))
        L_hist.append(total_angular_momentum(q, v, m))

    q_hist = np.array(q_hist)

    # --- Plot orbits ---
    # plot_orbits(q_hist, labels=["Sun", "Earth"])

    
    plt.figure()
    plt.plot(q_hist[:, 0, 0], q_hist[:, 0, 1], label="Sun", marker ='o', markersize=10,)
    plt.plot(q_hist[:, 1, 0], q_hist[:, 1, 1], label="Earth")
    plt.plot(q_hist[:, 2, 0], q_hist[:, 2, 1], label="Jupiter")
    plt.axis("equal")
    plt.legend()
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title("Orbits (symplectic integrator, G=1)")
    plt.show()

    # --- Plot energy evolution ---
    E_hist = np.array(E_hist)
    plt.figure()
    plt.plot((E_hist - E_hist[0]) / E_hist[0])
    plt.title("Fractional Energy Error")
    plt.xlabel("Step")
    plt.ylabel("(E - E₀)/E₀")
    plt.show()
    
    # --- Plot angular momentum evolution ---
    L_hist = np.array(L_hist)
    plt.figure()
    plt.plot((L_hist - L_hist[0]) / L_hist[0])
    plt.title("Fractional Angular Momentum Error")
    plt.xlabel("Step")
    plt.ylabel("(L - L₀)/L₀")
    plt.show()


# ==========================================================
main()
