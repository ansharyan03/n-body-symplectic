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
def integrate_step(q, v, m, dt, G):
    """Perform one integration step using chosen integrator."""
    # symplectic euler 
    a = acceleration(q, m, G)
    v_new = v + 1 * dt * a           # take a half step in dq
    q_new = q + dt * v_new             
    a_new = acceleration(q_new, m, G)   # update dp in a full step
    

    return q_new, v_new

#==============================================================
# converting units to meet G = 1
# test case

#==============================================================
def main():
    # Testing Sun & Earth case (solar units)
    G = 4 * np.pi ** 2
    M_sun = 1.0
    M_earth = 3.0e-6
    m = np.array([M_sun, M_earth])

    # Initial positions and velocities (center-of-mass frame)
    q_earth = np.array([1.0, 0.0, 0.0])             # 1 AU
    q_sun = -(M_earth / M_sun) * q_earth            # finding CoM pf sun & earth,
                                                    # then find position of sun
                                                    # so COM = 0
    v_earth = np.array([0.0, 6, 0.0])               # 1 AU circular orbit, 1 yr period
    v_sun = -(M_earth / M_sun) * v_earth            # COM stationary (at 0)

    q = np.array([q_sun, q_earth])
    v = np.array([v_sun, v_earth])

    # Time step 
    dt = 0.0001   
    Nsteps = 100000    # Energy error oscillation period is proportional to step size 

    # --- Storage arrays ---
    q_hist = []
    E_hist = []
    L_hist = []
    
    # --- Integrate ---
    for _ in range(Nsteps):
        q, v = integrate_step(q, v, m, dt, G)
        # Center-of-mass correction each step
        r_CoM = np.average(q, axis=0, weights=m)
        q_rel = q - r_CoM
        q_hist.append(q_rel.copy())
        E_hist.append(total_energy(q, v, m, G))
        L_hist.append(total_angular_momentum(q, v, m))

    q_hist = np.array(q_hist)

    # --- Plot orbits ---
    plt.figure()
    plt.plot(q_hist[:, 0, 0], q_hist[:, 0, 1], label="Sun", marker ='o', markersize=10,)
    plt.plot(q_hist[:, 1, 0], q_hist[:, 1, 1], label="Earth")
    plt.axis("equal")
    plt.legend()
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.title("Sun–Earth Orbits (symplectic integrator, G=1)")
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
