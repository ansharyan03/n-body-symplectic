#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 18:29:59 2025

@author: carolynshi
"""
import argparse	                 # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import vpython
from vpython import sphere, vector, rate


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

G = 4 * np.pi**2   # AU^3 / (Msun * yr^2)
# G = 6.67430e-8  # km kg^-1 s^2

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
# for plotting the bodies in 2D
def plot_orbits_2D(q_hist, labels=None):
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
# for plotting the bodies in 3D
def plot_orbits_3D(q_hist, labels=None):
    """
    q_hist: (Nsteps, N_bodies, 3)
    """
    Nsteps, N_bodies, _ = q_hist.shape

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(N_bodies):
        x = q_hist[:, i, 0]
        y = q_hist[:, i, 1]
        z = q_hist[:, i, 2]

        if labels is None:
            ax.plot(x, y, z, label=f"Body {i}")
        else:
            ax.plot(x, y, z, label=labels[i])

    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_zlabel("z [AU]")
    ax.set_title("3D N-body Orbits (COM frame)")
    ax.legend()
    plt.show()

#==============================================================
# for plotting animaitons of the bodies in 3D
def animate_orbits(q_hist, labels=None, interval=20):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)

    Nsteps, N_bodies, _ = q_hist.shape
    lines = []

    for i in range(N_bodies):
        (line,) = ax.plot([], [], label=labels[i])
        lines.append(line)

    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_aspect("equal")
    ax.legend()

    def update(frame):
        for i in range(N_bodies):
            x = q_hist[:frame, i, 0]
            y = q_hist[:frame, i, 1]
            lines[i].set_data(x, y)
        return lines

    anim = FuncAnimation(fig, update, frames=Nsteps, interval=20)
    plt.show()

#==============================================================
# for plotting animaitons of the bodies in 3D
def realtime_3d(q_hist):
    Nsteps, N_bodies, _ = q_hist.shape
    bodies = [sphere(pos=vector(*q_hist[0,i]), radius=0.2) for i in range(N_bodies)]

    for t in range(Nsteps):
        rate(200)
        for i in range(N_bodies):
            bodies[i].pos = vector(*q_hist[t,i])
            
#==============================================================            
# input:
#   which: integer array with elements between 1 and 8, with 1: Mercury...8: Neptune
# output:
#   mass: planet mass in kg
#   eps : eccentricity
#   rap : aphelion distance (in km)
#   vorb: aphelion velocity (in km/s)
#   torb: orbital period (in years)
#---------------------------------------------------------------
def get_planetdata(which):
    nplanets             = len(which)
    mass                 = np.array([1.989e30,3.3011e23,4.8675e24,5.972e24,6.41e23,1.89819e27,5.6834e26,8.6813e25,1.02413e26]) 
    eps                  = np.array([0.0,0.205,0.0067,0.0167,0.0934,0.0489,0.0565,0.0457,0.0113])
    rap                  = np.array([0.0,6.9816e10,1.0894e11,1.52139e11,2.49232432e11,8.1662e11,1.5145e12,3.00362e12,4.54567e12])
    vorb                 = np.array([0.0,3.87e4,3.479e4,2.929e4,2.197e4,1.244e4,9.09e3,6.49e3,5.37e3])
    yrorb                = np.array([0.0,0.241,0.615,1.0,1.881,1.1857e1,2.9424e1,8.3749e1,1.6373e2])
    rmass                = np.zeros(nplanets+1)
    reps                 = np.zeros(nplanets+1)
    rrap                 = np.zeros(nplanets+1)
    rvorb                = np.zeros(nplanets+1)
    ryrorb               = np.zeros(nplanets+1)
    rmass [0]            = mass [0]
    rmass [1:nplanets+1] = mass [which]
    reps  [1:nplanets+1] = eps  [which]
    rrap  [1:nplanets+1] = rap  [which]
    rvorb [1:nplanets+1] = vorb [which]
    ryrorb[1:nplanets+1] = yrorb[which]
    return rmass,reps,rrap,rvorb,ryrorb

#==============================================================
def initialize_solar_system(which):
    """
    which: list of indices into get_planetdata (use 0 for Sun, 1..8 for planets)
    Assumes get_planetdata returns arrays in:
      mass [kg], eps, rap [meters], vorb [m/s], torb [years]
    Returns:
      mass_solar (M_sun), q0 (AU), v0 (AU/yr)
    """
    mass_kg, eps, rap_m, vorb_ms, torb = get_planetdata(which)

    # conversion constants
    AU_m = 1.495978707e11        # meters in 1 AU
    M_sun_kg = 1.98847e30       # kg in 1 solar mass
    sec_per_year = 31557600.0   # Julian year seconds

    # m/s -> AU/yr  : multiply by (sec_per_year / AU_m)
    ms_to_AUyr = sec_per_year / AU_m

    # convert
    rap_AU = rap_m / AU_m                   # meters -> AU
    vorb_AUyr = vorb_ms * ms_to_AUyr      # m/s -> AU/yr
    mass_solar_all = mass_kg / M_sun_kg     # array includes Sun at index 0

    N = len(which)
    q0 = np.zeros((N, 3))
    v0 = np.zeros((N, 3))

    # note: get_planetdata was designed so index 0 = Sun (mass at 0),
    # and other entries are mapped accordingly; `which` should list indices
    # that correspond to those expanded arrays (for your calling style
    # you used e.g. which = [0,1,2,3,4])
    for i in range(N):
        idx = which[i]   # integer index into the arrays returned by get_planetdata

        if idx == 0:
            # put Sun at origin with zero vel (we will shift to COM after)
            q0[i] = np.array([0.0, 0.0, 0.0])
            v0[i] = np.array([0.0, 0.0, 0.0])
        else:
            # place planet at aphelion on +x axis
            q0[i] = np.array([rap_AU[idx], 0.0, 0.0])
            # give tangential velocity in +y direction (aphelion speed)
            v0[i] = np.array([0.0, vorb_AUyr[idx], 0.0])

    # return masses only for the requested indices
    return mass_solar_all[which], q0, v0




#==============================================================
def plot(prob):
    # --- Define masses ---
    M_sun   = 1.0
    M_mercury = 1.66e-7
    M_venus = 2.45e-6
    M_earth = 3.0e-6
    M_mars = 3.21e-7
    M_jupiter = 9.54e-4
    M_saturn = 2.86e-4
    M_uranus = 4.36e-5
    M_neptune = 5.15e-5
    M_pluto = 6.54e-9
    m = np.array([M_sun, M_mercury, M_venus, M_earth, M_mars, M_jupiter, M_saturn, M_uranus, M_neptune, M_pluto])


    # --- Define raw initial positions (heliocentric, for example) ---
    q = np.array([
        [0.0, 0.0, 0.0],   # Sun (arbitrary frame)
        [0.39, 0.0, 0.0],   # Mercury
        [0.72, 0.0, 0.0],   # Venus
        [1.0, 0.0, 0.0],   # Earth
        [1.52, 0.0, 0.0],    # Mars
        [5.20, 0.0, 0.0],    # Jupiter
        [9.54, 0.0, 0.0],    # Saturn
        [19.22, 0.0, 0.0],    # Uranus
        [30.06, 0.0, 0.0],    # Neptune
        [39.5, 0.0, 0.0]    # Pluto
    ])

    # --- Define raw initial velocities ---
    v = np.array([
        [0.0, 0.0, 0.0],   # Sun velocity (will be corrected)
        [0.0, 10.02, 0.0],   # Mercury
        [0.0, 7.39, 0.0],   # Venus
        [0.0, 6.283, 0.0],   # Earth
        [0.0, 5.099, 0.0],    # Mars
        [0.0, 5.2, 0.0],    # Jupiter
        [0.0, 2.754, 0.0],    # Saturn
        [0.0, 1.438, 0.0],    # Uranus
        [0.0, 1.147, 0.0],    # Neptune
        [0.0, 0.999, 0.0]    # Pluto
    ])
    

    # --- shift entire system into COM frame (generic for ANY N) ---
    q, v = shift_to_com_frame(q, v, m)


    
    # --- Integration parameters ---
    dt = 0.001
    Nsteps = 100000

    q_hist = []
    E_hist = []
    L_hist = []
    
    if (prob == 'euler'):
            
        # --- Integrate system ---
        for _ in range(Nsteps):
    
            q, v = implicit_midpoint_step(q, v, m, dt, G)
    
            # for plotting ONLY, not for integration
            Rcom = compute_com(q, m)
            q_rel = q - Rcom
    
            q_hist.append(q_rel.copy())
            E_hist.append(total_energy(q, v, m, G))
            L_hist.append(total_angular_momentum(q, v, m))
    
        q_hist = np.array(q_hist)
        
    elif (prob == 'stormer'):
        
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
        
    elif (prob == 'implicit'):
        
        # --- Integrate system ---
        for _ in range(Nsteps):
    
            q, v = implicit_midpoint_step(q, v, m, dt, G)
    
            # for plotting ONLY, not for integration
            Rcom = compute_com(q, m)
            q_rel = q - Rcom
    
            q_hist.append(q_rel.copy())
            E_hist.append(total_energy(q, v, m, G))
            L_hist.append(total_angular_momentum(q, v, m))
    
        q_hist = np.array(q_hist)

    # --- Plot orbits ---
    plt.ion()
    # realtime_3d(q_hist)
    # animate_orbits(q_hist, labels=["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"])
    plot_orbits_3D(q_hist, labels=["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"])
    # labels=["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
    
    
    # --- Plot Energy conservation ---
    E_hist = np.array(E_hist)
    plt.figure()
    plt.plot((E_hist - E_hist[0]) / E_hist[0])
    plt.xlabel("Step")
    plt.ylabel("(E - E0)/E0")
    plt.title("Fractional Energy Error")
    plt.grid(True)
    plt.show()
    
    # --- Plot Angular Momentum conservation ---
    L_hist = np.array(L_hist)
    L0 = L_hist[0]
    frac_L = np.linalg.norm(L_hist - L0, axis=1) / np.linalg.norm(L0)
    
    plt.figure()
    plt.plot(frac_L)
    plt.xlabel("Step")
    plt.ylabel("|L - L0| / |L0|")
    plt.title("Fractional Angular Momentum Error")
    plt.grid(True)
    plt.show()


# ==========================================================
def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("prob",type=str,default='euler',
                        help="stepping function:\n"
                             "   euler: Symplectic Euler step\n"
                             "   stormer  : stormer-verlet\n"
                             "   implicit  : Implicit midpoint (2nd order)\n")
    
    args   = parser.parse_args()
    prob   = args.prob
    
    plot(prob)
    
    
main()
