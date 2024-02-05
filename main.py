# OPTCON exam project
# Gualandi, Piraccini, Bosi
# Optimal control of a vehicle


import numpy as np
import matplotlib.pyplot as plt
import dynamics as dyn
import equilibrium_finder
import gradient_optcon_method as gom
import newton_optcon_method as nom
import LQR_tracking as LQR
import time
import plotter


# Define step-size and time horizon for discretization
delta_t = dyn.dt
TT = int(dyn.tf/delta_t)
n_x = dyn.n_x
n_u = dyn.n_u

##################################################################
##################################################################
### EQUILIBRIA

print("###############################################################################################")
print("###############################################################################################")
print("### EQUILIBRIA")

# Equilibrium trajectories
# Define two decision vector ww_eq = [beta_eq, delta_eq, Fx_eq] with random values
ww_eq1 = [0.1, 0.05, 50]
ww_eq2 = [0.1, 0.05, 50]

# Definition of the radius RR and linear (tangent) velocity VV
RR1 = 15
VV1 = 12
# psi_dot found applying trigonometry and basic mechanics law (psi_dot = w = v/r)
WW1 = VV1/RR1

# First equilibrium
print("Computing equilibria 1 [V1 = {}, R1 = {}]".format(VV1,RR1))
print("Initial arbitrary ww_eq [beta, delta, Fx] = {}".format(ww_eq1))
print("Running Newton's method for root finding ...")

ww_eq1 = equilibrium_finder.eq_finder(WW1, VV1, ww_eq1)

print("Final ww_eq [beta, delta, Fx] = {}".format(ww_eq1))
delta_x = dyn.dynamics([0,0,0,VV1,ww_eq1[0],WW1],[ww_eq1[1],ww_eq1[2]])[1]
print("Check dynamics ([0,0,0] ideal): {}".format(delta_x[3:]))

RR2 = -16
VV2 = 13
WW2 = VV2/RR2

# Second equilibrium
print("###############################################################################################")
print("Computing equilibria 2 [V2 = {}, R2 = {}]".format(VV2,RR2))
print("Initial arbitrary ww_eq [beta, delta, Fx] = {}".format(ww_eq2))
print("Running Newton's method for root finding ...")

ww_eq2 = equilibrium_finder.eq_finder(WW2, VV2, ww_eq2)

print("Final ww_eq [beta, delta, Fx] = {}".format(ww_eq2))
delta_x = dyn.dynamics([0,0,0,VV2,ww_eq2[0],WW2],[ww_eq2[1],ww_eq2[2]])[1]
print("Check dynamics ([0,0,0] ideal): {}".format(delta_x[3:]))


##################################################################
##################################################################
### UNFEASIBLE TRAJECTORY BETWEEN EQUILIBRIA (STEP)

# Initially compute a step reference.
# If smooth_reference, overwrite the values in the middle with a smooth transient
smooth_reference = True

xx_ref = np.zeros((n_x, TT))
uu_ref = np.zeros((n_u, TT))


# Define input reference (step function)
for kk in range(int(TT/2)): uu_ref[:,kk] = [ww_eq1[1], ww_eq1[2]]
for kk in range(int(TT/2), int(TT-1)): uu_ref[:,kk] = [ww_eq2[1], ww_eq2[2]]


# Define initial state variables (arbitrary)
xx_init = [0, -RR1, -ww_eq1[0], VV1, ww_eq1[0], WW1]
xx_ref[:,0] = xx_init

# Loop simulation
for kk in range(int(TT/2)):
    xx_ref[:,kk+1] = dyn.dynamics(xx_ref[:,kk],uu_ref[:,kk])[0]

# Define TT/2 state variables
xx_init = [xx_ref[0,int(TT/2-1)], xx_ref[1,int(TT/2-1)], xx_ref[2,int(TT/2-1)], VV2, ww_eq2[0], WW2]
xx_ref[:,int(TT/2)] = xx_init

# Loop simulation
for kk in range(int(TT/2), TT-1):
    xx_ref[:,kk+1] = dyn.dynamics(xx_ref[:,kk],uu_ref[:,kk])[0]

if smooth_reference:
    ### UNFEASIBLE TRAJECTORY BETWEEN EQUILIBRIA (SMOOTH)
    V_transient = TT/10
    delta_V = VV2-VV1
    V_slope = delta_V/V_transient

    beta_transient = TT/10
    delta_beta = ww_eq2[0]-ww_eq1[0]
    beta_slope = delta_beta/beta_transient

    psi_dot_transient = TT/10
    delta_psi_dot = WW2-WW1
    psi_dot_slope = delta_psi_dot/psi_dot_transient

    delta_transient = TT/10
    delta_delta = ww_eq2[1]-ww_eq1[1]
    delta_slope = delta_delta/delta_transient

    Fx_transient = TT/10
    delta_Fx =  ww_eq2[2]-ww_eq1[2]
    Fx_slope = delta_Fx/Fx_transient

    for kk in range(int(TT/2-V_transient/2), int(TT/2+V_transient/2)):
        xx_ref[3,kk] = VV1 + V_slope*(kk-(TT/2-V_transient/2))

    for kk in range(int(TT/2-beta_transient/2), int(TT/2+beta_transient/2)):
        xx_ref[4,kk] = ww_eq1[0] + beta_slope*(kk-(TT/2-beta_transient/2))

    for kk in range(int(TT/2-psi_dot_transient/2), int(TT/2+psi_dot_transient/2)):
        xx_ref[5,kk] = WW1 + psi_dot_slope*(kk-(TT/2-psi_dot_transient/2))

    for kk in range(int(TT/2-delta_transient/2), int(TT/2+delta_transient/2)):
        uu_ref[0,kk] = ww_eq1[1] + delta_slope*(kk-(TT/2-delta_transient/2))

    for kk in range(int(TT/2-Fx_transient/2), int(TT/2+Fx_transient/2)):
        uu_ref[1,kk] = ww_eq1[2] + Fx_slope*(kk-(TT/2-Fx_transient/2))

# Plot the reference values
print("###############################################################################################")
print(f'smooth reference: {smooth_reference}')
print('Plotting reference...')
#plotter.plot_ref(xx_ref, uu_ref, TT)

##################################################################
##################################################################
### FEASIBLE TRAJECTORY BETWEEN EQUILIBRIA (TASK 1 and 2)
    
print("###############################################################################################")
print("###############################################################################################")
print("### FEASIBLE TRAJECTORY BETWEEN EQUILIBRIA")

# Initial guess
xx_init = np.zeros((n_x, TT))
uu_init = np.zeros((n_u, TT))

xx_init[:,0] = [0, -RR1, -ww_eq1[0], VV1, ww_eq1[0], WW1]

# Initial guess equal to the first equilibrium for all time steps
for kk in range(int(TT-1)): uu_init[:,kk] = [ww_eq1[1], ww_eq1[2]]
for kk in range(int(TT-1)): xx_init[:,kk+1] = xx_init[:,0]

'''
# Define a smooth initial guess
# Does it work fine even if the inital guess is not a trajectory?? ->No

# Compute a set of equilibrium between the inital and final equilibrium
# Fix the evolution of V and psi_dot as linear from VV1,WW1 to VV2,WW2
# Compute beta, delta and Fx such that each point is an equilibrium
print('computing quasi-static trajectory... (takes 90 seconds)')

transient = TT/5
delta_V = VV2-VV1
V_slope = delta_V/transient

delta_psi_dot = WW2-WW1
psi_dot_slope = delta_psi_dot/transient

quasi_static_xx = np.zeros((n_x, TT))
quasi_static_uu = np.zeros((n_u, TT))

for kk in range(0,int(TT/2-transient/2)):
    quasi_static_xx[3,kk] = VV1
    quasi_static_xx[4,kk] = ww_eq1[0]
    quasi_static_xx[5,kk] = WW1

    quasi_static_uu[0,kk] = ww_eq1[1]
    quasi_static_uu[1,kk] = ww_eq1[2]

for kk in range(int(TT/2-transient/2), int(TT/2+transient/2)):
    quasi_static_xx[3,kk] = VV1 + V_slope*(kk-(TT/2-transient/2))
    quasi_static_xx[5,kk] = WW1 + psi_dot_slope*(kk-(TT/2-transient/2))
    ww_eq = [0.1, 0.05, 50]
    ww_eq = equilibrium_finder.eq_finder(quasi_static_xx[5,kk], quasi_static_xx[3,kk], ww_eq)
    quasi_static_xx[4,kk] = ww_eq[0]
    quasi_static_uu[0,kk] = ww_eq[1]
    quasi_static_uu[1,kk] = ww_eq[2]

for kk in range(int(TT/2+transient/2), TT-1):
    quasi_static_xx[3,kk] = VV2
    quasi_static_xx[4,kk] = ww_eq2[0]
    quasi_static_xx[5,kk] = WW2

    quasi_static_uu[0,kk] = ww_eq2[1]
    quasi_static_uu[1,kk] = ww_eq2[2]

# The offset has a meaning only in task 3, so now we set it to zero
offset = [0, 0, 0, 0, 0, 0]
xx_init, uu_init = LQR.LQR_trajectory(quasi_static_xx, quasi_static_uu, offset)
'''
#plotter.plot_init_guess(xx_init, uu_init, TT)

print("###############################################################################################")
#xx_opt, uu_opt = gom.optimal_trajectory(xx_ref, uu_ref, xx_init, uu_init)
print('Running newton method for optimal control...')
xx_opt, uu_opt, last_iter = nom.optimal_trajectory(xx_ref, uu_ref, xx_init, uu_init)

# Plot the comparison between the reference trajectory and the optimal trajectory
print('Plotting optimal trajectory...')
plotter.plot_opt_ref(xx_ref, uu_ref, xx_opt[:,:,last_iter], uu_opt[:,:,last_iter], TT, last_iter)
plotter.plot_opt_iters(xx_ref, uu_ref, xx_opt, uu_opt, TT, last_iter)

##################################################################
##################################################################
### TRAJECTORY TRACKING VIA LQR (TASK 3)

print("###############################################################################################")
print("###############################################################################################")
print("### OPTIMAL TRAJECTORY TRACKING")

# Now the offset has a meaning. The feedback input is able to compensate initial offset.
offset = np.array([0, 0, 0, 0.2, 0.09, 0.09])
alpha = [-1, 1, 2]
xx_LQR = np.zeros((n_x,TT,len(alpha)))
uu_LQR = np.zeros((n_u,TT,len(alpha)))

for kk in range(len(alpha)):
    temp_offset = offset*alpha[kk]
    print(f'Initial offset: {temp_offset}')
    print('Applying LRQ...')
    xx_LQR[:,:,kk], uu_LQR[:,:,kk] = LQR.LQR_trajectory(xx_opt[:,:,last_iter], uu_opt[:,:,last_iter], temp_offset)

# Plot the comparison between optimal trajectory and LQR feedback trajectory (with initial offset)
print('Plotting LQR result...')
plotter.plot_LQR_opt(xx_opt[:,:,last_iter], uu_opt[:,:,last_iter], xx_LQR, uu_LQR, TT, alpha)
plotter.plot_LQR_error(xx_opt[:,:,last_iter], uu_opt[:,:,last_iter], xx_LQR, uu_LQR, TT, alpha)


##################################################################
##################################################################
### ANIMATION SECTION

print("###############################################################################################")
print("###############################################################################################")
print("### ANIMATION")

import animation as anim

start_animation = input('press a button to start animation: ')

anim.drawTraj(xx_opt[:,:,last_iter])
kk = 0
#start_time = time.time()
while kk < TT-1:
    anim.moveVehicle(xx_opt[:,kk,last_iter],uu_opt[:,kk,last_iter], delta_t*kk)
    anim.window.update()
    kk += 1
    # spleep is useful to slow down tha animation.
    # (without the sleep the animation ends in 2 seconds)
    time.sleep(0.005)

# Avoid animation to close automatically
anim.window.mainloop()
