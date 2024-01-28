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
# Define a decision vector ww_eq = [beta_eq, delta_eq, Fx_eq]
# ww_eq = np.zeros(3)
ww_eq1 = [0.1, 0.05, 50]
ww_eq2 = [0.1, 0.05, 50]

# Definition of the radius RR and linear (tangent) velocity VV
RR1 = 15
VV1 = 12
# psi_dot found applying trigonometry and basic mechanics law (psi_dot = w = v/r)
WW1 = VV1/RR1

# First equilibrium
print("Computing equilibria 1 [V1 = {}, R1 = {}]".format(VV1,RR1))
ww_eq1 = equilibrium_finder.eq_finder(WW1, VV1, ww_eq1)


RR2 = 16
VV2 = 13
#psi_dot = w = v/r
WW2 = VV2/RR2

# Second equilibrium
print("###############################################################################################")
print("Computing equilibria 2 [V2 = {}, R2 = {}]".format(VV2,RR2))
ww_eq2 = equilibrium_finder.eq_finder(WW2, VV2, ww_eq2)



##################################################################
##################################################################
### UNFEASIBLE TRAJECTORY BETWEEN EQUILIBRIA (STEP)

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
    


steps = np.arange(TT-1)
plt.figure('Velocity')
plt.xlabel('steps')
plt.ylabel('V_ref')
plt.grid()
plt.plot(steps, xx_ref[3,:TT-1], color='red', label='V_ref')
plt.legend(loc="upper right")
plt.show()

plt.figure('Beta')
plt.xlabel('steps')
plt.ylabel('beta_ref')
plt.grid()
plt.plot(steps, xx_ref[4,:TT-1], color='green', label='beta_ref')
plt.legend(loc="upper right")
plt.show()

plt.figure('Psi_dot')
plt.xlabel('steps')
plt.ylabel('psi_dot_ref')
plt.grid()
plt.plot(steps, xx_ref[5,:TT-1], color='blue', label='psi_dot_ref')
plt.legend(loc="upper right")
plt.show()

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

# Define a smooth initial guess
# Does it work fine even if the inital guess is not a trajectory?? No

# Compute a set of equilibrium between the inital and final equilibrium
# Fix the evolution of V and psi_dot as linear from VV1,WW1 to VV2,WW2
# Compute beta, delta and Fx such that each point is an equilibrium

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

offset = [0, 0, 0, 0, 0, 0]
xx_init, uu_init = LQR.LQR_trajectory(quasi_static_xx, quasi_static_uu, offset)

#xx, uu = gom.optimal_trajectory(xx_ref, uu_ref, xx_init, uu_init)
xx_opt, uu_opt = nom.optimal_trajectory(xx_ref, uu_ref, xx_init, uu_init)


# the comparison between the reference trajectory and the optimal trajectory
#########################################
# State evolution
steps = np.arange(TT-1)
plt.figure('Velocity')
plt.xlabel('steps')
plt.ylabel('V_opt')
plt.grid()
plt.plot(steps, xx_ref[3,:TT-1], color='red',label='V_ref',linestyle='dashed')
plt.plot(steps, xx_opt[3,:TT-1,-1], color='red',label='V_opt')
plt.legend(loc="upper right")
plt.show()

plt.figure('Beta')
plt.xlabel('steps')
plt.ylabel('beta_opt')
plt.grid()
plt.plot(steps, xx_ref[4,:TT-1], color='green',label='beta_ref',linestyle='dashed')
plt.plot(steps, xx_opt[4,:TT-1,-1], color='green',label='beta_opt')
plt.legend(loc="upper right")
plt.show()

plt.figure('Psi_dot')
plt.xlabel('steps')
plt.ylabel('psi_dot_opt')
plt.grid()
plt.plot(steps, xx_ref[5,:TT-1], color='blue',label='psi_dot_ref',linestyle='dashed')
plt.plot(steps, xx_opt[5,:TT-1,-1], color='blue',label='psi_dot_opt')
plt.legend(loc="upper right")
plt.show()

#########################################
# Input evolution

plt.figure('delta')
plt.xlabel('steps')
plt.ylabel('delta_opt')
plt.grid()
plt.plot(steps, uu_ref[0,:TT-1], color='red',linestyle='dashed',label='delta_ref')
plt.plot(steps, uu_opt[0,:TT-1,-1], color='red',label='delta_opt')
plt.legend(loc="upper right")
plt.show()

plt.figure('F_x_opt')
plt.xlabel('steps')
plt.ylabel('F_x')
plt.grid()
plt.plot(steps, uu_ref[1,:TT-1], color='blue',linestyle='dashed',label='F_x_ref')
plt.plot(steps, uu_opt[1,:TT-1,-1], color='blue',label='F_x_opt')
plt.legend(loc="upper right")
plt.show()

##################################################################
##################################################################
### TRAJECTORY TRACKING VIA LQR (TASK 3)

offset = [0, 0, 0, 0.2, 0, 0]
xx_LQR, uu_LQR = LQR.LQR_trajectory(xx_opt[:,:,-1], uu_opt[:,:,-1],offset)

steps = np.arange(TT-1)
plt.figure('Velocity')
plt.xlabel('steps')
plt.ylabel('V_LQR')
plt.grid()
plt.plot(steps, xx_opt[3,:TT-1,-1], color='red',label='V_opt',linestyle='dashed')
plt.plot(steps, xx_LQR[3,:TT-1], color='red',label='V_LQR')
plt.legend(loc="upper right")
plt.show()

plt.figure('Beta')
plt.xlabel('steps')
plt.ylabel('beta_LQR')
plt.grid()
plt.plot(steps, xx_opt[4,:TT-1,-1], color='green',label='beta_opt',linestyle='dashed')
plt.plot(steps, xx_LQR[4,:TT-1], color='green',label='beta_LQR')
plt.legend(loc="upper right")
plt.show()

plt.figure('Psi_dot')
plt.xlabel('steps')
plt.ylabel('psi_dot_LQR')
plt.grid()
plt.plot(steps, xx_opt[5,:TT-1,-1], color='blue',label='psi_dot_opt',linestyle='dashed')
plt.plot(steps, xx_LQR[5,:TT-1], color='blue',label='psi_dot_LQR')
plt.legend(loc="upper right")
plt.show()

#########################################
# Input evolution

plt.figure('delta')
plt.xlabel('steps')
plt.ylabel('delta_LQR')
plt.grid()
plt.plot(steps, uu_opt[0,:TT-1,-1], color='red',linestyle='dashed',label='delta_opt')
plt.plot(steps, uu_LQR[0,:TT-1], color='red',label='delta_LQR')
plt.legend(loc="upper right")
plt.show()

plt.figure('F_x')
plt.xlabel('steps')
plt.ylabel('F_x_LQR')
plt.grid()
plt.plot(steps, uu_opt[1,:TT-1,-1], color='blue',linestyle='dashed',label='F_x_opt')
plt.plot(steps, uu_LQR[1,:TT-1], color='blue',label='F_x_LQR')
plt.legend(loc="upper right")
plt.show()

##################################################################
##################################################################
### ANIMATION SECTION
import animation as anim

start_animation = input('press a button to start animation: ')

anim.drawTraj(xx_opt[:,:,-1])
kk = 0
start_time = time.time()
while kk < TT-1:
    anim.moveVehicle(xx_opt[:,kk,-1],uu_opt[:,kk,-1], delta_t*kk)
    anim.window.update()
    kk += 1

    time.sleep(0.005)

# Avoid animation to close automatically

anim.window.mainloop()
