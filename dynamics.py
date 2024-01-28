import numpy as np

#dt = 0.001
dt = 0.01
tf = 20
n_x = 6
n_u = 2

# Declaring mechanical parameters of the vehicle
m = 1480
I_z = 1950
a = 1.421
b = 1.029
mu = 1
g = 9.81

# Computing vertical forces on the wheels (independent from xx)
F_z_f = m*g*b/(a+b)
F_z_r = m*g*a/(a+b)
#print('F_z_f = {}'.format(F_z_f))
#print('F_z_r = {}'.format(F_z_r))

def dynamics(xx, uu):
    # 'Extract' state variables. Easier to read the code
    #x = xx[0]
    #y = xx[1]
    psi = xx[2]
    V = xx[3]
    beta = xx[4]
    psi_dot = xx[5]

    # 'Extract' input variables. Easier to read the code
    delta = uu[0]
    F_x = uu[1]

    # Compute front and rear sideslip angles
    beta_f = delta-(V*np.sin(beta)+a*psi_dot)/(V*np.cos(beta))
    beta_r = -(V*np.sin(beta)-b*psi_dot)/(V*np.cos(beta))
    #print('beta_f = {}'.format(beta_f))
    #print('beta_r = {}'.format(beta_r))

    # compute lateral forces
    F_y_f = mu*F_z_f*beta_f
    F_y_r = mu*F_z_r*beta_r
    #print('F_y_f = {}'.format(F_y_f))
    #print('F_y_r = {}'.format(F_y_r))

    # Compute the velocities
    xx_dot = np.zeros(len(xx))
    xx_dot[0] = V*np.cos(beta)*np.cos(psi)-V*np.sin(beta)*np.sin(psi)
    xx_dot[1] = V*np.cos(beta)*np.sin(psi)+V*np.sin(beta)*np.cos(psi)
    xx_dot[2] = xx[5]
    xx_dot[3] = 1/m*(F_y_r*np.sin(beta)+F_x*np.cos(beta-delta)+F_y_f*np.sin(beta-delta))
    xx_dot[4] = (F_y_r*np.cos(beta)+F_y_f*np.cos(beta-delta)-F_x*np.sin(beta-delta))*(1/(m*V))-psi_dot
    xx_dot[5] = 1/I_z*((F_x*np.sin(delta)+F_y_f*np.cos(delta))*a-F_y_r*b)

    xxp = xx + xx_dot*dt
    
    return xxp, xx_dot