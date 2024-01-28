import numpy as np
import dynamics as dyn

# Compute the gradient of the vector [V_dot, beta_dot, psi_dotdot] wrt [beta, delta, Fx]
dt = dyn.dt

def partial_gradient(ww, V, psi_dot):
    # The partial gradient is used to compute equilibria with the root finding newton method.
    # For this reason we are interested only in a subset of the full gradient.
    grad = np.zeros([3,3])

    beta = ww[0]
    delta = ww[1]
    F_x = ww[2]

    # mechanical parameters
    a = dyn.a
    b = dyn.b
    mu =dyn.mu
    m = dyn.m
    I_z = dyn.I_z
    F_z_f = dyn.F_z_f
    F_z_r = dyn.F_z_r

    beta_f = delta-(V*np.sin(beta)+a*psi_dot)/(V*np.cos(beta))
    beta_r = -(V*np.sin(beta)-b*psi_dot)/(V*np.cos(beta))

    F_y_f = mu*F_z_f*beta_f
    F_y_r = mu*F_z_r*beta_r

    # Derivatives of F_y_f and F_y_r wrt V, beta, psi_dot, delta, F_x
    # These partial derivatives are necessary for the gradient computation

    dFyf_dV = mu*F_z_f*a*psi_dot/((V**2)*np.cos(beta)) 
    dFyr_dV = -mu*F_z_r*b*psi_dot/((V**2)*np.cos(beta))
    dFyf_dbeta = -mu*F_z_f*(V+a*psi_dot*np.sin(beta))/(V*(np.cos(beta)**2))
    dFyr_dbeta = -mu*F_z_r*(V-b*psi_dot*np.sin(beta))/(V*(np.cos(beta)**2))
    dFyf_dpsidot = -mu*F_z_f*a/(V*np.cos(beta))
    dFyr_dpsidot = mu*F_z_r*b/(V*np.cos(beta))
    dFyf_ddelta = mu*F_z_f
    dFyr_ddelta = 0
    dFyf_dFx = 0
    dFyr_dFx = 0


    # Derivatives of Vdot, betadot, psidotdot wrt beta, delta, F_x
    dVdot_dbeta = (dFyr_dbeta*np.sin(beta)+F_y_r*np.cos(beta)-F_x*np.sin(beta-delta)+dFyf_dbeta*np.sin(beta-delta)+F_y_f*np.cos(beta-delta))/m
    dVdot_ddelta = (F_x*np.sin(beta-delta)+dFyf_ddelta*np.sin(beta-delta)-F_y_f*np.cos(beta-delta))/m
    dVdot_dFx = (np.cos(beta-delta))/m

    dbetadot_dbeta = (dFyr_dbeta*np.cos(beta)-F_y_r*np.sin(beta)+dFyf_dbeta*np.cos(beta-delta)-F_y_f*np.sin(beta-delta)-F_x*np.cos(beta-delta))/(m*V)
    dbetadot_ddelta = (dFyf_ddelta*np.cos(beta-delta)+F_y_f*np.sin(beta-delta)+F_x*np.cos(beta-delta))/(m*V)
    dbetadot_dFx = (-np.sin(beta-delta))/(m*V)

    dpsidotdot_dbeta = (dFyf_dbeta*a*np.cos(delta)-b*dFyr_dbeta)/I_z
    dpsidotdot_ddelta = (F_x*np.cos(delta)+dFyf_ddelta*np.cos(delta)-F_y_f*np.sin(delta))*a/I_z
    dpsidotdot_dFx = (np.sin(delta))*a/I_z

    grad[0][0] = dVdot_dbeta
    grad[0][1] = dVdot_ddelta
    grad[0][2] = dVdot_dFx

    grad[1][0] = dbetadot_dbeta
    grad[1][1] = dbetadot_ddelta
    grad[1][2] = dbetadot_dFx
    
    grad[2][0] = dpsidotdot_dbeta
    grad[2][1] = dpsidotdot_ddelta
    grad[2][2] = dpsidotdot_dFx

    return grad

# Compute the gradient of the vector [x_dot, y_dot, psi_dot, V_dot, beta_dot, psi_dotdot]
n_x = dyn.n_x
n_u =dyn.n_u

def full_gradient(ss, uu):
    # The full gradient is used to compute the feasible trajectory between two equilibria.
    # We want to be general so the cost function is function of all the state variables, not
    # only those used to compute equilibrium (V, beta, psidot)
    
    full_state_grad = np.zeros([3,3])
    input_grad = np.zeros([3, 2])
    
    # state vector
    '''
    xx = ss[0]
    yy = ss[1]
    psi = ss[2]
    V = ss[3]
    beta = ss[4]
    psi_dot = ss[5]
    delta = uu[0]
    F_x = uu[1]
    '''
    
    V = ss[0]
    beta = ss[1]
    psi_dot = ss[2]
    delta = uu[0]
    F_x = uu[1]
    
    # mechanical parameters
    a = dyn.a
    b = dyn.b
    mu =dyn.mu
    m = dyn.m
    I_z = dyn.I_z
    F_z_f = dyn.F_z_f
    F_z_r = dyn.F_z_r

    beta_f = delta-(V*np.sin(beta)+a*psi_dot)/(V*np.cos(beta))
    beta_r = -(V*np.sin(beta)-b*psi_dot)/(V*np.cos(beta))

    F_y_f = mu*F_z_f*beta_f
    F_y_r = mu*F_z_r*beta_r

    # Derivatives of F_y_f and F_y_r wrt V, beta, psi_dot, delta, F_x
    # These partial derivatives are necessary for the gradient computation

    dFyf_dV = mu*F_z_f*a*psi_dot/((V**2)*np.cos(beta)) 
    dFyr_dV = -mu*F_z_r*b*psi_dot/((V**2)*np.cos(beta))
    dFyf_dbeta = -mu*F_z_f*(V+a*psi_dot*np.sin(beta))/(V*(np.cos(beta)**2))
    dFyr_dbeta = -mu*F_z_r*(V-b*psi_dot*np.sin(beta))/(V*(np.cos(beta)**2))
    dFyf_dpsidot = -mu*F_z_f*a/(V*np.cos(beta))
    dFyr_dpsidot = mu*F_z_r*b/(V*np.cos(beta))
    dFyf_ddelta = mu*F_z_f
    dFyr_ddelta = 0
    dFyf_dFx = 0
    dFyr_dFx = 0

    # Derivatives of ss_dot wrt ss
    
    dVdot_dV = (dFyr_dV*np.sin(beta)+dFyf_dV*np.sin(beta-delta))/m
    dVdot_dbeta = (dFyr_dbeta*np.sin(beta)+F_y_r*np.cos(beta)-F_x*np.sin(beta-delta)+dFyf_dbeta*np.sin(beta-delta)+F_y_f*np.cos(beta-delta))/m
    dVdot_dpsidot = (dFyr_dpsidot*np.sin(beta)+dFyf_dpsidot*np.sin(beta-delta))/m
    
    dbetadot_dV = -1/(m*V**2)*(F_y_r*np.cos(beta)+F_y_f*np.cos(beta-delta)-F_x*np.sin(beta-delta))+1/(m*V)*(dFyr_dV*np.cos(beta)+dFyf_dV*np.cos(beta-delta))
    dbetadot_dbeta = (dFyr_dbeta*np.cos(beta)-F_y_r*np.sin(beta)+dFyf_dbeta*np.cos(beta-delta)-F_y_f*np.sin(beta-delta)-F_x*np.cos(beta-delta))/(m*V)
    dbetadot_dpsidot = (dFyr_dpsidot*np.cos(beta)+dFyf_dpsidot*np.cos(beta-delta))/(m*V)-1
    
    dpsidotdot_dV = (dFyf_dV*a*np.cos(delta)-b*dFyr_dV)/I_z
    dpsidotdot_dbeta = (dFyf_dbeta*a*np.cos(delta)-b*dFyr_dbeta)/I_z
    dpsidotdot_dpsidot = (dFyf_dpsidot*a*np.cos(delta)-b*dFyr_dpsidot)/I_z
    
    # Derivatives of ss_dot wrt uu
    
    dVdot_ddelta = (F_x*np.sin(beta-delta)+dFyf_ddelta*np.sin(beta-delta)-F_y_f*np.cos(beta-delta))/m
    dVdot_dFx = (np.cos(beta-delta))/m

    dbetadot_ddelta = (dFyf_ddelta*np.cos(beta-delta)+F_y_f*np.sin(beta-delta)+F_x*np.cos(beta-delta))/(m*V)
    dbetadot_dFx = (-np.sin(beta-delta))/(m*V)

    dpsidotdot_ddelta = (F_x*np.cos(delta)+dFyf_ddelta*np.cos(delta)-F_y_f*np.sin(delta))*a/I_z
    dpsidotdot_dFx = (np.sin(delta))*a/I_z


    


    full_state_grad[0][0] = 1+dt*dVdot_dV
    full_state_grad[0][1] = dt*dVdot_dbeta
    full_state_grad[0][2] = dt*dVdot_dpsidot

    full_state_grad[1][0] = dt*dbetadot_dV
    full_state_grad[1][1] = 1+dt*dbetadot_dbeta
    full_state_grad[1][2] = dt*dbetadot_dpsidot

    full_state_grad[2][0] = dt*dpsidotdot_dV
    full_state_grad[2][1] = dt*dpsidotdot_dbeta
    full_state_grad[2][2] = 1+dt*dpsidotdot_dpsidot

    input_grad[0,0] = dt*dVdot_ddelta
    input_grad[0,1] = dt*dVdot_dFx

    input_grad[1,0] = dt*dbetadot_ddelta
    input_grad[1,1] = dt*dbetadot_dFx

    input_grad[2,0] = dt*dpsidotdot_ddelta
    input_grad[2,1] = dt*dpsidotdot_dFx

    return full_state_grad.T, input_grad.T
