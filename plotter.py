import matplotlib.pyplot as plt
import numpy as np

def plot_ref(xx_ref, uu_ref, TT):
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

    plt.figure('delta')
    plt.xlabel('steps')
    plt.ylabel('delta_ref')
    plt.grid()
    plt.plot(steps, uu_ref[0,:TT-1], color='red',label='delta_ref')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('F_x')
    plt.xlabel('steps')
    plt.ylabel('F_x_ref')
    plt.grid()
    plt.plot(steps, uu_ref[1,:TT-1], color='blue',label='F_x_ref')
    plt.legend(loc="upper right")
    plt.show()

def plot_init_guess(xx_guess, uu_guess, TT):
    steps = np.arange(TT-1)
    plt.figure('Velocity')
    plt.xlabel('steps')
    plt.ylabel('V_guess')
    plt.grid()
    plt.plot(steps, xx_guess[3,:TT-1], color='red', label='V_guess')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('Beta')
    plt.xlabel('steps')
    plt.ylabel('beta_guess')
    plt.grid()
    plt.plot(steps, xx_guess[4,:TT-1], color='green', label='beta_guess')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('Psi_dot')
    plt.xlabel('steps')
    plt.ylabel('psi_dot_guess')
    plt.grid()
    plt.plot(steps, xx_guess[5,:TT-1], color='blue', label='psi_dot_guess')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('delta')
    plt.xlabel('steps')
    plt.ylabel('delta_guess')
    plt.grid()
    plt.plot(steps, uu_guess[0,:TT-1], color='red',label='delta_guess')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('F_x')
    plt.xlabel('steps')
    plt.ylabel('F_x_guess')
    plt.grid()
    plt.plot(steps, uu_guess[1,:TT-1], color='blue',label='F_x_guess')
    plt.legend(loc="upper right")
    plt.show()


def plot_opt_ref(xx_ref, uu_ref, xx_opt, uu_opt, TT, itr):
    # State evolution
    steps = np.arange(TT-1)
    plt.figure(f'Velocity iter: {itr}')
    plt.xlabel('steps')
    plt.ylabel('V_opt')
    plt.grid()
    plt.plot(steps, xx_ref[3,:TT-1], color='red',label='V_ref',linestyle='dashed')
    plt.plot(steps, xx_opt[3,:TT-1], color='red',label='V_opt')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure('Beta')
    plt.xlabel('steps')
    plt.ylabel('beta_opt')
    plt.grid()
    plt.plot(steps, xx_ref[4,:TT-1], color='green',label='beta_ref',linestyle='dashed')
    plt.plot(steps, xx_opt[4,:TT-1], color='green',label='beta_opt')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure('Psi_dot')
    plt.xlabel('steps')
    plt.ylabel('psi_dot_opt')
    plt.grid()
    plt.plot(steps, xx_ref[5,:TT-1], color='blue',label='psi_dot_ref',linestyle='dashed')
    plt.plot(steps, xx_opt[5,:TT-1], color='blue',label='psi_dot_opt')
    plt.legend(loc="lower left")
    plt.show()

    #########################################
    # Input evolution

    plt.figure('delta')
    plt.xlabel('steps')
    plt.ylabel('delta_opt')
    plt.grid()
    plt.plot(steps, uu_ref[0,:TT-1], color='red',linestyle='dashed',label='delta_ref')
    plt.plot(steps, uu_opt[0,:TT-1], color='red',label='delta_opt')
    plt.legend(loc="lower left")
    plt.show()

    plt.figure('F_x_opt')
    plt.xlabel('steps')
    plt.ylabel('F_x')
    plt.grid()
    plt.plot(steps, uu_ref[1,:TT-1], color='blue',linestyle='dashed',label='F_x_ref')
    plt.plot(steps, uu_opt[1,:TT-1], color='blue',label='F_x_opt')
    plt.legend(loc="lower left")
    plt.show()


def plot_opt_iters(xx_ref, uu_ref, xx_opt, uu_opt, TT, itr):
    # State evolution
    steps = np.arange(TT-1)
    plt.figure(f'Velocity iter: {itr}')
    plt.xlabel('steps')
    plt.ylabel('V_opt')
    plt.grid()
    plt.plot(steps, xx_ref[3,:TT-1], color='red',label='V_ref',linestyle='dashed')

    plt.plot(steps, xx_opt[3,:TT-1,0],label=f'V_opt_iter {0}')
    plt.plot(steps, xx_opt[3,:TT-1,3],label=f'V_opt_iter {3}')
    plt.plot(steps, xx_opt[3,:TT-1,itr],label=f'V_opt_iter {itr}')

    plt.legend()
    plt.show()

    plt.figure('Beta')
    plt.xlabel('steps')
    plt.ylabel('beta_opt')
    plt.grid()
    plt.plot(steps, xx_ref[4,:TT-1],label='beta_ref',linestyle='dashed')

    plt.plot(steps, xx_opt[4,:TT-1,0],label=f'beta_opt_iter {0}')
    plt.plot(steps, xx_opt[4,:TT-1,3],label=f'beta_opt_iter {3}')
    plt.plot(steps, xx_opt[4,:TT-1,itr],label=f'beta_opt_iter {itr}')

    plt.legend()
    plt.show()

    plt.figure('Psi_dot')
    plt.xlabel('steps')
    plt.ylabel('psi_dot_opt')
    plt.grid()
    plt.plot(steps, xx_ref[5,:TT-1], color='blue',label='psi_dot_ref',linestyle='dashed')

    plt.plot(steps, xx_opt[5,:TT-1,0],label=f'psi_dot_opt_iter {0}')
    plt.plot(steps, xx_opt[5,:TT-1,3],label=f'psi_dot_opt_iter {3}')
    plt.plot(steps, xx_opt[5,:TT-1,itr],label=f'psi_dot_opt_iter {itr}')

    plt.legend()
    plt.show()

    #########################################
    # Input evolution

    plt.figure('delta')
    plt.xlabel('steps')
    plt.ylabel('delta_opt')
    plt.grid()
    plt.plot(steps, uu_ref[0,:TT-1], color='red',linestyle='dashed',label='delta_ref')

    plt.plot(steps, uu_opt[0,:TT-1,0],label=f'delta_opt_iter {0}')
    plt.plot(steps, uu_opt[0,:TT-1,3],label=f'delta_opt_iter {3}')
    plt.plot(steps, uu_opt[0,:TT-1,itr],label=f'delta_opt_iter {itr}')

    plt.legend()
    plt.show()

    plt.figure('F_x_opt')
    plt.xlabel('steps')
    plt.ylabel('F_x')
    plt.grid()
    plt.plot(steps, uu_ref[1,:TT-1], color='blue',linestyle='dashed',label='F_x_ref')

    plt.plot(steps, uu_opt[1,:TT-1,0],label=f'Fx_opt_iter {0}')
    plt.plot(steps, uu_opt[1,:TT-1,3],label=f'Fx_opt_iter {3}')
    plt.plot(steps, uu_opt[1,:TT-1,itr],label=f'Fx_opt_iter {itr}')

    plt.legend()
    plt.show()


def plot_LQR_opt(xx_opt, uu_opt, xx_LQR, uu_LQR, TT, alpha):

    n_offset = len(alpha)

    steps = np.arange(TT-1)
    plt.figure('Velocity')
    plt.xlabel('steps')
    plt.ylabel('V_LQR')
    plt.grid()
    plt.plot(steps, xx_opt[3,:TT-1], color='red',label='V_opt',linestyle='dashed')
    for kk in range(n_offset):
        plt.plot(steps, xx_LQR[3,:TT-1,kk],label=f'alpha = {alpha[kk]}')
    plt.legend()
    plt.show()

    plt.figure('Beta')
    plt.xlabel('steps')
    plt.ylabel('beta_LQR')
    plt.grid()
    plt.plot(steps, xx_opt[4,:TT-1], color='green',label='beta_opt',linestyle='dashed')
    for kk in range(n_offset):
        plt.plot(steps, xx_LQR[4,:TT-1,kk],label=f'alpha = {alpha[kk]}')
    plt.legend()
    plt.show()

    plt.figure('Psi_dot')
    plt.xlabel('steps')
    plt.ylabel('psi_dot_LQR')
    plt.grid()
    plt.plot(steps, xx_opt[5,:TT-1], color='blue',label='psi_dot_opt',linestyle='dashed')
    for kk in range(n_offset):
        plt.plot(steps, xx_LQR[5,:TT-1,kk],label=f'alpha = {alpha[kk]}')
    plt.legend()
    plt.show()

    #########################################
    # Input evolution

    plt.figure('delta')
    plt.xlabel('steps')
    plt.ylabel('delta_LQR')
    plt.grid()
    plt.plot(steps, uu_opt[0,:TT-1], color='red',linestyle='dashed',label='delta_opt')
    for kk in range(n_offset):
        plt.plot(steps, uu_LQR[0,:TT-1,kk],label=f'alpha = {alpha[kk]}')
    plt.legend()
    plt.show()

    plt.figure('F_x')
    plt.xlabel('steps')
    plt.ylabel('F_x_LQR')
    plt.grid()
    plt.plot(steps, uu_opt[1,:TT-1], color='blue',linestyle='dashed',label='F_x_opt')
    for kk in range(n_offset):
        plt.plot(steps, uu_LQR[1,:TT-1,kk],label=f'alpha = {alpha[kk]}')
    plt.legend()
    plt.show()

def plot_LQR_error(xx_opt, uu_opt, xx_LQR, uu_LQR, TT, alpha):

    n_offset = len(alpha)

    V_error = np.zeros((TT-1, n_offset))
    beta_error = np.zeros((TT-1, n_offset))
    psi_dot_error = np.zeros((TT-1, n_offset))
    delta_error = np.zeros((TT-1, n_offset))
    Fx_error = np.zeros((TT-1, n_offset))

    
    #beta_error = xx_opt[4,:] - xx_LQR[4,:,:]
    #psi_dot_error = xx_opt[5,:] - xx_LQR[5,:,:]
    #delta_error = uu_opt[0,:] - uu_LQR[0,:,:]
    #Fx_error = uu_opt[1,:] - uu_LQR[1,:,:]

    steps = np.arange(TT-1)
    plt.figure('Velocity')
    plt.xlabel('steps')
    plt.ylabel('V_LQR_error')
    plt.grid()
    for kk in range(n_offset):
        V_error = xx_LQR[3,:,kk] - xx_opt[3,:]
        plt.plot(steps, V_error[:TT-1],label=f'alpha = {alpha[kk]}')
    plt.legend()
    plt.show()

    plt.figure('Beta')
    plt.xlabel('steps')
    plt.ylabel('beta_LQR_error')
    plt.grid()
    for kk in range(n_offset):
        beta_error = xx_LQR[4,:,kk] - xx_opt[4,:]
        plt.plot(steps, beta_error[:TT-1],label=f'alpha = {alpha[kk]}')
    plt.legend()
    plt.show()

    plt.figure('Psi_dot')
    plt.xlabel('steps')
    plt.ylabel('psi_dot_LQR_error')
    plt.grid()
    for kk in range(n_offset):
        psi_dot_error = xx_LQR[5,:,kk] - xx_opt[5,:]
        plt.plot(steps, psi_dot_error[:TT-1],label=f'alpha = {alpha[kk]}')
    plt.legend()
    plt.show()

    #########################################
    # Input evolution

    plt.figure('delta')
    plt.xlabel('steps')
    plt.ylabel('delta_LQR_error')
    plt.grid()
    for kk in range(n_offset):
        delta_error = uu_LQR[0,:,kk] - uu_opt[0,:]
        plt.plot(steps, delta_error[:TT-1],label=f'alpha = {alpha[kk]}')
    plt.legend()
    plt.show()

    plt.figure('F_x')
    plt.xlabel('steps')
    plt.ylabel('F_x_LQR_error')
    plt.grid()
    for kk in range(n_offset):
        Fx_error = uu_LQR[1,:,kk] - uu_opt[1,:]
        plt.plot(steps, Fx_error[:TT-1],label=f'alpha = {alpha[kk]}')
    plt.legend()
    plt.show()