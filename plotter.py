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


def plot_opt_ref(xx_ref, uu_ref, xx_opt, uu_opt, TT):
    # State evolution
    steps = np.arange(TT-1)
    plt.figure('Velocity')
    plt.xlabel('steps')
    plt.ylabel('V_opt')
    plt.grid()
    plt.plot(steps, xx_ref[3,:TT-1], color='red',label='V_ref',linestyle='dashed')
    plt.plot(steps, xx_opt[3,:TT-1], color='red',label='V_opt')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('Beta')
    plt.xlabel('steps')
    plt.ylabel('beta_opt')
    plt.grid()
    plt.plot(steps, xx_ref[4,:TT-1], color='green',label='beta_ref',linestyle='dashed')
    plt.plot(steps, xx_opt[4,:TT-1], color='green',label='beta_opt')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('Psi_dot')
    plt.xlabel('steps')
    plt.ylabel('psi_dot_opt')
    plt.grid()
    plt.plot(steps, xx_ref[5,:TT-1], color='blue',label='psi_dot_ref',linestyle='dashed')
    plt.plot(steps, xx_opt[5,:TT-1], color='blue',label='psi_dot_opt')
    plt.legend(loc="upper right")
    plt.show()

    #########################################
    # Input evolution

    plt.figure('delta')
    plt.xlabel('steps')
    plt.ylabel('delta_opt')
    plt.grid()
    plt.plot(steps, uu_ref[0,:TT-1], color='red',linestyle='dashed',label='delta_ref')
    plt.plot(steps, uu_opt[0,:TT-1], color='red',label='delta_opt')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('F_x_opt')
    plt.xlabel('steps')
    plt.ylabel('F_x')
    plt.grid()
    plt.plot(steps, uu_ref[1,:TT-1], color='blue',linestyle='dashed',label='F_x_ref')
    plt.plot(steps, uu_opt[1,:TT-1], color='blue',label='F_x_opt')
    plt.legend(loc="upper right")
    plt.show()


def plot_LQR_opt(xx_opt, uu_opt, xx_LQR, uu_LQR, TT):

    steps = np.arange(TT-1)
    plt.figure('Velocity')
    plt.xlabel('steps')
    plt.ylabel('V_LQR')
    plt.grid()
    plt.plot(steps, xx_opt[3,:TT-1], color='red',label='V_opt',linestyle='dashed')
    plt.plot(steps, xx_LQR[3,:TT-1], color='red',label='V_LQR')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('Beta')
    plt.xlabel('steps')
    plt.ylabel('beta_LQR')
    plt.grid()
    plt.plot(steps, xx_opt[4,:TT-1], color='green',label='beta_opt',linestyle='dashed')
    plt.plot(steps, xx_LQR[4,:TT-1], color='green',label='beta_LQR')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('Psi_dot')
    plt.xlabel('steps')
    plt.ylabel('psi_dot_LQR')
    plt.grid()
    plt.plot(steps, xx_opt[5,:TT-1], color='blue',label='psi_dot_opt',linestyle='dashed')
    plt.plot(steps, xx_LQR[5,:TT-1], color='blue',label='psi_dot_LQR')
    plt.legend(loc="upper right")
    plt.show()

    #########################################
    # Input evolution

    plt.figure('delta')
    plt.xlabel('steps')
    plt.ylabel('delta_LQR')
    plt.grid()
    plt.plot(steps, uu_opt[0,:TT-1], color='red',linestyle='dashed',label='delta_opt')
    plt.plot(steps, uu_LQR[0,:TT-1], color='red',label='delta_LQR')
    plt.legend(loc="upper right")
    plt.show()

    plt.figure('F_x')
    plt.xlabel('steps')
    plt.ylabel('F_x_LQR')
    plt.grid()
    plt.plot(steps, uu_opt[1,:TT-1], color='blue',linestyle='dashed',label='F_x_opt')
    plt.plot(steps, uu_LQR[1,:TT-1], color='blue',label='F_x_LQR')
    plt.legend(loc="upper right")
    plt.show()