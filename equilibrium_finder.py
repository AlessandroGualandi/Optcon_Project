# OPTCON exam project
# Gualandi, Piraccini, Bosi
# # Find the circular trajectory target with radius RR and linear (tangent) velocity VV

import numpy as np
import gradient as grd
import dynamics as dyn

def eq_finder (WW, VV, ww_eq) :
    
    nIter = 1000
    stepsize = 0.02

    print("Initial arbitrary ww_eq [beta, delta, Fx] = {}".format(ww_eq))
    print("Running Newton's method for root finding ...")
    # Newton method for root finding

    for i in range(nIter):
        # The gradient function needs VV and WW
        grad_ww = -grd.partial_gradient(ww_eq, VV, WW)
        xx_full = [0, 0, 0, VV, ww_eq[0], WW]
        Delta_xx = dyn.dynamics(xx_full, [ww_eq[1], ww_eq[2]])[1][3:]
        
        '''
        # Armijo for step size
        b = 0.7
        c = 0.5
        stepsize = 1
        real_cost = grd.cost(VV,WW,ww_eq+stepsize*grad_ww)
        ref_cost = grd.cost(VV,WW,ww_eq)+c*stepsize*(-grad_ww).T@(grad_ww)
        while (real_cost > ref_cost):
            stepsize *= b
            real_cost = grd.cost(VV,WW,ww_eq+stepsize*grad_ww)
            ref_cost = grd.cost(VV,WW,ww_eq)+c*stepsize*(-grad_ww).T@(grad_ww)
            #print("real_cost: {}".format(real_cost))
            #print("ref_cost: {}".format(ref_cost))
        #print("stepsize: {}".format(ref_cost))
        '''
        ww_eq = ww_eq + stepsize*np.linalg.inv(grad_ww)@Delta_xx
        
    print("Final ww_eq [beta, delta, Fx] = {}".format(ww_eq))
    print("Check dynamics at t=0 ([0,0,0] ideal): {}".format(Delta_xx))


    return ww_eq
