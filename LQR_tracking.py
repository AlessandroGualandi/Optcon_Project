import numpy as np
import dynamics as dyn
import cost as cst
import gradient as grd
import solver_ltv_LQR as ltv_LQR

TT = int(dyn.tf/dyn.dt)
#n_x = dyn.n_x
#n_u = dyn.n_u
n_x = 3
n_u = 2

def LQR_trajectory(xx_opt, uu_opt, offset):

    AA = np.zeros((n_x,n_x,TT))
    BB = np.zeros((n_x,n_u,TT))
    #qq = np.zeros((n_x,TT))
    #rr = np.zeros((n_u,TT))

    QQtr = np.zeros((n_x,n_x,TT))
    RRtr = np.zeros((n_u,n_u,TT))
    SStr = np.zeros((n_u,n_x,TT))
    QQT = cst.termcost(xx_opt[3:,-1], np.zeros(3))[2]
    #QQT = cst.termcost(xx_opt[3:,-1], xx_ref[3:,-1])[2]

    x0 = xx_opt[:,0]

    for tt in reversed(range(TT-1)):  # integration backward in time

        fx, fu = grd.full_gradient(xx_opt[3:,tt], uu_opt[:,tt])

        AA[:,:,tt] = fx.T
        BB[:,:,tt] = fu.T
        
        QQtr[:,:,tt] = cst.stagecost(xx_opt[3:,tt], uu_opt[:,tt], np.zeros(n_x), np.zeros(n_u))[3]
        RRtr[:,:,tt] = cst.stagecost(xx_opt[3:,tt], uu_opt[:,tt], np.zeros(n_x), np.zeros(n_u))[4]
        SStr[:,:,tt] = cst.stagecost(xx_opt[3:,tt], uu_opt[:,tt], np.zeros(n_x), np.zeros(n_u))[5]

    KK = ltv_LQR.ltv_LQR(AA, BB, QQtr, RRtr, SStr, QQT, x0[3:], TT)[0]

    xx = np.zeros((6,TT))
    uu = np.zeros((n_u,TT))

    # set initial condition offset
    xx[:,0] = x0 + offset

    for tt in range(TT-1):
        uu[:,tt] = uu_opt[:,tt] + KK[:,:,tt]@(xx[3:,tt] - xx_opt[3:,tt])
        # apply the feedback input to the non linear system
        xx[:,tt+1] = dyn.dynamics(xx[:,tt], uu[:,tt])[0]

    return xx, uu