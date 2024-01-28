import dynamics as dyn
import cost as cst
import numpy as np
import gradient as grd
import matplotlib.pyplot as plt
import solver_ltv_LQR as ltv_LQR
import numpy as np

delta_t = dyn.dt
TT = int(dyn.tf/delta_t)
#n_x = dyn.n_x
#n_u = dyn.n_u
n_x = 3
n_u = 2


def optimal_trajectory(xx_ref, uu_ref, xx_init, uu_init):

    # ARMIJO PARAMETERS
    Armijo_plot = True
    trajectory_plot = True
    cc = 0.5
    beta = 0.7
    armijo_maxiters = 20
    term_cond = 1e-6

    #max_iters = int(1e2)
    max_iters = int(100)
    stepsize_0 = 1

    x0 = xx_ref[:,0]

    # Arrays to store data
    #xx = np.zeros((n_x, TT, max_iters))
    xx = np.zeros((6, TT, max_iters))
    uu = np.zeros((n_u, TT, max_iters))

    lmbd = np.zeros((n_x, TT, max_iters))

    deltau = np.zeros((n_u,TT, max_iters))
    dJ = np.zeros((n_u,TT, max_iters))

    JJ = np.zeros(max_iters)
    descent = np.zeros(max_iters)
    descent_arm = np.zeros(max_iters)

    # Computing feasible trajectory
    kk = 0
    #xx[:,:,0] = xx_init
    xx[:,:,0] = xx_init
    uu[:,:,0] = uu_init

    for kk in range(max_iters-1):
    
        JJ[kk] = 0
        # calculate cost
        for tt in range(TT-1):
            temp_cost = cst.stagecost(xx[3:,tt, kk], uu[:,tt,kk], xx_ref[3:,tt], uu_ref[:,tt])[0]
            JJ[kk] += temp_cost

        temp_cost = cst.termcost(xx[3:,-1,kk], xx_ref[3:,-1])[0]
        JJ[kk] += temp_cost

        ##################################
        # Descent direction calculation
        ##################################

        # lambda is useless if we consider the regularized matrices!
        lmbd_temp = cst.termcost(xx[3:,TT-1,kk], xx_ref[3:,TT-1])[1]
        lmbd[:,TT-1,kk] = lmbd_temp.squeeze()

        AA = np.zeros((n_x,n_x,TT))
        BB = np.zeros((n_x,n_u,TT))
        qq = np.zeros((n_x,TT))
        rr = np.zeros((n_u,TT))

        QQtr = np.zeros((n_x,n_x,TT))
        RRtr = np.zeros((n_u,n_u,TT))
        SStr = np.zeros((n_u,n_x,TT))
        qqT_temp, QQT = cst.termcost(xx[3:,-1, kk], xx_ref[3:,-1])[1:]
        qqT = qqT_temp.squeeze()

        for tt in reversed(range(TT-1)):  # integration backward in time

            at, bt = cst.stagecost(xx[3:,tt, kk], uu[:,tt,kk], xx_ref[3:,tt], uu_ref[:,tt])[1:3]
            fx, fu = grd.full_gradient(xx[3:,tt,kk], uu[:,tt,kk])

            AA[:,:,tt] = fx.T
            BB[:,:,tt] = fu.T

            #print(f'AA shape: {AA[:,:,tt].shape}')
            #print(f'lmbd shape: {lmbd[:,tt+1,kk].shape}')
            #print(f'qq shape: {at.shape}')

            lmbd_temp = AA[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + at
            dJ_temp = BB[:,:,tt].T@lmbd[:,tt+1,kk][:,None] + bt
            #deltau_temp = - dJ_temp

            #print(f'lmbd_temp shape: {lmbd_temp.shape}')
            lmbd[:,tt,kk] = lmbd_temp.squeeze()
            dJ[:,tt,kk] = dJ_temp.squeeze()
            #deltau[:,tt,kk] = deltau_temp.squeeze()

            qq[:,tt] = at.squeeze()
            rr[:,tt] = bt.squeeze()
      
            #lmbd_temp = At.T@lmbd[:,tt+1,kk][:,None] + at
            #lmbd[:,tt,kk] = lmbd_temp.squeeze()

            QQtr[:,:,tt] = cst.stagecost(xx[3:,tt, kk], uu[:,tt,kk], xx_ref[3:,tt], uu_ref[:,tt])[3]
            RRtr[:,:,tt] = cst.stagecost(xx[3:,tt, kk], uu[:,tt,kk], xx_ref[3:,tt], uu_ref[:,tt])[4]
            SStr[:,:,tt] = cst.stagecost(xx[3:,tt, kk], uu[:,tt,kk], xx_ref[3:,tt], uu_ref[:,tt])[5]


        delta_x0 = np.zeros(n_x)
        #KK, sigma, delta_x, deltau[:,:,kk] = ltv_LQR.ltv_LQR(AA, BB, QQtr, RRtr, SStr, QQT, delta_x0, TT, qq, rr, qqT)
        deltau[:,:,kk] = ltv_LQR.ltv_LQR(AA, BB, QQtr, RRtr, SStr, QQT, delta_x0, TT, qq, rr, qqT)[3]

        for tt in range(TT-1):
            descent[kk] += deltau[:,tt,kk].T@deltau[:,tt,kk]
            descent_arm[kk] += dJ[:,tt,kk].T@deltau[:,tt,kk]

        #print(f'deltau: {deltau}')
        #input(f'start {kk} iteration')

        ##################################
        # Stepsize selection - ARMIJO
        ##################################

        stepsizes = []  # list of stepsizes
        costs_armijo = []

        stepsize = stepsize_0

        for ii in range(armijo_maxiters):

            # temp solution update

            #xx_temp = np.zeros((n_x,TT))
            xx_temp = np.zeros((6,TT))
            uu_temp = np.zeros((n_u,TT))

            xx_temp[:,0] = x0

            for tt in range(TT-1):
                uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
                xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

            # temp cost calculation
            JJ_temp = 0

            for tt in range(TT-1):
                #temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                temp_cost = cst.stagecost(xx_temp[3:,tt], uu_temp[:,tt], xx_ref[3:,tt], uu_ref[:,tt])[0]
                JJ_temp += temp_cost

            #temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
            temp_cost = cst.termcost(xx_temp[3:,-1], xx_ref[3:,-1])[0]
            JJ_temp += temp_cost

            stepsizes.append(stepsize)
            costs_armijo.append(np.min([JJ_temp, 100*JJ[kk]]))
            #costs_armijo.append(JJ_temp)

            if JJ_temp > JJ[kk]  + cc*stepsize*descent_arm[kk]:
                # update the stepsize
                stepsize = beta*stepsize

            else:
                print('Armijo stepsize = {:.3e}'.format(stepsize))
                break


        if Armijo_plot and kk%10==0:
            steps = np.linspace(0,stepsize_0,int(2e1))
            costs = np.zeros(len(steps))

            for ii in range(len(steps)):

                step = steps[ii]

                # temp solution update

                #xx_temp = np.zeros((n_x,TT))
                xx_temp = np.zeros((6,TT))
                uu_temp = np.zeros((n_u,TT))

                xx_temp[:,0] = x0

                for tt in range(TT-1):
                    uu_temp[:,tt] = uu[:,tt,kk] + step*deltau[:,tt,kk]
                    xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

                # temp cost calculation
                JJ_temp = 0

                for tt in range(TT-1):
                    #temp_cost = cst.stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                    temp_cost = cst.stagecost(xx_temp[3:,tt], uu_temp[:,tt], xx_ref[3:,tt], uu_ref[:,tt])[0]
                    JJ_temp += temp_cost

                #temp_cost = cst.termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
                temp_cost = cst.termcost(xx_temp[3:,-1], xx_ref[3:,-1])[0]
                JJ_temp += temp_cost

                costs[ii] = np.min([JJ_temp, 100*JJ[kk]])
                #costs[ii] = JJ_temp


            plt.figure(1)
            plt.clf()

            plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
            plt.plot(steps, JJ[kk] + descent_arm[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            # plt.plot(steps, JJ[kk] - descent[kk]*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.plot(steps, JJ[kk] + cc*descent_arm[kk]*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

            plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

            plt.grid()
            plt.xlabel('stepsize')
            plt.legend()
            plt.draw()

            plt.show()



        #xx_temp = np.zeros((n_x,TT))
        xx_temp = np.zeros((6,TT))
        uu_temp = np.zeros((n_u,TT))

        xx_temp[:,0] = x0

        for tt in range(TT-1):
            uu_temp[:,tt] = uu[:,tt,kk] + stepsize*deltau[:,tt,kk]
            xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

        xx[:,:,kk+1] = xx_temp
        uu[:,:,kk+1] = uu_temp

        # Plot intermediate trajectory
        if trajectory_plot and kk%10==0:
            steps = np.arange(TT-1)

            steps = np.arange(TT-1)
            plt.figure('Velocity')
            plt.xlabel('steps')
            plt.ylabel('V_opt')
            plt.grid()
            plt.plot(steps, xx[3,:TT-1,kk], color='red', label='V_opt')
            plt.plot(steps, xx_ref[3,:TT-1], color='red',linestyle='dashed', label='V_ref')
            plt.legend(loc="upper right")
            plt.show()

            plt.figure('Beta')
            plt.xlabel('steps')
            plt.ylabel('beta_opt')
            plt.grid()
            plt.plot(steps, xx[4,:TT-1,kk], color='green', label='beta_opt')
            plt.plot(steps, xx_ref[4,:TT-1], color='green',linestyle='dashed', label='beta_ref')
            plt.legend(loc="upper right")
            plt.show()

            plt.figure('Psi_dot')
            plt.xlabel('steps')
            plt.ylabel('psi_dot_opt')
            plt.grid()
            plt.plot(steps, xx[5,:TT-1,kk], color='blue', label='psi_dot_opt')
            plt.plot(steps, xx_ref[5,:TT-1], color='blue',linestyle='dashed', label='psi_dot_ref')
            plt.legend(loc="upper right")
            plt.show()

            plt.figure('delta')
            plt.xlabel('steps')
            plt.ylabel('delta_opt')
            plt.grid()
            plt.plot(steps, uu[0,:TT-1,kk], color='blue', label='delta_opt')
            plt.plot(steps, uu_ref[0,:TT-1],color='blue',linestyle='dashed', label='delta_ref')
            plt.legend(loc="upper right")
            plt.show()


            plt.figure('Fx')
            plt.xlabel('steps')
            plt.ylabel('Fx_opt')
            plt.grid()
            plt.plot(steps, uu[1,:TT-1,kk], color='red', label='Fx_opt')
            plt.plot(steps, uu_ref[1,:TT-1],color='red',linestyle='dashed', label='Fx_ref')
            plt.legend(loc="upper right")
            plt.show()

        ############################
        # Termination condition
        ############################

        print('Iter = {}\t Descent = {:.3e}\t Cost = {:.3e}'.format(kk,descent[kk], JJ[kk]))

        if descent[kk] <= term_cond:
            max_iters = kk
            break
    

    plt.figure('descent direction')
    plt.plot(np.arange(max_iters), descent[:max_iters])
    plt.xlabel('$k$')
    plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)


    plt.figure('cost')
    plt.plot(np.arange(max_iters), JJ[:max_iters])
    plt.xlabel('$k$')
    plt.ylabel('$J(\\mathbf{u}^k)$')
    plt.yscale('log')
    plt.grid()
    plt.show(block=False)

    return xx, uu