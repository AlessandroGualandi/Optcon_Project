import numpy as np
import dynamics as dyn

#n_x= dyn.n_x
#n_u = dyn.n_u
n_x = 3
n_u = 2


#QQt = np.array([[10000, 0], [0, 100]])
#QQt = 0.1*np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
QQt = 0.1*np.diag([1.0, 100.0, 100.0])
RRt = 0.01*np.diag([1.0, 1.0])
# RRt = 1*np.eye(ni)

QQT = QQt


def stagecost(xx,uu, xx_ref, uu_ref):
  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

      - uu \in \R^1 input at time t
      - uu_ref \in \R^2 input reference at time t


    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """

  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

  lx = QQt@(xx - xx_ref)
  lu = RRt@(uu - uu_ref)

  l2x = QQt
  l2u = RRt
  lxu = np.zeros((n_u,n_x))

  return ll.squeeze(), lx, lu, l2x, l2u, lxu

def termcost(xx,xx_ref):
  """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """

  xx = xx[:,None]
  xx_ref = xx_ref[:,None]

  llT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)

  lTx = QQT@(xx - xx_ref)

  l2Tx = QQt


  return llT.squeeze(), lTx, l2Tx