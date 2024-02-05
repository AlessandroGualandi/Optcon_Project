#
# OPTCON exam project
# Gualandi, Piraccini, Bosi
# 
# check the dimension of the matrices in order to compute the trajectory tracking
# 1/12/2023
#

def check_matrix(AAin, BBin, QQin, RRin, SSin, TT, qqin = None, rrin = None, qqfin = None) :
    try:
        # check if matrix is (.. x .. x TT) - 3 dimensional array 
        ns, lA = AAin.shape[1:]
    except:
        # if not 3 dimensional array, make it (.. x .. x 1)
        AAin = AAin[:,:,None]
        ns, lA = AAin.shape[1:]

    try:  
        ni, lB = BBin.shape[1:]
    except:
        BBin = BBin[:,:,None]
        ni, lB = BBin.shape[1:]

    try:
        nQ, lQ = QQin.shape[1:]
    except:
        QQin = QQin[:,:,None]
        nQ, lQ = QQin.shape[1:]

    try:
        nR, lR = RRin.shape[1:]
    except:
        RRin = RRin[:,:,None]
        nR, lR = RRin.shape[1:]

    try:
        nSi, nSs, lS = SSin.shape
    except:
        SSin = SSin[:,:,None]
        nSi, nSs, lS = SSin.shape

    # Check dimensions consistency -- safety
    if nQ != ns:
        print("Matrix Q does not match number of states")
        exit()
    if nR != ni:
        print("Matrix R does not match number of inputs")
        exit()
    if nSs != ns:
        print("Matrix S does not match number of states")
        exit()
    if nSi != ni:
        print("Matrix S does not match number of inputs")
        exit()


    if lA < TT:
        AAin = AAin.repeat(TT, axis=2)
    if lB < TT:
        BBin = BBin.repeat(TT, axis=2)
    if lQ < TT:
        QQin = QQin.repeat(TT, axis=2)
    if lR < TT:
        RRin = RRin.repeat(TT, axis=2)
    if lS < TT:
        SSin = SSin.repeat(TT, axis=2)

    # Check for affine terms

    augmented = False

    if qqin is not None or rrin is not None or qqfin is not None:
        augmented = True
        print("Augmented term!")

    return [ns, ni, AAin, BBin, QQin, RRin, SSin]