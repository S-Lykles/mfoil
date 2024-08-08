import numpy as np
from scipy import sparse
from mfoil.geometry import TE_info, Panel, space_geom
from mfoil.utils import cosd, sind, vprint, norm2, atan2
from typing import List
from numpy.typing import NDArray

F64Arr = NDArray[np.float64]


class Isol:  # inviscid solution variables
    def __init__(self):
        self.AIC: F64Arr = np.array([])  # aero influence coeff matrix
        self.gamref: F64Arr = np.array([])  # 0,90-deg alpha vortex strengths at airfoil nodes
        self.gam: F64Arr = np.array([])  # vortex strengths at airfoil nodes (for current alpha)
        self.sstag: float = 0.0  # s location of stagnation point
        self.xstag: F64Arr = np.array([])  #
        self.sstag_g = np.array([0, 0])  # lin of sstag w.r.t. adjacent gammas
        self.sstag_ue = np.array([0, 0])  # lin of sstag w.r.t. adjacent ue values
        self.stag_idx: List[int] = [0, 0]  # node indices before/after stagnation
        self.sign_ue: F64Arr = np.array([])  # +/- 1 on upper/lower surface nodes
        self.xi: F64Arr = np.array([])  # distance from the stagnation at all points
        self.uewi: F64Arr = np.array([])  # inviscid edge velocity in wake
        self.uewiref: F64Arr = np.array([])  # 0,90-deg alpha inviscid ue solutions on wake
        self.ue_sigma: F64Arr = np.array([])  # d(ue)/d(source) matrix
        self.sigma_m: F64Arr = np.array([])  # d(source)/d(mass) matrix
        self.ue_m: F64Arr = np.array([])  # linearization of ue w.r.t. mass (all nodes)


def build_gamma(foil: Panel, param, alpha: float, chord=1) -> Isol:
    """
    Builds and solves the inviscid linear system for alpha=0, 90, and input angle

    Parameters
    ----------
    M : Mfoil
        Mfoil structure that contains the aerodynamic model
    alpha : float
        Angle of attack in degrees

    Updates
    -------
    M.isol.gamref : (N, 2) ndarray
        0,90deg vorticity distributions at each node
    M.isol.gam : (N,) ndarray
        Gamma for the particular input angle, alpha
    M.isol.AIC : (N+1, N+1) ndarray
        Aerodynamic influence coefficient matrix, filled in

    Notes
    -----
    - Utilizes a streamfunction approach with constant psi at each node
    - Implements a continuous linear vorticity distribution on the airfoil panels
    - Enforces the Kutta condition at the trailing edge (TE)
    - Accounts for TE gap through constant source/vorticity panels
    - Handles sharp TE through gamma extrapolation
    """
    isol = Isol()
    isol.sign_ue = np.ones(foil.N)  # do not distinguish sign of ue if inviscid
    N = foil.N  # number of points
    A = np.zeros([N + 1, N + 1])  # influence matrix
    rhs = np.zeros([N + 1, 2])  # right-hand sides for 0,90
    t, hTE, dtdx, tcp, tdp = TE_info(foil.x)  # trailing-edge info
    nogap = abs(hTE) < 1e-10 * chord  # indicates no TE gap

    vprint(param.verb, 1, '\n <<< Solving the inviscid problem >>> \n')

    # Build influence matrix and rhs
    for i in range(N):  # loop over nodes
        xi = foil.x[:, i]  # coord of node i
        for j in range(N - 1):  # loop over panels
            aij, bij = panel_linvortex_stream(foil.x[:, [j, j + 1]], xi)
            A[i, j] += aij
            A[i, j + 1] += bij
        A[i, N] = -1  # last unknown = streamfunction value on surf

        # right-hand sides
        rhs[i, :] = [-xi[1], xi[0]]
        # TE source influence
        a = panel_constsource_stream(foil.x[:, [N - 1, 0]], xi)
        A[i, 0] += -a * (0.5 * tcp)
        A[i, N - 1] += a * (0.5 * tcp)
        # TE vortex panel
        a, b = panel_linvortex_stream(foil.x[:, [N - 1, 0]], xi)
        A[i, 0] += -(a + b) * (-0.5 * tdp)
        A[i, N - 1] += (a + b) * (-0.5 * tdp)

    # special Nth equation (extrapolation of gamma differences) if no gap
    if nogap:
        A[N - 1, :] = 0
        A[N - 1, [0, 1, 2, N - 3, N - 2, N - 1]] = [1, -2, 1, -1, 2, -1]

    # Kutta condition
    A[N, 0] = 1
    A[N, N - 1] = 1

    # Solve system for unknown vortex strengths
    isol.AIC = A
    g = np.linalg.solve(isol.AIC, rhs)

    isol.gamref = g[0:N, :]  # last value is surf streamfunction
    isol.gam = isol.gamref[:, 0] * cosd(alpha) + isol.gamref[:, 1] * sind(alpha)
    return isol


def inviscid_velocity(X: F64Arr, G: F64Arr, Vinf: float, alpha: float, x: F64Arr, dolin: bool):
    """
    Returns the inviscid velocity at a point due to gamma on panels and freestream velocity

    Parameters
    ----------
    X : (N, 2) ndarray
        Coordinates of N panel nodes (N-1 panels)
    G : (N,) ndarray
        Vector of gamma values at each airfoil node
    Vinf : float
        Freestream speed magnitude
    alpha : float
        Angle of attack in degrees
    x : (2,) ndarray
        Location of the point at which the velocity vector is desired
    dolin : bool
        True to also return linearization

    Returns
    -------
    V : (2,) ndarray
        Velocity at the desired point.
    V_G : (2, N) ndarray, None
        Linearization of V with respect to G, returned if dolin is True

    Notes
    -----
    - Utilizes linear vortex panels on the airfoil
    - Accounts for trailing edge constant source/vortex panel
    - Includes the freestream contribution
    """

    N = X.shape[1]  # number of points
    V = np.zeros(2)  # velocity
    if dolin:
        V_G = np.zeros([2, N])
    t, hTE, dtdx, tcp, tdp = TE_info(X)  # trailing-edge info
    # assume x is not a midpoint of a panel (can check for this)
    for j in range(N - 1):  # loop over panels
        a, b = panel_linvortex_velocity(X[:, [j, j + 1]], x, None, False)
        V += a * G[j] + b * G[j + 1]
        if dolin:
            V_G[:, j] += a
            V_G[:, j + 1] += b
    # TE source influence
    a = panel_constsource_velocity(X[:, [N - 1, 0]], x, None)
    f1, f2 = a * (-0.5 * tcp), a * 0.5 * tcp
    V += f1 * G[0] + f2 * G[N - 1]
    if dolin:
        V_G[:, 0] += f1
        V_G[:, N - 1] += f2
    # TE vortex influence
    a, b = panel_linvortex_velocity(X[:, [N - 1, 0]], x, None, False)
    f1, f2 = (a + b) * (0.5 * tdp), (a + b) * (-0.5 * tdp)
    V += f1 * G[0] + f2 * G[N - 1]
    if dolin:
        V_G[:, 0] += f1
        V_G[:, N - 1] += f2
    # freestream influence
    V += Vinf * np.array([cosd(alpha), sind(alpha)])
    if dolin:
        return V, V_G
    else:
        return V, None


def build_wake(isol: Isol, foil: Panel, oper, wakelen, chord: float = 1):
    # builds wake panels from the inviscid solution
    # INPUT
    #   M     : mfoil class with a valid inviscid solution (gam)
    # OUTPUT
    #   M.wake.N  : Nw, the number of wake points
    #   M.wake.x  : coordinates of the wake points (2xNw)
    #   M.wake.s  : s-values of wake points (continuation of airfoil) (1xNw)
    #   M.wake.t  : tangent vectors at wake points (2xNw)
    # DETAILS
    #   Constructs the wake path through repeated calls to inviscid_velocity
    #   Uses a predictor-corrector method
    #   Point spacing is geometric; prescribed wake length and number of points

    assert len(isol.gam) > 0, 'No inviscid solution'
    N = foil.N  # number of points on the airfoil
    Vinf = oper.Vinf  # freestream speed
    Nw = int(np.ceil(N / 10 + 10 * wakelen))  # number of points on wake
    S = foil.s  # airfoil S values
    ds1 = 0.5 * (S[1] - S[0] + S[N - 1] - S[N - 2])  # first nominal wake panel size
    sv = space_geom(ds1, wakelen * chord, Nw)  # geometrically-spaced points
    xyw = np.zeros([2, Nw])
    tw = xyw.copy()  # arrays of x,y points and tangents on wake
    xy1, xyN = foil.x[:, 0], foil.x[:, N - 1]  # airfoil TE points
    xyte = 0.5 * (xy1 + xyN)  # TE midpoint
    n = xyN - xy1
    t = np.array([n[1], -n[0]])  # normal and tangent
    assert t[0] > 0, 'Wrong wake direction; ensure airfoil points are CCW'
    xyw[:, 0] = xyte + 1e-5 * t * chord  # first wake point, just behind TE
    sw = S[N - 1] + sv  # s-values on wake, measured as continuation of the airfoil

    # loop over rest of wake
    for i in range(Nw - 1):
        v1, _ = inviscid_velocity(foil.x, isol.gam, Vinf, oper.alpha, xyw[:, i], False)
        v1 = v1 / norm2(v1)
        tw[:, i] = v1  # normalized
        xyw[:, i + 1] = xyw[:, i] + (sv[i + 1] - sv[i]) * v1  # forward Euler (predictor) step
        v2, _ = inviscid_velocity(foil.x, isol.gam, Vinf, oper.alpha, xyw[:, i + 1], False)
        v2 = v2 / norm2(v2)
        tw[:, i + 1] = v2  # normalized
        xyw[:, i + 1] = xyw[:, i] + (sv[i + 1] - sv[i]) * 0.5 * (v1 + v2)  # corrector step

    # determine inviscid ue in the wake, and 0,90deg ref ue too
    uewi = np.zeros([Nw, 1])
    uewiref = np.zeros([Nw, 2])
    for i in range(Nw):
        v, _ = inviscid_velocity(foil.x, isol.gam, Vinf, oper.alpha, xyw[:, i], False)
        uewi[i] = np.dot(v, tw[:, i])
        v, _ = inviscid_velocity(foil.x, isol.gamref[:, 0], Vinf, 0, xyw[:, i], False)
        uewiref[i, 0] = np.dot(v, tw[:, i])
        v, _ = inviscid_velocity(foil.x, isol.gamref[:, 1], Vinf, 90, xyw[:, i], False)
        uewiref[i, 1] = np.dot(v, tw[:, i])

    isol.uewi = uewi
    isol.uewiref = uewiref

    # set values
    wake = Panel()
    wake.N = Nw
    wake.x = xyw
    wake.s = sw
    wake.t = tw
    return wake


def calc_ue_sigma(isol: Isol, foil: Panel, wake: Panel):
    # calculates sensitivity matrix of ue w.r.t. sources
    # INPUT
    #   M : mfoil class with wake already built
    # OUTPUT
    #   M.vsol.sigma_m : d(source)/d(mass) matrix, for computing source strengths
    #   M.vsol.ue_m    : d(ue)/d(mass) matrix, for computing tangential velocity
    # DETAILS
    #   "mass" flow refers to area flow (we exclude density)
    #   sigma_m and ue_m return values at each node (airfoil and wake)
    #   airfoil panel sources are constant strength
    #   wake panel sources are two-piece linear

    assert len(isol.gam) > 0, 'No inviscid solution'
    N, Nw = foil.N, wake.N  # number of points on the airfoil/wake
    assert Nw > 0, 'No wake'

    # Cgam = d(wake uei)/d(gamma)   [Nw x N]   (not sparse)
    Cgam = np.zeros([Nw, N])
    for i in range(Nw):
        v, v_G = inviscid_velocity(foil.x, isol.gam, 0, 0, wake.x[:, i], True)
        Cgam[i, :] = v_G[0, :] * wake.t[0, i] + v_G[1, :] * wake.t[1, i]

    # B = d(airfoil surf streamfunction)/d(source)  [(N+1) x (N+Nw-2)]  (not sparse)
    B = np.zeros([N + 1, N + Nw - 2])  # note, N+Nw-2 = # of panels

    for i in range(N):  # loop over points on the airfoil
        xi = foil.x[:, i]  # coord of point i
        for j in range(N - 1):  # loop over airfoil panels
            B[i, j] = panel_constsource_stream(foil.x[:, [j, j + 1]], xi)
        for j in range(Nw - 1):  # loop over wake panels
            Xj = wake.x[:, [j, j + 1]]  # panel endpoint coordinates
            Xm = 0.5 * (Xj[:, 0] + Xj[:, 1])  # panel midpoint
            Xj = np.transpose(np.vstack((Xj[:, 0], Xm, Xj[:, 1])))  # left, mid, right coords on panel
            if j == (Nw - 2):
                Xj[:, 2] = 2 * Xj[:, 2] - Xj[:, 1]  # ghost extension at last point
            a, b = panel_linsource_stream(Xj[:, [0, 1]], xi)  # left half panel
            if j > 0:
                B[i, N - 1 + j] += 0.5 * a + b
                B[i, N - 1 + j - 1] += 0.5 * a
            else:
                B[i, N - 1 + j] += b
            a, b = panel_linsource_stream(Xj[:, [1, 2]], xi)  # right half panel
            B[i, N - 1 + j] += a + 0.5 * b
            if j < Nw - 2:
                B[i, N - 1 + j + 1] += 0.5 * b
            else:
                B[i, N - 1 + j] += 0.5 * b

    # Bp = - inv(AIC) * B   [N x (N+Nw-2)]  (not sparse)
    # Note, Bp is d(airfoil gamma)/d(source)
    Bp = -np.linalg.solve(isol.AIC, B)  # this has N+1 rows, but the last one is zero
    Bp = Bp[:-1, :]  # trim the last row

    # Csig = d(wake uei)/d(source) [Nw x (N+Nw-2)]  (not sparse)
    Csig = np.zeros([Nw, N + Nw - 2])
    for i in range(Nw):
        xi, ti = wake.x[:, i], wake.t[:, i]  # point, tangent on wake

        # first/last airfoil panel effects on i=0 wake point handled separately
        jstart, jend = 0 + (i == 0), N - 1 - (i == 0)
        for j in range(jstart, jend):  # constant sources on airfoil panels
            Csig[i, j] = panel_constsource_velocity(foil.x[:, [j, j + 1]], xi, ti)

        # piecewise linear sources across wake panel halves (else singular)
        for j in range(Nw):  # loop over wake points
            idx = [max(j - 1, 0), j, min(j + 1, Nw - 1)]  # left, self, right

            Xj = wake.x[:, idx]  # point coordinates
            Xj[:, 0] = 0.5 * (Xj[:, 0] + Xj[:, 1])  # left midpoint
            Xj[:, 2] = 0.5 * (Xj[:, 1] + Xj[:, 2])  # right midpoint

            if j == Nw - 1:
                Xj[:, 2] = 2 * Xj[:, 1] - Xj[:, 0]  # ghost extension at last point
            d1 = norm2(Xj[:, 1] - Xj[:, 0])  # left half-panel length
            d2 = norm2(Xj[:, 2] - Xj[:, 1])  # right half-panel length
            if i == j:
                if j == 0:  # first point: special TE system (three panels meet)
                    dl = norm2(foil.x[:, 1] - foil.x[:, 0])  # lower surface panel length
                    du = norm2(foil.x[:, N - 1] - foil.x[:, N - 2])  # upper surface panel length
                    Csig[i, 0] += (0.5 / np.pi) * (np.log(dl / d2) + 1)  # lower panel effect
                    Csig[i, N - 2] += (0.5 / np.pi) * (np.log(du / d2) + 1)  # upper panel effect
                    Csig[i, N - 1] += -0.5 / np.pi  # self effect
                elif j == Nw - 1:  # last point: no self effect of last pan (ghost extension)
                    Csig[i, N - 1 + j - 1] += 0  # hence the 0
                else:  # all other points
                    aa = (0.25 / np.pi) * np.log(d1 / d2)
                    Csig[i, N - 1 + j - 1] += aa + 0.5 / np.pi
                    Csig[i, N - 1 + j] += aa - 0.5 / np.pi
            else:
                if j == 0:  # first point only has a half panel on the right
                    a, b = panel_linsource_velocity(Xj[:, [1, 2]], xi, ti)
                    Csig[i, N - 1] += b  # right half panel effect
                    Csig[i, 0] += a  # lower airfoil panel effect
                    Csig[i, N - 2] += a  # upper airfoil panel effect
                elif j == Nw - 1:  # last point has a constant source ghost extension
                    a = panel_constsource_velocity(Xj[:, [0, 2]], xi, ti)
                    Csig[i, N + Nw - 3] += a  # full const source panel effect
                else:  # all other points have a half panel on left and right
                    [a1, b1] = panel_linsource_velocity(Xj[:, [0, 1]], xi, ti)  # left half-panel ue contrib
                    [a2, b2] = panel_linsource_velocity(Xj[:, [1, 2]], xi, ti)  # right half-panel ue contrib
                    Csig[i, N - 1 + j - 1] += a1 + 0.5 * b1
                    Csig[i, N - 1 + j] += 0.5 * a2 + b2

    # compute ue_sigma = d(unsigned ue)/d(source) [(N+Nw) x (N+Nw-2)] (not sparse)
    # Df = +/- Bp = d(foil uei)/d(source)  [N x (N+Nw-2)]  (not sparse)
    # Dw = (Cgam*Bp + Csig) = d(wake uei)/d(source)  [Nw x (N+Nw-2)]  (not sparse)
    Dw = np.dot(Cgam, Bp) + Csig
    Dw[0, :] = Bp[-1, :]  # ensure first wake point has same ue as TE
    isol.ue_sigma = np.vstack((Bp, Dw))  # store combined matrix
    # return ue_sigma


def calc_ue_m(isol: Isol, foil: Panel, wake: Panel):
    """Calculates sensitivity matrix of ue w.r.t. transpiration BC mass sources"""
    calc_ue_sigma(isol, foil, wake)

    # initialize sigma_m as lil_matrix (sparse), this is efficient for building
    isol.sigma_m = sparse.lil_matrix((foil.N + wake.N - 2, foil.N + wake.N))  # empty matrix
    # build ue_m from ue_sigma, using sgnue
    rebuild_ue_m(isol, foil, wake)


def rebuild_ue_m(isol: Isol, foil: Panel, wake: Panel):
    # rebuilds ue_m matrix after stagnation panel change (new sgnue)
    # INPUT
    #   M : mfoil class with calc_ue_m already called once
    # OUTPUT
    #   M.vsol.sigma_m : d(source)/d(mass) matrix, for computing source strengths
    #   M.vsol.ue_m    : d(ue)/d(mass) matrix, for computing tangential velocity
    # DETAILS
    #   "mass" flow refers to area flow (we exclude density)
    #   sigma_m and ue_m return values at each node (airfoil and wake)
    #   airfoil panel sources are constant strength
    #   wake panel sources are two-piece linear

    assert len(isol.ue_sigma) > 0, 'Need ue_sigma to build ue_m'

    # sigma_m = d(source)/d(mass)  [(N+Nw-2) x (N+Nw)]  (sparse)
    N, Nw = foil.N, wake.N  # number of points on the airfoil/wake
    isol.sigma_m *= 0.0
    for i in range(N - 1):
        ds = foil.s[i + 1] - foil.s[i]
        # Note, at stagnation: ue = K*s, dstar = const, m = K*s*dstar
        # sigma = dm/ds = K*dstar = m/s (separate for each side, +/-)
        isol.sigma_m[i, [i, i + 1]] = isol.sign_ue[[i, i + 1]] * np.array([-1.0, 1.0]) / ds
    for i in range(Nw - 1):
        ds = wake.s[i + 1] - wake.s[i]
        isol.sigma_m[N - 1 + i, [N + i, N + i + 1]] = np.array([-1.0, 1.0]) / ds

    # convert to csr format, will only do something if not already csr
    isol.sigma_m = isol.sigma_m.tocsr()

    # sign of ue at all points (wake too)
    sgue = np.concatenate((isol.sign_ue, np.ones(Nw)))

    # ue_m = ue_sigma * sigma_m [(N+Nw) x (N+Nw)] (not sparse)
    isol.ue_m = sparse.spdiags(sgue, 0, N + Nw, N + Nw, 'csr') @ isol.ue_sigma @ isol.sigma_m


def panel_info(Xj, xi):
    # calculates common panel properties (distance, angles)
    # INPUTS
    #   Xj    : X(:,[1,2]) = panel endpoint coordinates
    #   xi    : control point coordinates (2x1)
    #   vdir  : direction of dot product, or None
    #   onmid : true means xi is on the panel midpoint
    # OUTPUTS
    #   t, n   : panel-aligned tangent and normal vectors
    #   x, z   : control point coords in panel-aligned coord system
    #   d      : panel length
    #   r1, r2 : distances from panel left/right edges to control point
    #   theta1, theta2 : left/right angles

    # panel coordinates
    xj1, zj1 = Xj[0, 0], Xj[1, 0]
    xj2, zj2 = Xj[0, 1], Xj[1, 1]

    # panel-aligned tangent and normal vectors
    t = np.array([xj2 - xj1, zj2 - zj1])
    t /= norm2(t)
    n = np.array([-t[1], t[0]])

    # control point relative to (xj1,zj1)
    xz = np.array([(xi[0] - xj1), (xi[1] - zj1)])
    x = np.dot(xz, t)  # in panel-aligned coord system
    z = np.dot(xz, n)  # in panel-aligned coord system

    # distances and angles
    def dist(a, b):
        return np.sqrt(a**2 + b**2)

    d = dist(xj2 - xj1, zj2 - zj1)  # panel length
    r1 = dist(x, z)  # left edge to control point
    r2 = dist(x - d, z)  # right edge to control point
    theta1 = atan2(z, x)  # left angle
    theta2 = atan2(z, x - d)  # right angle

    return t, n, x, z, d, r1, r2, theta1, theta2


def panel_linvortex_velocity(Xj, xi, vdir, onmid):
    # calculates the velocity coefficients for a linear vortex panel
    # INPUTS
    #   Xj    : X(:,[1,2]) = panel endpoint coordinates
    #   xi    : control point coordinates (2x1)
    #   vdir  : direction of dot product, or None
    #   onmid : true means xi is on the panel midpoint
    # OUTPUTS
    #   a,b   : velocity influence coefficients of the panel
    # DETAILS
    #   The velocity due to the panel is then a*g1 + b*g2
    #   where g1 and g2 are the vortex strengths at the panel endpoints
    #   If vdir is None, a,b are 2x1 vectors with velocity components
    #   Otherwise, a,b are dotted with vdir

    # panel info
    t, n, x, z, d, r1, r2, theta1, theta2 = panel_info(Xj, xi)

    # velocity in panel-aligned coord system
    if onmid:
        ug1, ug2 = 1 / 2 - 1 / 4, 1 / 4
        wg1, wg2 = -1 / (2 * np.pi), 1 / (2 * np.pi)
    else:
        temp1 = (theta2 - theta1) / (2 * np.pi)
        temp2 = (2 * z * np.log(r1 / r2) - 2 * x * (theta2 - theta1)) / (4 * np.pi * d)
        ug1 = temp1 + temp2
        ug2 = -temp2
        temp1 = np.log(r2 / r1) / (2 * np.pi)
        temp2 = (x * np.log(r1 / r2) - d + z * (theta2 - theta1)) / (2 * np.pi * d)
        wg1 = temp1 + temp2
        wg2 = -temp2

    # velocity influence in original coord system
    a = np.array([ug1 * t[0] + wg1 * n[0], ug1 * t[1] + wg1 * n[1]])  # point 1
    b = np.array([ug2 * t[0] + wg2 * n[0], ug2 * t[1] + wg2 * n[1]])  # point 2
    if vdir is not None:
        a = np.dot(a, vdir)
        b = np.dot(b, vdir)

    return a, b


def panel_linvortex_stream(Xj, xi):
    # calculates the streamfunction coefficients for a linear vortex panel
    # INPUTS
    #   Xj  : X(:,[1,2]) = panel endpoint coordinates
    #   xi  : control point coordinates (2x1)
    # OUTPUTS
    #   a,b : streamfunction influence coefficients
    # DETAILS
    #   The streamfunction due to the panel is then a*g1 + b*g2
    #   where g1 and g2 are the vortex strengths at the panel endpoints

    # panel info
    t, n, x, z, d, r1, r2, theta1, theta2 = panel_info(Xj, xi)

    # check for r1, r2 zero
    ep = 1e-9
    logr1 = 0.0 if (r1 < ep) else np.log(r1)
    logr2 = 0.0 if (r2 < ep) else np.log(r2)

    # streamfunction components
    P1 = (0.5 / np.pi) * (z * (theta2 - theta1) - d + x * logr1 - (x - d) * logr2)
    P2 = x * P1 + (0.5 / np.pi) * (0.5 * r2**2 * logr2 - 0.5 * r1**2 * logr1 - r2**2 / 4 + r1**2 / 4)

    # influence coefficients
    a = P1 - P2 / d
    b = P2 / d

    return a, b


def panel_constsource_velocity(Xj, xi, vdir):
    # calculates the velocity coefficient for a constant source panel
    # INPUTS
    #   Xj    : X(:,[1,2]) = panel endpoint coordinates
    #   xi    : control point coordinates (2x1)
    #   vdir  : direction of dot product, or None
    # OUTPUTS
    #   a     : velocity influence coefficient of the panel
    # DETAILS
    #   The velocity due to the panel is then a*s
    #   where s is the panel source strength
    #   If vdir is None, a,b are 2x1 vectors with velocity components
    #   Otherwise, a,b are dotted with vdir

    # panel info
    t, n, x, z, d, r1, r2, theta1, theta2 = panel_info(Xj, xi)

    ep = 1e-9
    logr1, theta1, theta2 = (0, np.pi, np.pi) if (r1 < ep) else (np.log(r1), theta1, theta2)
    logr2, theta1, theta2 = (0, 0, 0) if (r2 < ep) else (np.log(r2), theta1, theta2)

    # velocity in panel-aligned coord system
    u = (0.5 / np.pi) * (logr1 - logr2)
    w = (0.5 / np.pi) * (theta2 - theta1)

    # velocity in original coord system dotted with given vector
    a = np.array([u * t[0] + w * n[0], u * t[1] + w * n[1]])
    if vdir is not None:
        a = np.dot(a, vdir)

    return a


def panel_constsource_stream(Xj, xi):
    # calculates the streamfunction coefficient for a constant source panel
    # INPUTS
    #   Xj    : X(:,[1,2]) = panel endpoint coordinates
    #   xi    : control point coordinates (2x1)
    # OUTPUTS
    #   a     : streamfunction influence coefficient of the panel
    # DETAILS
    #   The streamfunction due to the panel is then a*s
    #   where s is the panel source strength

    # panel info
    t, n, x, z, d, r1, r2, theta1, theta2 = panel_info(Xj, xi)

    # streamfunction
    ep = 1e-9
    logr1, theta1, theta2 = (0, np.pi, np.pi) if (r1 < ep) else (np.log(r1), theta1, theta2)
    logr2, theta1, theta2 = (0, 0, 0) if (r2 < ep) else (np.log(r2), theta1, theta2)

    P = (x * (theta1 - theta2) + d * theta2 + z * logr1 - z * logr2) / (2 * np.pi)

    dP = d  # delta psi
    P = (P - 0.25 * dP) if ((theta1 + theta2) > np.pi) else (P + 0.75 * dP)

    # influence coefficient
    a = P

    return a


def panel_linsource_velocity(Xj, xi, vdir):
    # calculates the velocity coefficients for a linear source panel
    # INPUTS
    #   Xj    : X(:,[1,2]) = panel endpoint coordinates
    #   xi    : control point coordinates (2x1)
    #   vdir  : direction of dot product
    # OUTPUTS
    #   a,b   : velocity influence coefficients of the panel
    # DETAILS
    #   The velocity due to the panel is then a*s1 + b*s2
    #   where s1 and s2 are the source strengths at the panel endpoints
    #   If vdir is None, a,b are 2x1 vectors with velocity components
    #   Otherwise, a,b are dotted with vdir

    # panel info
    t, n, x, z, d, r1, r2, theta1, theta2 = panel_info(Xj, xi)

    # velocity in panel-aligned coord system
    temp1 = np.log(r1 / r2) / (2 * np.pi)
    temp2 = (x * np.log(r1 / r2) - d + z * (theta2 - theta1)) / (2 * np.pi * d)
    ug1 = temp1 - temp2
    ug2 = temp2
    temp1 = (theta2 - theta1) / (2 * np.pi)
    temp2 = (-z * np.log(r1 / r2) + x * (theta2 - theta1)) / (2 * np.pi * d)
    wg1 = temp1 - temp2
    wg2 = temp2

    # velocity influence in original coord system
    a = np.array([ug1 * t[0] + wg1 * n[0], ug1 * t[1] + wg1 * n[1]])  # point 1
    b = np.array([ug2 * t[0] + wg2 * n[0], ug2 * t[1] + wg2 * n[1]])  # point 2
    if vdir is not None:
        a, b = np.dot(a, vdir), np.dot(b, vdir)

    return a, b


def panel_linsource_stream(Xj, xi):
    # calculates the streamfunction coefficients for a linear source panel
    # INPUTS
    #   Xj  : X(:,[1,2]) = panel endpoint coordinates
    #   xi  : control point coordinates (2x1)
    # OUTPUTS
    #   a,b : streamfunction influence coefficients
    # DETAILS
    #   The streamfunction due to the panel is then a*s1 + b*s2
    #   where s1 and s2 are the source strengths at the panel endpoints

    # panel info
    t, n, x, z, d, r1, r2, theta1, theta2 = panel_info(Xj, xi)

    # make branch cut at theta = 0
    if theta1 < 0:
        theta1 = theta1 + 2 * np.pi
    if theta2 < 0:
        theta2 = theta2 + 2 * np.pi

    # check for r1, r2 zero
    ep = 1e-9
    logr1, theta1, theta2 = (0, np.pi, np.pi) if (r1 < ep) else (np.log(r1), theta1, theta2)
    logr2, theta1, theta2 = (0, 0, 0) if (r2 < ep) else (np.log(r2), theta1, theta2)

    # streamfunction components
    P1 = (0.5 / np.pi) * (x * (theta1 - theta2) + theta2 * d + z * logr1 - z * logr2)
    P2 = x * P1 + (0.5 / np.pi) * (0.5 * r2**2 * theta2 - 0.5 * r1**2 * theta1 - 0.5 * z * d)

    # influence coefficients
    a = P1 - P2 / d
    b = P2 / d

    return a, b
