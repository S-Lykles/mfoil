import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from mfoil.inviscid import Isol, rebuild_ue_m, calc_ue_m, build_wake, build_gamma
from mfoil.utils import vprint, sind, cosd, norm2
from mfoil.geometry import Geom, Panel, TE_info, naca_points, set_coords, make_panels, mgeom_flap, mgeom_addcamber, mgeom_derotate


class Oper:  # operating conditions and flags
    def __init__(self):
        self.Vinf: float = 1.0  # velocity magnitude
        self.alpha: float = 0.0  # angle of attack, in degrees
        self.rho: float = 1.0  # density
        self.cltgt: float = 0.0  # lift coefficient target
        self.givencl = False  # True if cl is given instead of alpha
        self.initbl = True  # True to initialize the boundary layer
        self.viscous = False  # True to do viscous
        self.redowake = False  # True to rebuild wake after alpha changes
        self.Re: float = 1e5  # viscous Reynolds number
        self.Ma: float = 0.0  # Mach number


class Vsol:  # viscous solution variables
    def __init__(self):
        self.th = []  # theta = momentum thickness [Nsys]
        self.ds = []  # delta star = displacement thickness [Nsys]
        self.Is = []  # 3 arrays of surface indices
        self.wgap = []  # wake gap over wake points
        self.ue_m = []  # linearization of ue w.r.t. mass (all nodes)
        self.sigma_m = []  # d(source)/d(mass) matrix
        self.turb = []  # flag over nodes indicating if turbulent (1) or lam (0)
        self.xt: float = 0.0  # transition location (xi) on current surface under consideration
        self.Xt = np.zeros((2, 2))  # transition xi/x for lower and upper surfaces


class Glob:  # global parameters
    def __init__(self):
        self.Nsys = 0  # number of equations and states
        self.U = []  # primary states (th,ds,sa,ue) [4 x Nsys]
        self.conv = True  # converged flag
        self.R = []  # residuals [3*Nsys x 1]
        self.R_U = []  # residual Jacobian w.r.t. primary states
        self.R_x = []  # residual Jacobian w.r.t. xi (s-values) [3*Nsys x Nsys]
        self.R_V = []  # global Jacobian [4*Nsys x 4*Nsys]
        self.realloc = False  # if True, system Jacobians will be re-allocated


class Post:  # post-processing outputs, distributions
    def __init__(self):
        self.cp = []  # cp distribution
        self.cpi = []  # inviscid cp distribution
        self.cl = 0  # lift coefficient
        self.cl_ue = []  # linearization of cl w.r.t. ue [N, airfoil only]
        self.cl_alpha = 0  # linearization of cl w.r.t. alpha
        self.cm = 0  # moment coefficient
        self.cdpi = 0  # near-field pressure drag coefficient
        self.cd = 0  # total drag coefficient
        self.cd_U = []  # linearization of cd w.r.t. last wake state [4]
        self.cdf = 0  # skin friction drag coefficient
        self.cdp = 0  # pressure drag coefficient
        self.rfile = None  # results output file name

        # distributions
        self.th = []  # theta = momentum thickness distribution
        self.ds = []  # delta* = displacement thickness distribution
        self.sa = []  # amplification factor/shear lag coeff distribution
        self.ue = []  # edge velocity (compressible) distribution
        self.uei = []  # inviscid edge velocity (compressible) distribution
        self.cf = []  # skin friction distribution
        self.Ret = []  # Re_theta distribution
        self.Hk = []  # kinematic shape parameter distribution


class Param:  # parameters
    def __init__(self):
        self.verb: int = 1  # printing verbosity level (higher -> more verbose)
        self.rtol: float = 1e-10  # residual tolerance for Newton
        self.niglob: int = 400  # maximum number of global iterations

        # viscous parameters
        self.ncrit: float = 9.0  # critical amplification factor
        self.Cuq: float = 1.0  # scales the uq term in the shear lag equation
        self.Dlr: float = 0.9  # wall/wake dissipation length ratio
        self.SlagK: float = 5.6  # shear lag constant

        # initial Ctau after transition
        self.CtauC: float = 1.8  # Ctau constant
        self.CtauE: float = 3.3  # Ctau exponent

        # G-beta locus: G = GA*sqrt(1+GB*beta) + GC/(H*Rt*sqrt(cf/2))
        self.GA: float = 6.7  # G-beta A constant
        self.GB: float = 0.75  # G-beta B constant
        self.GC: float = 18.0  # G-beta C constant

        # operating conditions and thermodynamics
        self.Minf: float = 0.0  # freestream Mach number
        self.Vinf: float = 0.0  # freestream speed
        self.muinf: float = 0.0  # freestream dynamic viscosity
        self.mu0: float = 0.0  # stagnation dynamic viscosity
        self.rho0: float = 1.0  # stagnation density
        self.H0: float = 0.0  # stagnation enthalpy
        self.Tsrat: float = 0.35  # Sutherland Ts/Tref
        self.gam: float = 1.4  # ratio of specific heats
        self.KTb: float = 1.0  # Karman-Tsien beta = sqrt(1-Minf^2)
        self.KTl: float = 0.0  # Karman-Tsien lambda = Minf^2/(1+KTb)^2
        self.cps: float = 0.0  # sonic cp


class Mfoil:
    def __init__(self, coords=None, naca='0012', npanel=199):
        self.version = '2022-02-22'  # version
        self.geom = Geom()  # geometry
        self.foil = Panel()  # airfoil panels
        self.wake = Panel()  # wake panels
        self.oper = Oper()  # operating conditions
        self.isol: Isol = Isol()  # inviscid solution variables
        self.vsol = Vsol()  # viscous solution variables
        self.glob = Glob()  # global system variables
        self.post = Post()  # post-processing quantities
        self.param = Param()  # parameters
        if coords is not None:
            set_coords(self.geom, coords)
        else:
            self.geom = naca_points(naca)
        self.make_panels(npanel)

    # set operating conditions
    def setoper(self, alpha=None, cl=None, Re=None, Ma=None, visc=None):
        if alpha is not None:
            self.oper.alpha = alpha
        if cl is not None:
            self.oper.cltgt = cl
            self.oper.givencl = True
        if Re is not None:
            self.oper.Re = Re
            self.oper.viscous = True
        if Ma is not None:
            self.oper.Ma = Ma
        if visc is not None:
            if visc is not self.oper.viscous:
                clear_solution(self)
            self.oper.viscous = visc

    def make_panels(self, npanel, stgt=None):
        self.foil = make_panels(self.geom, npanel, stgt)
        clear_solution(self)

    # solve current point
    def solve(self):
        if self.oper.viscous:
            solve_viscous(self)
            calc_force(self)
            get_distributions(self)
        else:
            solve_inviscid(self)
            calc_force(self)

    # geometry functions
    def geom_flap(self, xzhinge, eta):
        self.foil = mgeom_flap(self.geom, self.foil.N, xzhinge, eta)
        clear_solution(self)

    def geom_addcamber(self, zcamb):
        self.foil = mgeom_addcamber(self.geom, self.foil.N, zcamb)  # increment camber

    def geom_derotate(self):
        self.foil = mgeom_derotate(self.geom, self.foil.N)  # derotate: make chordline horizontal


def calc_force(M: Mfoil):
    """
    Calculates force and moment coefficients and updates M.post values

    Parameters
    ----------
    M : Mfoil
        Mfoil structure with solution (inviscid or viscous)

    Notes
    -----
    - Lift and moment are computed from a panel pressure integration
    - The cp distribution is stored as well
    - Accounts for both inviscid and viscous contributions to the force coefficients
    """

    chord = M.geom.chord
    xref = M.geom.xref  # chord and ref moment point
    Vinf = M.param.Vinf
    rho = M.oper.rho
    alpha = M.oper.alpha
    qinf = 0.5 * rho * Vinf**2  # dynamic pressure
    N = M.foil.N  # number of points on the airfoil
    param = M.param

    # calculate the pressure coefficient at each node
    ue = M.glob.U[3, :] if M.oper.viscous else get_ueinv(M)
    cp, cp_ue = get_cp(ue, M.param)
    M.post.cp = cp
    M.post.cpi, cpi_ue = get_cp(get_ueinv(M), M.param)  # inviscid cp

    # lift, moment, near-field pressure cd coefficients by cp integration
    cl, cl_ue, cl_alpha, cm, cdpi = 0, np.zeros(N), 0, 0, 0
    for i0 in range(1, N + 1):
        i, ip = (0, N - 1) if (i0 == N) else (i0, i0 - 1)
        x1, x2 = M.foil.x[:, ip], M.foil.x[:, i]  # panel points
        dxv = x2 - x1
        dx1 = x1 - xref
        dx2 = x2 - xref
        dx1nds = dxv[0] * dx1[0] + dxv[1] * dx1[1]  # (x1-xref) cross n*ds
        dx2nds = dxv[0] * dx2[0] + dxv[1] * dx2[1]  # (x2-xref) cross n*ds
        dx = -dxv[0] * cosd(alpha) - dxv[1] * sind(alpha)  # minus from CW node ordering
        dz = dxv[1] * cosd(alpha) - dxv[0] * sind(alpha)  # for drag
        cp1, cp2 = cp[ip], cp[i]
        cpbar = 0.5 * (cp1 + cp2)  # average cp on the panel
        cl = cl + dx * cpbar
        idx = [ip, i]
        cl_ue[idx] += dx * 0.5 * cp_ue[idx]
        cl_alpha += cpbar * (sind(alpha) * dxv[0] - cosd(alpha) * dxv[1]) * np.pi / 180
        cm += cp1 * dx1nds / 3 + cp1 * dx2nds / 6 + cp2 * dx1nds / 6 + cp2 * dx2nds / 3
        cdpi = cdpi + dz * cpbar
        cl /= chord
        cm /= chord**2
        cdpi /= chord
        M.post.cl, M.post.cl_ue, M.post.cl_alpha = cl, cl_ue, cl_alpha
        M.post.cm, M.post.cdpi = cm, cdpi

    # viscous contributions
    cd, cdf = 0, 0
    if M.oper.viscous:
        # Squire-Young relation for total drag (extrapolates theta from end of wake)
        iw = M.vsol.Is[2][-1]  # station at the end of the wake
        U = M.glob.U[:, iw]
        H, H_U = get_H(U)
        uk, uk_ue = get_uk(U[3], M.param)
        cd = 2.0 * U[0] * (uk / Vinf) ** ((5 + H) / 2.0)
        M.post.cd_U = 2.0 * U[0] * (uk / Vinf) ** ((5 + H) / 2.0) * np.log(uk / Vinf) * 0.5 * H_U
        M.post.cd_U[0] += 2.0 * (uk / Vinf) ** ((5 + H) / 2.0)
        M.post.cd_U[3] += 2.0 * U[0] * (5 + H) / 2.0 * (uk / Vinf) ** ((3 + H) / 2.0) * (1.0 / Vinf) * uk_ue

        # skin friction drag
        Df = 0.0
        for si in range(2):
            Is = M.vsol.Is[si]  # surface point indices
            cf1 = 0  # first cf value
            ue1 = 0
            rho1 = rho
            x1 = M.isol.xstag
            for i in range(len(Is)):
                turb = M.vsol.turb[Is[i]]
                cf2, cf2_U = get_cf(M.glob.U[:, Is[i]], param, turb, wake=False)  # get cf value
                ue2, ue2_ue = get_uk(M.glob.U[3, Is[i]], param)
                rho2, rho2_U = get_rho(M.glob.U[:, Is[i]], param)
                x2 = M.foil.x[:, Is[i]]
                dxv = x2 - x1
                dx = dxv[0] * cosd(alpha) + dxv[1] * sind(alpha)
                Df += 0.25 * (rho1 * cf1 * ue1**2 + rho2 * cf2 * ue2**2) * dx
                cf1 = cf2
                ue1 = ue2
                x1 = x2
                rho1 = rho2
        cdf = Df / (qinf * chord)

    # store results
    M.post.cd, M.post.cdf, M.post.cdp = cd, cdf, cd - cdf

    # print out current values
    s = f'  alpha={M.oper.alpha:.2f}deg, cl={M.post.cl:.6f}, cm={M.post.cm:.6f}, cdpi={M.post.cdpi:.6f}, cd={M.post.cd:.6f}, cdf={M.post.cdf:.6f}, cdp={M.post.cdp:.6f}'
    vprint(M.param.verb, 1, s)


def get_distributions(M: Mfoil):
    """
    Computes various distributions (quantities at nodes) and stores them in M.post
    Sets M.post with distribution quantities calculated

    Parameters
    ----------
    M : Mfoil
        Mfoil class with a valid solution in M.glob.U

    Notes
    -----
    - Relevant for viscous solutions
    """

    assert M.glob.U is not None, 'no global solution'

    # quantities already in the global state
    M.post.th = M.glob.U[0, :].copy()  # theta
    M.post.ds = M.glob.U[1, :].copy()  # delta*
    M.post.sa = M.glob.U[2, :].copy()  # amp or ctau
    M.post.ue, uk_ue = get_uk(M.glob.U[3, :], M.param)  # compressible edge velocity
    M.post.uei = get_ueinv(M)  # compressible inviscid edge velocity

    param = M.param

    # derived viscous quantities
    N = M.glob.Nsys
    cf = np.zeros(N)
    Ret = np.zeros(N)
    Hk = np.zeros(N)
    for si in range(3):  # loop over surfaces
        Is = M.vsol.Is[si]  # surface point indices
        wake = si == 2
        for i in range(len(Is)):  # loop over points
            j = Is[i]
            Uj = M.glob.U[:, j]
            turb = M.vsol.turb[j]
            uk, uk_ue = get_uk(Uj[3], param)  # corrected edge speed
            cfloc, cf_u = get_cf(Uj, param, turb, wake)  # local skin friction coefficient
            cf[j] = cfloc * uk * uk / (param.Vinf * param.Vinf)  # free-stream-based cf
            Ret[j], Ret_U = get_Ret(Uj, param)  # Re_theta
            Hk[j], Hk_U = get_Hk(Uj, param)  # kinematic shape factor

    M.post.cf, M.post.Ret, M.post.Hk = cf, Ret, Hk


# ============ INVISCID FUNCTIONS ==============
def clear_solution(M: Mfoil):
    """
    Clears inviscid/viscous solutions by re-initializing structures
    Modifies M to remove inviscid or viscous solution

    Parameters
    ----------
    M : Mfoil
        Mfoil structure
    """

    M.isol = Isol()
    M.vsol = Vsol()
    M.glob = Glob()
    M.post = Post()
    M.wake.N = 0
    M.wake.x = []
    M.wake.s = []
    M.wake.t = []


def solve_inviscid(M: Mfoil):
    """
    Solves the inviscid system, rebuilds 0,90deg solutions
    Computes inviscid vorticity distribution

    Parameters
    ----------
    M : Mfoil
        Mfoil structure

    Notes
    -----
    - Uses the angle of attack in M.oper.alpha
    - Also initializes thermo variables for normalization
    """

    assert M.foil.N > 0, 'No panels'
    init_thermo(M)
    M.isol = build_gamma(M.foil, M.param, M.oper.alpha, M.geom.chord)
    M.glob.conv = True  # no coupled system ... convergence is guaranteed


def get_ueinv(M: Mfoil):
    """
    Computes inviscid tangential velocity at every node
    Returns ueinv

    Parameters
    ----------
    M : Mfoil
        Mfoil structure

    Returns
    -------
    ueinv : np.ndarray
        Inviscid velocity at airfoil and wake (if exists) points

    Notes
    -----
    - The airfoil velocity is computed directly from gamma
    - The tangential velocity is measured + in the streamwise direction
    """

    assert len(M.isol.gam) > 0, 'No inviscid solution'
    alpha = M.oper.alpha
    cs = np.array([cosd(alpha), sind(alpha)])
    uea = M.isol.sign_ue * np.dot(M.isol.gamref, cs)  # airfoil
    if (M.oper.viscous) and (M.wake.N > 0):
        uew = np.dot(M.isol.uewiref, cs)  # wake
        uew[0] = uea[-1]  # ensures continuity of upper surface and wake ue
    else:
        uew = np.array([])
    ueinv = np.concatenate((uea, uew))  # airfoil/wake edge velocity
    return ueinv.transpose()


def get_ueinvref(M: Mfoil) -> NDArray[np.float64]:
    """
    Computes 0,90deg inviscid tangential velocities at every node

    Parameters
    ----------
    M : Mfoil
        Mfoil structure containing the aerodynamic model and data

    Returns
    -------
    ueinvref : (N+Nw, 2) ndarray
        0,90 inviscid tangential velocity at all points, including both airfoil (N) and wake (Nw) nodes

    Notes
    -----
    - Utilizes `gamref` for the airfoil to calculate inviscid tangential velocities
    - Uses `uewiref` for the wake (if exists) to compute wake contributions
    """

    assert len(M.isol.gam) > 0, 'No inviscid solution'
    uearef = M.isol.sign_ue[:, np.newaxis] * M.isol.gamref
    if (M.oper.viscous) and (M.wake.N > 0):
        uewref = M.isol.uewiref  # wake
        uewref[0, :] = uearef[-1, :]  # continuity of upper surface and wake
        return np.concatenate((uearef, uewref))
    else:
        return uearef


def init_thermo(M: Mfoil):
    # initializes thermodynamics variables in param structure
    # INPUT
    #   M  : mfoil class with oper structure set
    # OUTPUT
    #   M.param fields filled in based on M.oper
    #   Gets ready for compressibilty corrections if M.oper.Ma > 0

    g = M.param.gam
    gmi = g - 1
    rhoinf = M.oper.rho  # freestream density
    Vinf = M.oper.Vinf
    M.param.Vinf = Vinf  # freestream speed
    M.param.muinf = rhoinf * Vinf * M.geom.chord / M.oper.Re  # freestream dyn viscosity
    Minf = M.oper.Ma
    M.param.Minf = Minf  # freestream Mach
    if Minf > 0:
        M.param.KTb = np.sqrt(1 - Minf**2)  # Karman-Tsien beta
        M.param.KTl = Minf**2 / (1 + M.param.KTb) ** 2  # Karman-Tsien lambda
        M.param.H0 = (1 + 0.5 * gmi * Minf**2) * Vinf**2 / (gmi * Minf**2)  # stagnation enthalpy
        Tr = 1 - 0.5 * Vinf**2 / M.param.H0  # freestream/stagnation temperature ratio
        finf = Tr**1.5 * (1 + M.param.Tsrat) / (Tr + M.param.Tsrat)  # Sutherland's ratio
        M.param.cps = 2 / (g * Minf**2) * (((1 + 0.5 * gmi * Minf**2) / (1 + 0.5 * gmi)) ** (g / gmi) - 1)
    else:
        finf = 1  # incompressible case

    M.param.mu0 = M.param.muinf / finf  # stag visc (Sutherland ref temp is stag)
    M.param.rho0 = rhoinf * (1 + 0.5 * gmi * Minf**2) ** (1 / gmi)  # stag density


def identify_surfaces(M: Mfoil):
    # identifies lower/upper/wake surfaces
    # INPUT
    #   M  : mfoil class with stagnation point found
    # OUTPUT
    #   M.vsol.Is : list of of node indices for lower(1), upper(2), wake(3)

    M.vsol.Is = [
        range(M.isol.stag_idx[0], -1, -1),
        range(M.isol.stag_idx[1], M.foil.N),
        range(M.foil.N, M.foil.N + M.wake.N),
    ]


def set_wake_gap(M: Mfoil):
    # sets height (delta*) of dead air in wake
    # INPUT
    #   M  : mfoil class with wake built and stagnation point found
    # OUTPUT
    #   M.vsol.wgap : wake gap at each wake point
    # DETAILS
    #   Uses cubic function to extrapolate the TE gap into the wake
    #   See Drela, IBL for Blunt Trailing Edges, 1989, 89-2166-CP

    t, hTE, dtdx, tcp, tdp = TE_info(M.foil.x)  # trailing-edge info
    flen = 2.5  # length-scale factor
    dtdx = min(max(dtdx, -3.0 / flen), 3.0 / flen)  # clip TE thickness slope
    Lw = flen * hTE
    wgap = np.zeros(M.wake.N)
    for i in range(M.wake.N):
        xib = (M.isol.xi[M.foil.N + i] - M.isol.xi[M.foil.N]) / Lw
        if xib <= 1:
            wgap[i] = hTE * (1 + (2 + flen * dtdx) * xib) * (1 - xib) ** 2
    M.vsol.wgap = wgap


def stagpoint_find(isol: Isol, foil: Panel, wake: Panel):
    # finds the LE stagnation point on the airfoil (using inviscid solution)
    # INPUTS
    #   M  : mfoil class with inviscid solution, gam
    # OUTPUTS
    #   M.isol.sstag   : scalar containing s value of stagnation point
    #   M.isol.sstag_g : linearization of sstag w.r.t gamma (1xN)
    #   M.isol.Istag   : [i,i+1] node indices before/after stagnation (1x2)
    #   M.isol.sgnue   : sign conversion from CW to tangential velocity (1xN)
    #   M.isol.xi      : distance from stagnation point at each node (1xN)

    assert len(isol.gam) > 0, 'No inviscid solution'
    N = foil.N  # number of points on the airfoil
    j = 0
    for j in range(N):
        if isol.gam[j] > 0:
            break
    else:
        assert False, 'no stagnation point'
    idx = [j - 1, j]
    G = isol.gam[idx]
    S = foil.s[idx]
    isol.stag_idx = idx  # indices of neighboring gammas
    den = G[1] - G[0]
    w1 = G[1] / den
    w2 = -G[0] / den
    isol.sstag = w1 * S[0] + w2 * S[1]  # s location
    isol.xstag = foil.x[:, j - 1] * w1 + foil.x[:, j] * w2  # x location
    st_g1 = G[1] * (S[0] - S[1]) / (den * den)
    isol.sstag_g = np.array([st_g1, -st_g1])
    isol.sign_ue = np.sign(isol.gam)
    isol.xi = np.concatenate((abs(foil.s - isol.sstag), wake.s - isol.sstag))


def stagpoint_move(M: Mfoil):
    # moves the LE stagnation point on the airfoil using the global solution ue
    # INPUT
    #   M  : mfoil class with a valid solution in M.glob.U
    # OUTPUT
    #   New sstag, sstag_ue, xi in M.isol
    #   Possibly new stagnation panel, Istag, and hence new surfaces and matrices

    N = M.foil.N  # number of points on the airfoil
    idx = M.isol.stag_idx  # current adjacent node indices
    ue = M.glob.U[3, :]  # edge velocity
    sstag0 = M.isol.sstag  # original stag point location

    newpanel = True  # are we moving to a new panel?
    if ue[idx[1]] < 0:
        # move stagnation point up (larger s, new panel)
        vprint(M.param.verb, 2, '  Moving stagnation point up')
        for j in range(idx[1], N):
            if ue[j] > 0:
                break
        assert j < N, 'no stagnation point'
        I1 = j
        for j in range(idx[1], I1):
            ue[j] *= -1.0
        idx[0], idx[1] = I1 - 1, I1  # new panel
    elif ue[idx[0]] < 0:
        # move stagnation point down (smaller s, new panel)
        vprint(M.param.verb, 2, '  Moving stagnation point down')
        for j in range(idx[0], -1, -1):
            if ue[j] > 0:
                break
        assert j > 0, 'no stagnation point'
        I0 = j
        for j in range(I0 + 1, idx[0] + 1):
            ue[j] *= -1.0
        idx[0], idx[1] = I0, I0 + 1  # new panel
    else:
        newpanel = False  # staying on the current panel

    # move point along panel
    ues, S = ue[idx], M.foil.s[idx]
    assert (ues[0] > 0) and (ues[1] > 0), 'stagpoint_move: velocity error'
    den = ues[0] + ues[1]
    w1 = ues[1] / den
    w2 = ues[0] / den
    M.isol.sstag = w1 * S[0] + w2 * S[1]  # s location
    M.isol.xstag = np.dot(M.foil.x[:, idx], np.r_[w1, w2])  # x location
    M.isol.sstag_ue = np.r_[ues[1], -ues[0]] * (S[1] - S[0]) / (den * den)
    vprint(M.param.verb, 2, f'Moving stagnation point: s={sstag0:.15e} -> s={M.isol.sstag:.15e}')

    # set new xi coordinates for every point
    M.isol.xi = np.concatenate((abs(M.foil.s - M.isol.sstag), M.wake.s - M.isol.sstag))

    # matrices need to be recalculated if on a new panel
    if newpanel:
        vprint(M.param.verb, 2, f'  New stagnation panel = {idx[0]} {idx[1]}')
        M.isol.stag_idx = idx  # new panel indices
        for i in range(idx[0] + 1):
            M.isol.sign_ue[i] = -1
        for i in range(idx[0] + 1, N):
            M.isol.sign_ue[i] = 1
        identify_surfaces(M)  # re-identify surfaces
        M.glob.U[3, :] = ue  # sign of ue changed on some points near stag
        M.glob.realloc = True
        rebuild_ue_m(M.isol, M.foil, M.wake)


def solve_viscous(M: Mfoil):
    # solves the viscous system (BL + outer flow concurrently)
    # INPUT
    #   M  : mfoil class with an airfoil
    # OUTPUT
    #   M.glob.U : global solution
    #   M.post   : post-processed quantities

    solve_inviscid(M)
    M.oper.viscous = True
    init_thermo(M)
    M.wake = build_wake(M.isol, M.foil, M.oper, M.geom.wakelen, M.geom.chord)
    stagpoint_find(M.isol, M.foil, M.wake)  # from the inviscid solution
    identify_surfaces(M)
    set_wake_gap(M)  # blunt TE dead air extent in wake
    calc_ue_m(M.isol, M.foil, M.wake)
    init_boundary_layer(M)  # initialize boundary layer from ue
    stagpoint_move(M)  # move stag point, using viscous solution
    solve_coupled(M)  # solve coupled system


def solve_coupled(M: Mfoil):
    # Solves the coupled inviscid and viscous system
    # INPUT
    #   M  : mfoil class with an inviscid solution
    # OUTPUT
    #   M.glob.U : global coupled solution
    # DETAILS
    #   Inviscid solution should exist, and BL variables should be initialized
    #   The global variables are [th, ds, sa, ue] at every node
    #   th = momentum thickness
    #   ds = displacement thickness
    #   sa = amplification factor or sqrt(ctau)
    #   ue = edge velocity
    #   Nsys = N + Nw = total number of unknowns
    #   ue is treated as a separate variable for improved solver robustness
    #   The alternative is to eliminate ue, ds and use mass flow (not done here):
    #     Starting point: ue = uinv + D*m -> ue_m = D
    #     Since m = ue*ds, we have ds = m/ue = m/(uinv + D*m)
    #     So, ds_m = diag(1/ue) - diag(ds/ue)*D
    #     The residual linearization is then: R_m = R_ue*ue_m + R_ds*ds_m

    # Newton loop
    nNewton = M.param.niglob  # number of iterations
    M.glob.conv = False
    M.glob.realloc = True  # reallocate Jacobian on first iter
    vprint(M.param.verb, 1, '\n <<< Beginning coupled solver iterations >>>')

    for iNewton in range(nNewton):
        # set up the global system
        vprint(M.param.verb, 2, 'Building global system')
        build_glob_sys(M)

        # compute forces if in cl target mode
        if M.oper.givencl:
            vprint(M.param.verb, 2, 'Calculating force')
            calc_force(M)

        # convergence check
        Rnorm = norm2(M.glob.R)
        vprint(M.param.verb, 1, f'\nNewton iteration {iNewton}, Rnorm = {Rnorm:.10e}')
        if Rnorm < M.param.rtol:
            M.glob.conv = True
            break

        # solve global system
        vprint(M.param.verb, 2, 'Solving global system')
        dU, dalpha = solve_glob(M)

        # update the state
        vprint(M.param.verb, 2, 'Updating the state')
        update_state(M, dU, dalpha)

        M.glob.realloc = False  # assume Jacobian will not get reallocated

        # update stagnation point; Newton still OK; had R_x effects in R_U
        vprint(M.param.verb, 2, 'Moving stagnation point')
        stagpoint_move(M)

        # update transition
        vprint(M.param.verb, 2, 'Updating transition')
        update_transition(M)

    if not M.glob.conv:
        vprint(M.param.verb, 1, '\n** Global Newton NOT CONVERGED **\n')


def update_state(M: Mfoil, dU, dalpha: float):
    # updates state, taking into account physical constraints
    # INPUT
    #   M  : mfoil class with a valid solution (U) and proposed update (dU)
    # OUTPUT
    #   M.glob.U : updated solution, possibly with a fraction of dU added
    # DETAILS
    #   U = U + omega * dU
    #   omega = under-relaxation factor
    #   Calculates omega to prevent big changes in the state or negative values

    if any(np.iscomplex(M.glob.U[2, :])):
        raise ValueError('imaginary amp in U')
    if any(np.iscomplex(dU[2, :])):
        raise ValueError('imaginary amp in dU')

    # max ctau
    It = np.nonzero(M.vsol.turb)[0]
    ctmax = max(M.glob.U[2, It])

    # starting under-relaxation factor
    omega = 1.0

    # first limit theta and delta*
    for k in range(2):
        Uk = M.glob.U[k, :]
        dUk = dU[k, :]
        # prevent big decreases in th, ds
        fmin = min(dUk / Uk)  # find most negative ratio
        om = abs(0.5 / fmin) if (fmin < -0.5) else 1.0
        if om < omega:
            omega = om
            vprint(M.param.verb, 3, f'  th/ds decrease: omega = {omega:.5f}')

    # limit negative amp/ctau
    Uk = M.glob.U[2, :]
    dUk = dU[2, :]
    for i in range(len(Uk)):
        if (not M.vsol.turb[i]) and (Uk[i] < 0.2):
            continue  # do not limit very small amp (too restrictive)
        if (M.vsol.turb[i]) and (Uk[i] < 0.1 * ctmax):
            continue  # do not limit small ctau
        if (Uk[i] == 0.0) or (dUk[i] == 0.0):
            continue
        if Uk[i] + dUk[i] < 0:
            om = 0.8 * abs(Uk[i] / dUk[i])
            if om < omega:
                omega = om
                vprint(M.param.verb, 3, f'  neg sa: omega = {omega:.5f}')

    # prevent big changes in amp
    idx = np.nonzero(M.vsol.turb)[0]
    if any(np.iscomplex(Uk[idx])):
        raise ValueError('imaginary amplification')
    dumax = max(abs(dUk[idx]))
    om = abs(2.0 / dumax) if (dumax > 0) else 1.0
    if om < omega:
        omega = om
        vprint(M.param.verb, 3, f'  amp: omega = {omega:.5f}')

    # prevent big changes in ctau
    idx = np.nonzero(M.vsol.turb)[0]
    dumax = max(abs(dUk[idx]))
    om = abs(0.05 / dumax) if (dumax > 0) else 1.0
    if om < omega:
        omega = om
        vprint(M.param.verb, 3, f'  ctau: omega = {omega:.5f}')

    # prevent large ue changes
    dUk = dU[3, :]
    fmax = max(abs(dUk) / M.oper.Vinf)
    om = 0.2 / fmax if (fmax > 0) else 1.0
    if om < omega:
        omega = om
        vprint(M.param.verb, 3, f'  ue: omega = {omega:.5f}')

    # prevent large alpha changes
    if abs(dalpha) > 2:
        omega = min(omega, abs(2 / dalpha))

    # take the update
    vprint(M.param.verb, 2, f'  state update: under-relaxation = {omega:.5f}')
    M.glob.U += omega * dU
    M.oper.alpha += omega * dalpha

    # fix bad Hk after the update
    for side in range(3):  # loop over surfaces
        Hkmin = 1.00005 if (side == 2) else 1.02
        Is = M.vsol.Is[side]  # surface point indices
        for i in range(len(Is)):  # loop over points
            j = Is[i]
            Uj = M.glob.U[:, j]
            Hk, Hk_U = get_Hk(Uj, M.param)
            if Hk < Hkmin:
                M.glob.U[1, j] += 2 * (Hkmin - Hk) * M.glob.U[1, j]

    # fix negative ctau after the update
    for ii in range(len(It)):
        i = It[ii]
        if M.glob.U[2, i] < 0:
            M.glob.U[2, i] = 0.1 * ctmax

    def rebuild_isol(M: Mfoil):
        """rebuilds inviscid solution, after an angle of attack change"""

        assert len(M.isol.gam) > 0, 'No inviscid solution'
        vprint(M.param.verb, 2, '\n  Rebuilding the inviscid solution.')
        alpha = M.oper.alpha
        M.isol.gam = M.isol.gamref[:, 0] * cosd(alpha) + M.isol.gamref[:, 1] * sind(alpha)
        if not M.oper.viscous:
            stagpoint_find(M.isol, M.foil, M.wake)  # viscous stag point movement is handled separately
        elif M.oper.redowake:
            M.wake = build_wake(M.isol, M.foil, M.oper, M.geom.wakelen, M.geom.chord)
            identify_surfaces(M)
            calc_ue_m(M.isol, M.foil, M.wake)  # rebuild matrices due to changed wake geometry

    # rebuild inviscid solution (gam, wake) if angle of attack changed
    if abs(omega * dalpha) > 1e-10:
        rebuild_isol(M)


def solve_glob(M: Mfoil):
    # solves global system for the primary variable update dU
    # INPUT
    #   M  : mfoil class with residual and Jacobian calculated
    # OUTPUT
    #   M.glob.dU : proposed solution update
    # DETAILS
    #   Uses the augmented system: fourth residual = ue equation
    #   Supports lift-constrained mode, with an extra equation: cl - cltgt = 0
    #   Extra variable in cl-constrained mode is angle of attack
    #   Solves sparse matrix system for for state/alpha update

    Nsys = M.glob.Nsys  # number of dofs
    docl = M.oper.givencl  # 1 if in cl-constrained mode

    # get edge velocity and displacement thickness
    ue = M.glob.U[3, :]
    ds = M.glob.U[1, :]
    uemax = max(abs(ue))
    for i in range(len(ue)):
        ue[i] = max(ue[i], 1e-10 * uemax)  # avoid 0/negative ue

    # use augmented system: variables = th, ds, sa, ue

    # inviscid edge velocity on the airfoil and wake
    ueinv = get_ueinv(M)

    # initialize the global variable Jacobian
    NN = 4 * Nsys + docl
    if (M.glob.realloc) or isinstance(M.glob.R_V, list) or not (M.glob.R_V.shape == (NN, NN)):
        alloc_R_V = True
        M.glob.R_V = sparse.lil_matrix((NN, NN))  # +1 for cl-alpha constraint
    else:
        alloc_R_V = False
        M.glob.R_V *= 0.0  # matrix already allocated, just zero it out

    # state indices in the global system
    Ids = slice(1, 4 * Nsys, 4)  # delta star indices
    Iue = slice(3, 4 * Nsys, 4)  # ue indices

    # assemble the residual
    R = np.concatenate((M.glob.R, ue - (ueinv + M.isol.ue_m @ (ds * ue))))
    # print('first Norm(R) = %.10e'%(norm2(R)))

    # assemble the Jacobian
    M.glob.R_V[0 : 3 * Nsys, 0 : 4 * Nsys] = M.glob.R_U
    idx = slice(3 * Nsys, 4 * Nsys, 1)
    M.glob.R_V[idx, Iue] = sparse.identity(Nsys) - M.isol.ue_m @ np.diag(ds)
    M.glob.R_V[idx, Ids] = -M.isol.ue_m @ np.diag(ue)

    if docl:
        # include cl-alpha residual and Jacobian
        Rcla, Ru_alpha, Rcla_U = clalpha_residual(M)
        R = np.concatenate((R, np.array([Rcla])))
        M.glob.R_V[idx, 4 * Nsys] = Ru_alpha
        M.glob.R_V[4 * Nsys, :] = Rcla_U

    # solve system for dU, dalpha
    if alloc_R_V:
        M.glob.R_V = M.glob.R_V.tocsr()
    dV = -sparse.linalg.spsolve(M.glob.R_V, R)

    # store dU, reshaped, in M
    dU = np.reshape(dV[0 : 4 * Nsys], (4, Nsys), order='F')
    if docl:
        return dU, dV[-1]
    else:
        return dU, 0


def clalpha_residual(M: Mfoil):
    # computes cl constraint (or just prescribed alpha) residual and Jacobian
    # INPUT
    #   M  : mfoil class with inviscid solution and post-processed cl_alpha, cl_ue
    # OUTPUT
    #   Rcla     : cl constraint residual = cl - cltgt (scalar)
    #   Ru_alpha : lin of ue residual w.r.t. alpha (Nsys x 1)
    #   Rcla_U   : lin of cl residual w.r.t state (1 x 4*Nsys)
    # DETAILS
    #   Used for cl-constrained mode, with alpha as the extra variable
    #   Should be called with up-to-date cl and cl linearizations

    Nsys = M.glob.Nsys  # number of dofs
    N = M.foil.N  # number of points (dofs) on airfoil
    alpha = M.oper.alpha  # angle of attack (deg)

    if M.oper.givencl:  # cl is prescribed, need to trim alpha
        Rcla = M.post.cl - M.oper.cltgt  # cl constraint residual
        Rcla_U = np.zeros(4 * Nsys + 1)
        Rcla_U[-1] = M.post.cl_alpha
        Rcla_U[3 : 4 * N : 4] = M.post.cl_ue  # only airfoil nodes affected
        # Ru = ue - [uinv + ue_m*(ds.*ue)], uinv = uinvref*[cos(alpha);sin(alpha)]
        Ru_alpha = -get_ueinvref(M) @ np.r_[-sind(alpha), cosd(alpha)] * np.pi / 180
    else:  # alpha is prescribed, easy
        Rcla = 0  # no residual
        Ru_alpha = np.zeros(Nsys, 1)  # not really, but alpha is not changing
        Rcla_U = np.zeros(4 * Nsys + 1)
        Rcla_U[-1] = 1

    return Rcla, Ru_alpha, Rcla_U


def build_glob_sys(M: Mfoil):
    # builds the primary variable global residual system for the coupled problem
    # INPUT
    #   M  : mfoil class with a valid solution in M.glob.U
    # OUTPUT
    #   M.glob.R   : global residual vector (3*Nsys x 1)
    #   M.glob.R_U : residual Jacobian matrix (3*Nsys x 4*Nsys, sparse)
    #   M.glob.R_x : residual linearization w.r.t. x (3*Nsys x Nsys, sparse)
    # DETAILS
    #   Loops over nodes/stations to assemble residual and Jacobian
    #   Transition dicated by M.vsol.turb, which should be consistent with the state
    #   Accounts for wake initialization and first-point similarity solutions
    #   Also handles stagnation point on node via simple extrapolation

    Nsys = M.glob.Nsys

    # allocate matrices if [], if size changed, or if global realloc flag is true
    if (M.glob.realloc) or isinstance(M.glob.R, list) or not (M.glob.R.shape[0] == 3 * Nsys):
        M.glob.R = np.zeros(3 * Nsys)
    else:
        M.glob.R *= 0.0
    if (M.glob.realloc) or isinstance(M.glob.R_U, list) or not (M.glob.R_U.shape == (3 * Nsys, 4 * Nsys)):
        alloc_R_U = True
        M.glob.R_U = sparse.lil_matrix((3 * Nsys, 4 * Nsys))
    else:
        alloc_R_U = False
        M.glob.R_U *= 0.0

    if (M.glob.realloc) or isinstance(M.glob.R_x, list) or not (M.glob.R_x == (3 * Nsys, Nsys)):
        alloc_R_x = True
        M.glob.R_x = sparse.lil_matrix((3 * Nsys, Nsys))
    else:
        alloc_R_x = False
        M.glob.R_x *= 0.0

    for si in range(3):  # loop over surfaces
        Is = M.vsol.Is[si]  # surface point indices
        xi = M.isol.xi[Is]  # distance from LE stag point
        N = len(Is)  # number of points on this surface
        U = M.glob.U[:, Is]  # [th, ds, sa, ue] states at all points on this surface
        Aux = np.zeros(N)  # auxiliary data at all points: [wgap]

        # set auxiliary data
        if si == 2:
            Aux[:] = M.vsol.wgap

        # special case of tiny first xi -- will set to stagnation state later
        i0 = 1 if (si < 2) and (xi[0] < 1e-8 * xi[-1]) else 0

        # first point system
        if si < 2:
            # calculate the stagnation state, a function of U1 and U2
            Ip = [i0, i0 + 1]
            Ust, Ust_U, Ust_x, xst = stagnation_state(U[:, Ip], xi[Ip])  # stag state
            turb, wake, simi = False, False, True  # similarity station flag
            R1, R1_Ut, R1_x = residual_station(M.param, np.r_[xst, xst], np.stack((Ust, Ust), axis=-1), Aux[[i0, i0]], turb, wake, simi)
            simi = False
            R1_Ust = R1_Ut[:, 0:4] + R1_Ut[:, 4:8]
            R1_U = np.dot(R1_Ust, Ust_U)
            R1_x = np.dot(R1_Ust, Ust_x)
            J = [Is[i0], Is[i0 + 1]]

            if i0 == 1:
                # i0=0 point landed right on stagnation: set value to Ust
                vprint(M.param.verb, 2, 'hit stagnation!')
                Ig = slice(3 * Is[0], 3 * Is[0] + 3)
                M.glob.R[Ig] = U[0:3, 0] - Ust[0:3]
                M.glob.R_U[Ig, 4 * Is[0] : (4 * Is[0] + 4)] += np.eye(3, 4)
                M.glob.R_U[Ig, 4 * J[0] : (4 * J[0] + 4)] -= Ust_U[0:3, 0:4]
                M.glob.R_U[Ig, 4 * J[1] : (4 * J[1] + 4)] -= Ust_U[0:3, 4:8]
                M.glob.R_x[Ig, J] = -Ust_x[0:3, :]

        else:
            # wake initialization
            R1, R1_U, J = wake_sys(M, M.param)
            R1_x = []  # no xi dependence of first wake residual
            # force turbulent in wake if still laminar
            turb, wake = (
                True,
                True,
            )

        # store first point system in global residual, Jacobian
        Ig = slice(3 * Is[i0], 3 * Is[i0] + 3)
        M.glob.R[Ig] = R1
        if alloc_R_U:
            R1_U += 1e-15  # hack: force lil sparse format to allocate
        if (alloc_R_x) and (len(R1_x) > 0):
            R1_x += 1e-15  # hack: force lil sparse format to allocate
        for j in range(len(J)):
            M.glob.R_U[Ig, 4 * J[j] : (4 * J[j] + 4)] += R1_U[:, 4 * j : (4 * j + 4)]
            if len(R1_x) > 0:
                M.glob.R_x[Ig, J[j]] += R1_x[:, j : (j + 1)]

        # march over rest of points
        for i in range(i0 + 1, N):
            Ip = [i - 1, i]  # two points involved in the calculation

            tran = M.vsol.turb[Is[i - 1]] ^ M.vsol.turb[Is[i]]  # transition flag

            # residual, Jacobian for point i
            if tran:
                Ri, Ri_U, Ri_x = residual_transition(M, M.param, xi[Ip], U[:, Ip], Aux[Ip], wake, simi)
                store_transition(M, si, i)
            else:
                Ri, Ri_U, Ri_x = residual_station(M.param, xi[Ip], U[:, Ip], Aux[Ip], turb, wake, simi)

            # store point i contribution in global residual, Jacobian
            Ig = slice(3 * Is[i], 3 * Is[i] + 3)
            if alloc_R_U:
                Ri_U += 1e-15  # hack: force lil sparse format to allocate
            if alloc_R_x:
                Ri_x += 1e-15  # hack: force lil sparse format to allocate
            M.glob.R[Ig] += Ri
            M.glob.R_U[Ig, 4 * Is[i - 1] : (4 * Is[i - 1] + 4)] += Ri_U[:, 0:4]
            M.glob.R_U[Ig, 4 * Is[i] : (4 * Is[i] + 4)] += Ri_U[:, 4:8]
            M.glob.R_x[Ig, [Is[i - 1], Is[i]]] += Ri_x

            # following transition, all stations will be turbulent
            if tran:
                turb = True

    # include effects of R_x into R_U: R_ue += R_x*x_st*st_ue
    #   The global residual Jacobian has a column for ue sensitivity
    #   ue, the edge velocity, also affects the location of the stagnation point
    #   The location of the stagnation point (st) dictates the x value at each node
    #   The residual also depends on the x value at each node (R_x)
    #   We use the chain rule (formula above) to account for this
    Nsys = M.glob.Nsys  # number of dofs
    Iue = range(3, 4 * Nsys, 4)  # ue indices in U
    x_st = -M.isol.sign_ue  # st = stag point [Nsys x 1]
    x_st = np.concatenate((x_st, -np.ones(M.wake.N)))  # wake same sens as upper surface
    R_st = M.glob.R_x @ x_st[:, np.newaxis]  # [3*Nsys x 1]
    Ist, st_ue = M.isol.stag_idx, M.isol.sstag_ue  # stag points, sens
    if (alloc_R_x) or (alloc_R_U):
        R_st += 1e-15  # hack to avoid sparse matrix warning
    M.glob.R_U[:, Iue[Ist[0]]] += R_st * st_ue[0]
    M.glob.R_U[:, Iue[Ist[1]]] += R_st * st_ue[1]

    if alloc_R_U:
        M.glob.R_U = M.glob.R_U.tocsr()
    if alloc_R_x:
        M.glob.R_x = M.glob.R_x.tocsr()


def stagnation_state(U, x):
    # extrapolates two states in U, first ones in BL, to stagnation
    # INPUT
    #   U  : [U1,U2] = states at first two nodes (4x2)
    #   x  : [x1,x2] = x-locations of first two nodes (2x1)
    # OUTPUT
    #   Ust    : stagnation state (4x1)
    #   Ust_U  : linearization of Ust w.r.t. U1 and U2 (4x8)
    #   Ust_x  : linearization of Ust w.r.t. x1 and x2 (4x2)
    #   xst    : stagnation point location ... close to 0
    # DETAILS
    #   fits a quadratic to the edge velocity: 0 at x=0, then through two states
    #   linearly extrapolates other states in U to x=0, from U1 and U2

    # pull off states
    U1, U2, x1, x2 = U[:, 0], U[:, 1], x[0], x[1]
    dx = x2 - x1
    dx_x = np.r_[-1, 1]
    rx = x2 / x1
    rx_x = np.r_[-rx, 1] / x1

    # linear extrapolation weights and stagnation state
    w1 = x2 / dx
    w1_x = -w1 / dx * dx_x + np.r_[0, 1] / dx
    w2 = -x1 / dx
    w2_x = -w2 / dx * dx_x + np.r_[-1, 0] / dx
    Ust = U1 * w1 + U2 * w2

    # quadratic extrapolation of the edge velocity for better slope, ue=K*x
    wk1 = rx / dx
    wk1_x = rx_x / dx - wk1 / dx * dx_x
    wk2 = -1 / (rx * dx)
    wk2_x = -wk2 * (rx_x / rx + dx_x / dx)
    K = wk1 * U1[3] + wk2 * U2[3]
    K_U = np.r_[0, 0, 0, wk1, 0, 0, 0, wk2]
    K_x = U1[3] * wk1_x + U2[3] * wk2_x

    # stagnation coord cannot be zero, but must be small
    xst = 1e-6
    Ust[3] = K * xst  # linear dep of ue on x near stagnation
    Ust_U = np.block([[w1 * np.eye(3, 4), w2 * np.eye(3, 4)], [K_U * xst]])
    Ust_x = np.vstack((np.outer(U1[0:3], w1_x) + np.outer(U2[0:3], w2_x), K_x * xst))

    return Ust, Ust_U, Ust_x, xst


def thwaites_init(K, nu):
    # uses Thwaites correlation to initialize first node in stag point flow
    # INPUT
    #   K  : stagnation point constant
    #   nu : kinematic viscosity
    # OUTPUT
    #   th : momentum thickness
    #   ds : displacement thickness
    # DETAILS
    #   ue = K*x -> K = ue/x = stag point flow constant
    #   th^2 = ue^(-6) * 0.45 * nu * int_0^x ue^5 dx = 0.45*nu/(6*K)
    #   ds = Hstag*th = 2.2*th

    th = np.sqrt(0.45 * nu / (6.0 * K))  # momentum thickness
    ds = 2.2 * th  # displacement thickness

    return th, ds


def wake_sys(M: Mfoil, param: Param):
    # constructs residual system corresponding to wake initialization
    # INPUT
    #   param  : parameters
    # OUTPUT
    #   R   : 3x1 residual vector for th, ds, sa
    #   R_U : 3x12 residual linearization, as three 3x4 blocks
    #   J   : indices of the blocks of U in R_U (lower, upper, wake)

    il = M.vsol.Is[0][-1]
    Ul = M.glob.U[:, il]  # lower surface TE index, state
    iu = M.vsol.Is[1][-1]
    Uu = M.glob.U[:, iu]  # upper surface TE index, state
    iw = M.vsol.Is[2][0]
    Uw = M.glob.U[:, iw]  # first wake index, state
    t, hTE, dtdx, tcp, tdp = TE_info(M.foil.x)  # trailing-edge gap is hTE

    # Obtain wake shear stress from upper/lower; transition if not turb
    turb = True
    if M.vsol.turb[il]:
        ctl = Ul[2]
        ctl_Ul = np.r_[0, 0, 1, 0]  # already turb; use state
    else:
        ctl, ctl_Ul = get_cttr(Ul, param, turb)  # transition shear stress, lower
    if M.vsol.turb[iu]:
        ctu = Uu[2]
        ctu_Uu = np.r_[0, 0, 1, 0]  # already turb; use state
    else:
        ctu, ctu_Uu = get_cttr(Uu, param, turb)  # transition shear stress, upper
    thsum = Ul[0] + Uu[0]  # sum of thetas
    ctw = (ctl * Ul[0] + ctu * Uu[0]) / thsum  # theta-average
    ctw_Ul = (ctl_Ul * Ul[0] + (ctl - ctw) * np.r_[1, 0, 0, 0]) / thsum
    ctw_Uu = (ctu_Uu * Uu[0] + (ctu - ctw) * np.r_[1, 0, 0, 0]) / thsum

    # residual; note, delta star in wake includes the TE gap, hTE
    R = np.r_[Uw[0] - (Ul[0] + Uu[0]), Uw[1] - (Ul[1] + Uu[1] + hTE), Uw[2] - ctw]
    J = [il, iu, iw]  # R depends on states at these nodes
    R_Ul = np.vstack((-np.eye(2, 4), -ctw_Ul))
    R_Uu = np.vstack((-np.eye(2, 4), -ctw_Uu))
    R_Uw = np.eye(3, 4)
    R_U = np.hstack((R_Ul, R_Uu, R_Uw))
    return R, R_U, J


def wake_init(M: Mfoil, ue):
    # initializes the first point of the wake, using data in M.glob.U
    # INPUT
    #   ue  : edge velocity at the wake point
    # OUTPUT
    #   Uw  : 4x1 state vector at the wake point

    iw = M.vsol.Is[2][0]
    Uw = M.glob.U[:, iw]  # first wake index, state
    [R, R_U, J] = wake_sys(M, M.param)  # construct the wake system
    Uw[0:3] -= R
    Uw[3] = ue  # solve the wake system, use ue
    return Uw


def init_boundary_layer(M: Mfoil):
    # initializes BL solution on foil and wake by marching with given edge vel, ue
    # INPUT
    #   The edge velocity field ue must be filled in on the airfoil and wake
    # OUTPUT
    #   The state in M.glob.U is filled in for each point

    Hmaxl = 3.8  # above this shape param value, laminar separation occurs
    Hmaxt = 2.5  # above this shape param value, turbulent separation occurs

    ueinv = get_ueinv(M)  # get inviscid velocity

    M.glob.Nsys = M.foil.N + M.wake.N  # number of global variables (nodes)

    # do we need to initialize?
    if (not M.oper.initbl) and (M.glob.U.shape[1] == M.glob.Nsys):
        vprint(M.param.verb, 1, '\n <<< Starting with current boundary layer >>> \n')
        M.glob.U[3, :] = ueinv  # do set a new edge velocity
        return

    vprint(M.param.verb, 1, '\n <<< Initializing the boundary layer >>> \n')

    M.glob.U = np.zeros((4, M.glob.Nsys))  # global solution matrix
    M.vsol.turb = np.zeros(M.glob.Nsys, dtype=int)  # node flag: 0 = lam, 1 = turb

    for side in range(3):  # loop over surfaces
        vprint(M.param.verb, 3, f'\nSide is = {side}:\n')

        Is = M.vsol.Is[side]  # surface point indices
        xi = M.isol.xi[Is]  # distance from LE stag point
        ue = ueinv[Is]  # edge velocities
        N = len(Is)  # number of points
        U = np.zeros([4, N])  # states at all points: [th, ds, sa, ue]
        Aux = np.zeros(N)  # auxiliary data at all points: [wgap]

        # ensure edge velocities are not tiny
        uemax = max(abs(ue))
        for i in range(N):
            ue[i] = max(ue[i], 1e-8 * uemax)

        wake = side == 2
        turb = wake  # the wake is fully turbulent

        # set auxiliary data
        if side == 2:
            Aux[:] = M.vsol.wgap

        # initialize state at first point
        i0 = 0
        if side < 2:
            # Solve for the stagnation state (Thwaites initialization + Newton)
            if xi[0] < 1e-8 * xi[-1]:
                K, hitstag = ue[1] / xi[1], True
            else:
                K, hitstag = ue[0] / xi[0], False
            th, ds = thwaites_init(K, M.param.mu0 / M.param.rho0)
            xst = 1.0e-6  # small but nonzero
            Ust = np.array([th, ds, 0, K * xst])
            nNewton = 20
            for iNewton in range(nNewton):
                # call residual at stagnation
                turb, simi = False, True  # similarity station flag
                R, R_U, R_x = residual_station(
                    M.param,
                    np.r_[xst, xst],
                    np.stack((Ust, Ust), axis=-1),
                    np.zeros(2),
                    turb=False,
                    wake=False,
                    simi=True,
                )
                simi = False
                if norm2(R) < 1e-10:
                    break
                A = R_U[:, 4:7] + R_U[:, 0:3]
                b = -R
                dU = np.append(np.linalg.solve(A, b), 0)
                # under-relaxation
                dm = max(abs(dU[0] / Ust[0]), abs(dU[1] / Ust[1]))
                omega = 1 if (dm < 0.2) else 0.2 / dm
                dU = dU * omega
                Ust = Ust + dU

            # store stagnation state in first one (rarely two) points
            if hitstag:
                U[:, 0] = Ust
                U[3, 0] = ue[0]
                i0 = 1
            U[:, i0] = Ust
            U[3, i0] = ue[i0]

        else:  # wake
            U[:, 0] = wake_init(M, ue[0])  # initialize wake state properly
            turb = True  # force turbulent in wake if still laminar
            M.vsol.turb[Is[0]] = True  # wake starts turbulent

        # march over rest of points
        tran = False  # flag indicating that we are at transition
        i = i0 + 1
        while i < N:
            Ip = [i - 1, i]  # two points involved in the calculation
            U[:, i] = U[:, i - 1]
            U[3, i] = ue[i]  # guess = same state, new ue
            if tran:  # set shear stress at transition interval
                ct, ct_U = get_cttr(U[:, i], M.param, turb)
                U[2, i] = ct
            M.vsol.turb[Is[i]] = tran or turb  # flag node i as turbulent
            direct = True  # default is direct mode
            nNewton, iNswitch = 30, 12
            for iNewton in range(nNewton):
                # call residual at this station
                if tran:  # we are at transition
                    vprint(M.param.verb, 4, f'i={i}, residual_transition (iNewton = {iNewton}) \n')
                    try:
                        R, R_U, R_x = residual_transition(M, M.param, xi[Ip], U[:, Ip], Aux[Ip], wake, simi)
                    except ValueError:
                        vprint(M.param.verb, 1, 'Transition calculation failed in BL init. Continuing.')
                        M.vsol.xt = 0.5 * sum(xi[Ip])
                        U[:, i] = U[:, i - 1]
                        U[3, i] = ue[i]
                        U[2, i] = ct
                        R = 0  # so we move on
                else:
                    vprint(M.param.verb, 4, f'i={i}, residual_station (iNewton = {iNewton})')
                    R, R_U, R_x = residual_station(M.param, xi[Ip], U[:, Ip], Aux[Ip], turb, wake, simi)
                if norm2(R) < 1e-10:
                    break

                if direct:  # direct mode => ue is prescribed => solve for th, ds, sa
                    A = R_U[:, 4:7]
                    b = -R
                    dU = np.append(np.linalg.solve(A, b), 0)
                else:  # inverse mode => Hk is prescribed
                    Hk, Hk_U = get_Hk(U[:, i], M.param)
                    A = np.vstack((R_U[:, 4:8], Hk_U))
                    b = np.r_[-R, Hktgt - Hk]  # noqa F281
                    dU = np.linalg.solve(A, b)

                # under-relaxation
                dm = max(abs(dU[0] / U[0, i - 1]), abs(dU[1] / U[1, i - 1]))
                if not direct:
                    dm = max(dm, abs(dU[3] / U[3, i - 1]))
                if turb:
                    dm = max(dm, abs(dU[2] / U[2, i - 1]))
                elif direct:
                    dm = max(dm, abs(dU[2] / 10))

                omega = 0.3 / dm if (dm > 0.3) else 1
                dU = dU * omega

                # trial update
                Ui = U[:, i] + dU

                # clip extreme values
                if turb:
                    Ui[2] = max(min(Ui[2], 0.3), 1e-7)
                # Hklim = 1.02; if (M.param.wake), Hklim = 1.00005; end
                # [Hk,Hk_U] = get_Hk(Ui, M.param);
                # dH = max(0,Hklim-Hk); Ui(2) = Ui(2) + dH*Ui(1);

                # check if about to separate
                Hmax = Hmaxt if (turb) else Hmaxl
                Hk, Hk_U = get_Hk(Ui, M.param)

                if (direct) and ((Hk > Hmax) or (iNewton > iNswitch)):
                    # no update; need to switch to inverse mode: prescribe Hk
                    direct = False
                    vprint(M.param.verb, 2, f'** switching to inverse: i={i}, iNewton={iNewton}')
                    [Hk, Hk_U] = get_Hk(U[:, i - 1], M.param)
                    Hkr = (xi[i] - xi[i - 1]) / U[0, i - 1]
                    if wake:
                        H2 = Hk
                        for k in range(6):
                            H2 -= (H2 + 0.03 * Hkr * (H2 - 1) ** 3 - Hk) / (1 + 0.09 * Hkr * (H2 - 1) ** 2)
                        Hktgt = max(H2, 1.01)
                    elif turb:
                        Hktgt = Hk - 0.15 * Hkr  # turb: decrease in Hk
                    else:
                        Hktgt = Hk + 0.03 * Hkr  # lam: increase in Hk
                    if not wake:
                        Hktgt = max(Hktgt, Hmax)
                    if iNewton > iNswitch:  # reinit
                        U[:, i] = U[:, i - 1]
                        U[3, i] = ue[i]
                else:
                    U[:, i] = Ui  # take the update

            if iNewton >= nNewton - 1:
                vprint(M.param.verb, 1, f'** BL init not converged: si={side}, i={i} **\n')
                # extrapolate values
                U[:, i] = U[:, i - 1]
                U[3, i] = ue[i]
                if side < 3:
                    U[0, i] = U[0, i - 1] * (xi[i] / xi[i - 1]) ** 0.5
                    U[1, i] = U[1, i - 1] * (xi[i] / xi[i - 1]) ** 0.5
                else:
                    rlen = (xi[i] - xi[i - 1]) / (10.0 * U[1, i - 1])
                    U[1, i] = (U[1, i - 1] + U[0, i - 1] * rlen) / (1.0 + rlen)  # TODO check on this extrap

            # check for transition
            if (not turb) and (not tran) and (U[2, i] > M.param.ncrit):
                vprint(M.param.verb, 2, f'Identified transition at (si={side}, i={i}): n={U[2, i]:.5f}, ncrit={M.param.ncrit:.5f}\n')
                tran = True
                continue  # redo station with transition

            if tran:
                store_transition(M, side, i)  # store transition location
                turb = True
                tran = False  # turbulent after transition

            i += 1  # next point

        # store states
        M.glob.U[:, Is] = U


def store_transition(M: Mfoil, si, i):
    # stores xi and x transition locations using current M.vsol.xt
    # INPUT
    #   si,i : side,station number
    # OUTPUT
    #   M.vsol.Xt stores the transition location s and x values

    xt = M.vsol.xt
    i0, i1 = M.vsol.Is[si][i - 1], M.vsol.Is[si][i]  # pre/post transition nodes
    xi0, xi1 = M.isol.xi[i0], M.isol.xi[i1]  # xi (s) locations at nodes
    assert (i0 < M.foil.N) and (i1 < M.foil.N), 'Can only store transition on airfoil'
    x0, x1 = M.foil.x[0, i0], M.foil.x[0, i1]  # x locations at nodes
    if (xt < xi0) or (xt > xi1):
        vprint(M.param.verb, 1, f'Warning: transition ({xt:.3f}) off interval ({xi0:.3f},{xi1:.3f})!')
    M.vsol.Xt[si, 0] = xt  # xi location
    M.vsol.Xt[si, 1] = x0 + (xt - xi0) / (xi1 - xi0) * (x1 - x0)  # x location
    slu = ['lower', 'upper']
    vprint(M.param.verb, 1, f'  transition on {slu[si]} side at x={M.vsol.Xt[si, 1]:.5f}')


def update_transition(M: Mfoil):
    # updates transition location using current state
    # INPUT
    #   a valid state in M.glob.U
    # OUTPUT
    #   M.vsol.turb : updated with latest lam/turb flags for each node
    #   M.glob.U    : updated with amp factor or shear stress as needed at each node

    for side in range(2):  # loop over lower/upper surfaces
        Is = M.vsol.Is[side]  # surface point indices
        N = len(Is)  # number of points

        # current last laminar station
        for ilam0 in range(N):
            if M.vsol.turb[Is[ilam0]]:
                ilam0 -= 1
                break

        # current amp/ctau solution (so we do not change it unnecessarily)
        sa = M.glob.U[2, Is].copy()

        # march amplification equation to get new last laminar station
        ilam = march_amplification(M, side)

        if ilam == ilam0:
            M.glob.U[2, Is] = sa[:]  # no change
            continue

        vprint(M.param.verb, 2, f'  Update transition: last lam [{ilam0}]->[{ilam}]')

        if ilam < ilam0:
            # transition is now earlier: fill in turb between [ilam+1, ilam0]
            turb = True
            sa0, temp = get_cttr(M.glob.U[:, Is[ilam + 1]], M.param, turb)
            sa1 = M.glob.U[2, Is[ilam0 + 1]] if (ilam0 < N - 1) else sa0
            xi = M.isol.xi[Is]
            dx = xi[min(ilam0 + 1, N - 1)] - xi[ilam + 1]
            for i in range(ilam + 1, ilam0 + 1):
                f = 0 if (dx == 0) or (i == ilam + 1) else (xi[i] - xi[ilam + 1]) / dx
                if (ilam + 1) == ilam0:
                    f = 1
                M.glob.U[2, Is[i]] = sa0 + f * (sa1 - sa0)
                assert M.glob.U[2, Is[i]] > 0, 'negative ctau in update_transition'
                M.vsol.turb[Is[i]] = True

        elif ilam > ilam0:
            # transition is now later: lam already filled in; leave turb alone
            for i in range(ilam0, ilam + 1):
                M.vsol.turb[Is[i]] = False


def march_amplification(M: Mfoil, si):
    # marches amplification equation on surface si
    # INPUT
    #   si : surface number index
    # OUTPUT
    #   ilam : index of last laminar station before transition
    #   M.glob.U : updated with amp factor at each (new) laminar station
    param = M.param
    Is = M.vsol.Is[si]  # surface point indices
    N = len(Is)  # number of points
    U = M.glob.U[:, Is]  # states
    turb = M.vsol.turb[Is]  # turbulent station flag

    # loop over stations, calculate amplification
    U[2, 0] = 0.0  # no amplification at first station
    i = 1
    while i < N:
        U1, U2 = U[:, i - 1], U[:, i].copy()  # states
        if turb[i]:
            U2[2] = U1[2] * 1.01  # initialize amp if turb
        dx = M.isol.xi[Is[i]] - M.isol.xi[Is[i - 1]]  # interval length

        # Newton iterations, only needed if adding extra amplification in damp
        nNewton = 20
        for iNewton in range(nNewton):
            # amplification rate, averaged
            damp1, damp1_U1 = get_damp(U1, param)
            damp2, damp2_U2 = get_damp(U2, param)
            damp, damp_U = upwind(0.5, 0, damp1, damp1_U1, damp2, damp2_U2)

            Ramp = U2[2] - U1[2] - damp * dx

            if iNewton > 11:
                vprint(param.verb, 3, f'i={i}, iNewton={iNewton}, sa = [{U1[2]:.5e}, {U2[2]:.5e}], damp = {damp:.5e}, Ramp = {Ramp:.5e}')

            if abs(Ramp) < 1e-12:
                break  # converged
            Ramp_U = np.r_[0, 0, -1, 0, 0, 0, 1, 0] - damp_U * dx
            dU = -Ramp / Ramp_U[6]
            omega = 1
            dmax = 0.5 * (1.01 - iNewton / nNewton)
            if abs(dU) > dmax:
                omega = dmax / abs(dU)
            U2[2] += omega * dU

        if iNewton >= nNewton - 1:
            vprint(param.verb, 1, 'march amp Newton unconverged!')

        # check for transition
        if U2[2] > param.ncrit:
            vprint(param.verb, 2, f'  march_amplification (si,i={si},{i}): {U2[2]:.5e} is above critical.')
            break
        else:
            M.glob.U[2, Is[i]] = U2[2]  # store amplification in M.glob.U (also seen in view U)
            U[2, i] = U2[2]
            if np.iscomplex(U[2, i]):
                raise ValueError('imaginary amp during march')

        i += 1  # next station

    return i - 1  # return last laminar station


def residual_transition(M: Mfoil, param: Param, x, U, Aux, wake: bool, simi: bool):
    # calculates the combined lam + turb residual for a transition station
    # INPUT
    #   param : parameter structure
    #   x     : 2x1 vector, [x1, x2], containing xi values at the points
    #   U     : 4x2 matrix, [U1, U2], containing the states at the points
    #   Aux   : ()x2 matrix, [Aux1, Aux2] of auxiliary data at the points
    # OUTPUT
    #   R     : 3x1 transition residual vector
    #   R_U   : 3x8 residual Jacobian, [R_U1, R_U2]
    #   R_x   : 3x2 residual linearization w.r.t. x, [R_x1, R_x2]
    # DETAILS
    #   The state U1 should be laminar; U2 should be turbulent
    #   Calculates and linearizes the transition location in the process
    #   Assumes linear variation of th and ds from U1 to U2

    # states
    U1 = U[:, 0]
    U2 = U[:, 1]
    sa = U[2, :]
    I1 = range(4)
    I2 = range(4, 8)
    Z = np.zeros(4)

    # interval
    x1 = x[0]
    x2 = x[1]
    dx = x2 - x1

    # determine transition location (xt) using amplification equation
    xt = x1 + 0.5 * dx  # guess
    ncrit = param.ncrit  # critical amp factor
    nNewton = 20
    vprint(param.verb, 3, f'  Transition interval = [{x1:.5e}, {x2:.5e}]')
    #  U1, U2
    for iNewton in range(nNewton):
        w2 = (xt - x1) / dx
        w1 = 1.0 - w2  # weights
        Ut = w1 * U1 + w2 * U2
        Ut_xt = (U2 - U1) / dx  # state at xt
        Ut[2] = ncrit
        Ut_xt[2] = 0.0  # amplification at transition
        damp1, damp1_U1 = get_damp(U1, param)
        dampt, dampt_Ut = get_damp(Ut, param)
        dampt_Ut[2] = 0.0
        Rxt = ncrit - sa[0] - 0.5 * (xt - x1) * (damp1 + dampt)
        Rxt_xt = -0.5 * (damp1 + dampt) - 0.5 * (xt - x1) * np.dot(dampt_Ut, Ut_xt)
        dxt = -Rxt / Rxt_xt
        vprint(param.verb, 4, f'   Transition: iNewton,Rxt,xt = {iNewton},{Rxt:.5e},{xt:.5e}')
        dmax = 0.2 * dx * (1.1 - iNewton / nNewton)
        if abs(dxt) > dmax:
            dxt = dxt * dmax / abs(dxt)
        if abs(Rxt) < 1e-10:
            break
        if iNewton < nNewton:
            xt += dxt

    if iNewton >= nNewton:
        vprint(param.verb, 1, 'Transition location calculation failed.')
    M.vsol.xt = xt  # save transition location

    # prepare for xt linearizations
    Rxt_U = -0.5 * (xt - x1) * np.concatenate((damp1_U1 + dampt_Ut * w1, dampt_Ut * w2))
    Rxt_U[2] -= 1.0
    Ut_x1 = (U2 - U1) * (w2 - 1) / dx
    Ut_x2 = (U2 - U1) * (-w2) / dx  # at fixed xt
    Ut_x1[2] = 0
    Ut_x2[2] = 0  # amp at xt is always ncrit
    Rxt_x1 = 0.5 * (damp1 + dampt) - 0.5 * (xt - x1) * np.dot(dampt_Ut, Ut_x1)
    Rxt_x2 = -0.5 * (xt - x1) * np.dot(dampt_Ut, Ut_x2)

    # sensitivity of xt w.r.t. U,x from Rxt(xt,U,x) = 0 constraint
    xt_U = -Rxt_U / Rxt_xt
    xt_U1 = xt_U[I1]
    xt_U2 = xt_U[I2]
    xt_x1 = -Rxt_x1 / Rxt_xt
    xt_x2 = -Rxt_x2 / Rxt_xt

    # include derivatives w.r.t. xt in Ut_x1 and Ut_x2
    Ut_x1 += Ut_xt * xt_x1
    Ut_x2 += Ut_xt * xt_x2

    # sensitivity of Ut w.r.t. U1 and U2
    Ut_U1 = w1 * np.eye(4) + np.outer((U2 - U1), xt_U1) / dx  # w1*I + U1*w1_xt*xt_U1 + U2*w2_xt*xt_U1;
    Ut_U2 = w2 * np.eye(4) + np.outer((U2 - U1), xt_U2) / dx  # w2*I + U1*w1_xt*xt_U2 + U2*w2_xt*xt_U2;

    # laminar and turbulent states at transition
    Utl = Ut.copy()
    Utl_U1 = Ut_U1.copy()
    Utl_U2 = Ut_U2.copy()
    Utl_x1 = Ut_x1.copy()
    Utl_x2 = Ut_x2.copy()
    Utl[2] = ncrit
    Utl_U1[2, :] = Z
    Utl_U2[2, :] = Z
    Utl_x1[2] = 0
    Utl_x2[2] = 0
    Utt = Ut.copy()
    Utt_U1 = Ut_U1.copy()
    Utt_U2 = Ut_U2.copy()
    Utt_x1 = Ut_x1.copy()
    Utt_x2 = Ut_x2.copy()

    # set turbulent shear coefficient, sa, in Utt
    cttr, cttr_Ut = get_cttr(Ut, param, turb=True)
    Utt[2] = cttr
    Utt_U1[2, :] = np.dot(cttr_Ut, Ut_U1)
    Utt_U2[2, :] = np.dot(cttr_Ut, Ut_U2)
    Utt_x1[2] = np.dot(cttr_Ut, Ut_x1)
    Utt_x2[2] = np.dot(cttr_Ut, Ut_x2)

    # laminar/turbulent residuals and linearizations
    Rl, Rl_U, Rl_x = residual_station(param, np.r_[x1, xt], np.stack((U1, Utl), axis=-1), Aux, False, False, False)
    Rl_U1 = Rl_U[:, I1]
    Rl_Utl = Rl_U[:, I2]
    Rt, Rt_U, Rt_x = residual_station(param, np.r_[xt, x2], np.stack((Utt, U2), axis=-1), Aux, True, False, False)
    Rt_Utt = Rt_U[:, I1]
    Rt_U2 = Rt_U[:, I2]

    # combined residual and linearization
    R = Rl + Rt
    if any(np.imag(R) != 0):
        raise ValueError('imaginary transition residual')
    R_U1 = Rl_U1 + np.dot(Rl_Utl, Utl_U1) + np.outer(Rl_x[:, 1], xt_U1) + np.dot(Rt_Utt, Utt_U1) + np.outer(Rt_x[:, 0], xt_U1)
    R_U2 = np.dot(Rl_Utl, Utl_U2) + np.outer(Rl_x[:, 1], xt_U2) + np.dot(Rt_Utt, Utt_U2) + Rt_U2 + np.outer(Rt_x[:, 0], xt_U2)
    R_U = np.hstack((R_U1, R_U2))
    R_x = np.stack(
        (
            Rl_x[:, 0] + Rl_x[:, 1] * xt_x1 + Rt_x[:, 0] * xt_x1 + np.dot(Rl_Utl, Utl_x1) + np.dot(Rt_Utt, Utt_x1),
            Rt_x[:, 1] + Rl_x[:, 1] * xt_x2 + Rt_x[:, 0] * xt_x2 + np.dot(Rl_Utl, Utl_x2) + np.dot(Rt_Utt, Utt_x2),
        ),
        axis=-1,
    )

    return R, R_U, R_x


def residual_station(param: Param, x, Uin, Aux, turb: bool, wake: bool, simi: bool):
    # calculates the viscous residual at one non-transition station
    # INPUT
    #   param : parameter structure
    #   x     : 2x1 vector, [x1, x2], containing xi values at the points
    #   U     : 4x2 matrix, [U1, U2], containing the states at the points
    #   Aux   : ()x2 matrix, [Aux1, Aux2] of auxiliary data at the points
    # OUTPUT
    #   R     : 3x1 residual vector (mom, shape-param, amp/lag)
    #   R_U   : 3x8 residual Jacobian, [R_U1, R_U2]
    #   R_x   : 3x2 residual linearization w.r.t. x, [R_x1, R_x2]
    # DETAILS
    #   The input states are U = [U1, U2], each with th,ds,sa,ue

    # so that we do not overwrite Uin
    U = Uin.copy()

    # modify ds to take out wake gap (in Aux) for all calculations below
    U[1, :] -= Aux

    # states
    U1 = U[:, 0]
    U2 = U[:, 1]
    Um = 0.5 * (U1 + U2)
    th = U[0, :]
    ds = U[1, :]
    sa = U[2, :]

    # speed needs compressibility correction
    uk1, uk1_u = get_uk(U1[3], param)
    uk2, uk2_u = get_uk(U2[3], param)

    # log changes
    thlog = np.log(th[1] / th[0])
    thlog_U = np.r_[-1.0 / th[0], 0, 0, 0, 1.0 / th[1], 0, 0, 0]
    uelog = np.log(uk2 / uk1)
    uelog_U = np.r_[0, 0, 0, -uk1_u / uk1, 0, 0, 0, uk2_u / uk2]
    xlog = np.log(x[1] / x[0])
    xlog_x = np.r_[-1.0 / x[0], 1.0 / x[1]]
    dx = x[1] - x[0]
    dx_x = np.r_[-1.0, 1.0]

    # upwinding factor
    upw, upw_U = get_upw(U1, U2, param, wake)

    # shape parameter
    H1, H1_U1 = get_H(U1)
    H2, H2_U2 = get_H(U2)
    H = 0.5 * (H1 + H2)
    H_U = 0.5 * np.r_[H1_U1, H2_U2]

    # Hstar = KE shape parameter, averaged
    Hs1, Hs1_U1 = get_Hs(U1, param, turb, wake)
    Hs2, Hs2_U2 = get_Hs(U2, param, turb, wake)
    Hs, Hs_U = upwind(0.5, 0, Hs1, Hs1_U1, Hs2, Hs2_U2)

    # log change in Hstar
    Hslog = np.log(Hs2 / Hs1)
    Hslog_U = np.r_[-1.0 / Hs1 * Hs1_U1, 1.0 / Hs2 * Hs2_U2]

    # similarity station is special: U1 = U2, x1 = x2
    if simi:
        thlog = 0.0
        thlog_U *= 0.0
        Hslog = 0.0
        Hslog_U *= 0.0
        uelog = 1.0
        uelog_U *= 0.0
        xlog = 1.0
        xlog_x = np.r_[0.0, 0.0]
        dx = 0.5 * (x[0] + x[1])
        dx_x = np.r_[0.5, 0.5]

    # Hw = wake shape parameter
    Hw1, Hw1_U1 = get_Hw(U1, Aux[0])
    Hw2, Hw2_U2 = get_Hw(U2, Aux[1])
    Hw = 0.5 * (Hw1 + Hw2)
    Hw_U = 0.5 * np.r_[Hw1_U1, Hw2_U2]

    # set up shear lag or amplification factor equation
    if turb:
        # log change of root shear stress coeff
        salog = np.log(sa[1] / sa[0])
        salog_U = np.r_[0, 0, -1.0 / sa[0], 0, 0, 0, 1.0 / sa[1], 0]

        # BL thickness measure, averaged
        de1, de1_U1 = get_de(U1, param)
        de2, de2_U2 = get_de(U2, param)
        de, de_U = upwind(0.5, 0, de1, de1_U1, de2, de2_U2)

        # normalized slip velocity, averaged
        Us1, Us1_U1 = get_Us(U1, param, turb, wake)
        Us2, Us2_U2 = get_Us(U2, param, turb, wake)
        Us, Us_U = upwind(0.5, 0, Us1, Us1_U1, Us2, Us2_U2)

        # Hk, upwinded
        Hk1, Hk1_U1 = get_Hk(U1, param)
        Hk2, Hk2_U2 = get_Hk(U2, param)
        Hk, Hk_U = upwind(upw, upw_U, Hk1, Hk1_U1, Hk2, Hk2_U2)

        # Re_theta, averaged
        Ret1, Ret1_U1 = get_Ret(U1, param)
        Ret2, Ret2_U2 = get_Ret(U2, param)
        Ret, Ret_U = upwind(0.5, 0, Ret1, Ret1_U1, Ret2, Ret2_U2)

        # skin friction, upwinded
        cf1, cf1_U1 = get_cf(U1, param, turb, wake)
        cf2, cf2_U2 = get_cf(U2, param, turb, wake)
        cf, cf_U = upwind(upw, upw_U, cf1, cf1_U1, cf2, cf2_U2)

        # displacement thickness, averaged
        dsa = 0.5 * (ds[0] + ds[1])
        dsa_U = 0.5 * np.r_[0, 1, 0, 0, 0, 1, 0, 0]

        # uq = equilibrium 1/ue * due/dx
        uq, uq_U = get_uq(dsa, dsa_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, param, wake)

        # cteq = root equilibrium wake layer shear coeficient: (ctau eq)^.5
        cteq1, cteq1_U1 = get_cteq(U1, param, turb, wake)
        cteq2, cteq2_U2 = get_cteq(U2, param, turb, wake)
        cteq, cteq_U = upwind(upw, upw_U, cteq1, cteq1_U1, cteq2, cteq2_U2)

        # root of shear coefficient (a state), upwinded
        saa, saa_U = upwind(upw, upw_U, sa[0], np.r_[0, 0, 1, 0], sa[1], np.r_[0, 0, 1, 0])

        # lag coefficient
        Klag = param.SlagK
        beta = param.GB
        Clag = Klag / beta * 1.0 / (1.0 + Us)
        Clag_U = -Clag / (1.0 + Us) * Us_U

        # extra dissipation in wake
        ald = 1.0
        if wake:
            ald = param.Dlr

        # shear lag equation
        Rlag = Clag * (cteq - ald * saa) * dx - 2 * de * salog + 2 * de * (uq * dx - uelog) * param.Cuq
        Rlag_U = (
            Clag_U * (cteq - ald * saa) * dx
            + Clag * (cteq_U - ald * saa_U) * dx
            - 2 * de_U * salog
            - 2 * de * salog_U
            + 2 * de_U * (uq * dx - uelog) * param.Cuq
            + 2 * de * (uq_U * dx - uelog_U) * param.Cuq
        )
        Rlag_x = Clag * (cteq - ald * saa) * dx_x + 2 * de * uq * dx_x

    else:
        # laminar, amplification factor equation

        if simi:
            # similarity station
            Rlag = sa[0] + sa[1]  # no amplification
            Rlag_U = np.array([0, 0, 1, 0, 0, 0, 1, 0])
            Rlag_x = np.array([0, 0])
        else:
            # amplification factor equation in Rlag

            # amplification rate, averaged
            damp1, damp1_U1 = get_damp(U1, param)
            damp2, damp2_U2 = get_damp(U2, param)
            damp, damp_U = upwind(0.5, 0, damp1, damp1_U1, damp2, damp2_U2)

            Rlag = sa[1] - sa[0] - damp * dx
            Rlag_U = np.array([0, 0, -1, 0, 0, 0, 1, 0]) - damp_U * dx
            Rlag_x = -damp * dx_x

    # squared mach number, symmetrical average
    Ms1, Ms1_U1 = get_Mach2(U1, param)
    Ms2, Ms2_U2 = get_Mach2(U2, param)
    Ms, Ms_U = upwind(0.5, 0, Ms1, Ms1_U1, Ms2, Ms2_U2)

    # skin friction * x/theta, symmetrical average
    cfxt1, cfxt1_U1, cfxt1_x1 = get_cfxt(U1, x[0], param, turb, wake)
    cfxt2, cfxt2_U2, cfxt2_x2 = get_cfxt(U2, x[1], param, turb, wake)
    cfxtm, cfxtm_Um, cfxtm_xm = get_cfxt(Um, 0.5 * (x[0] + x[1]), param, turb, wake)
    cfxt = 0.25 * cfxt1 + 0.5 * cfxtm + 0.25 * cfxt2
    cfxt_U = 0.25 * np.concatenate((cfxt1_U1 + cfxtm_Um, cfxtm_Um + cfxt2_U2))
    cfxt_x = 0.25 * np.array([cfxt1_x1 + cfxtm_xm, cfxtm_xm + cfxt2_x2])

    # momentum equation
    Rmom = thlog + (2 + H + Hw - Ms) * uelog - 0.5 * xlog * cfxt
    Rmom_U = thlog_U + (H_U + Hw_U - Ms_U) * uelog + (2 + H + Hw - Ms) * uelog_U - 0.5 * xlog * cfxt_U
    Rmom_x = -0.5 * xlog_x * cfxt - 0.5 * xlog * cfxt_x

    # dissipation function times x/theta: cDi = (2*cD/H*)*x/theta, upwinded
    cDixt1, cDixt1_U1, cDixt1_x1 = get_cDixt(U1, x[0], param, turb, wake)
    cDixt2, cDixt2_U2, cDixt2_x2 = get_cDixt(U2, x[1], param, turb, wake)
    cDixt, cDixt_U = upwind(upw, upw_U, cDixt1, cDixt1_U1, cDixt2, cDixt2_U2)
    cDixt_x = np.array([(1.0 - upw) * cDixt1_x1, upw * cDixt2_x2])

    # cf*x/theta, upwinded
    cfxtu, cfxtu_U = upwind(upw, upw_U, cfxt1, cfxt1_U1, cfxt2, cfxt2_U2)
    cfxtu_x = np.array([(1.0 - upw) * cfxt1_x1, upw * cfxt2_x2])

    # Hss = density shape parameter, averaged
    [Hss1, Hss1_U1] = get_Hss(U1, param)
    [Hss2, Hss2_U2] = get_Hss(U2, param)
    [Hss, Hss_U] = upwind(0.5, 0, Hss1, Hss1_U1, Hss2, Hss2_U2)

    Rshape = Hslog + (2 * Hss / Hs + 1 - H - Hw) * uelog + xlog * (0.5 * cfxtu - cDixt)
    Rshape_U = (
        Hslog_U
        + (2 * Hss_U / Hs - 2 * Hss / Hs**2 * Hs_U - H_U - Hw_U) * uelog
        + (2 * Hss / Hs + 1 - H - Hw) * uelog_U
        + xlog * (0.5 * cfxtu_U - cDixt_U)
    )
    Rshape_x = xlog_x * (0.5 * cfxtu - cDixt) + xlog * (0.5 * cfxtu_x - cDixt_x)

    # put everything together
    R = np.array([Rmom, Rshape, Rlag])
    R_U = np.vstack((Rmom_U, Rshape_U, Rlag_U))
    R_x = np.vstack((Rmom_x, Rshape_x, Rlag_x))

    return R, R_U, R_x


# ============ GET FUNCTIONS ==============


def get_upw(U1, U2, param: Param, wake: bool):
    # calculates a local upwind factor (0.5 = trap; 1 = BE) based on two states
    # INPUT
    #   U1,U2 : first/upwind and second/downwind states (4x1 each)
    #   param : parameter structure
    # OUTPUT
    #   upw   : scalar upwind factor
    #   upw_U : 1x8 linearization vector, [upw_U1, upw_U2]
    # DETAILS
    #   Used to ensure a stable viscous discretization
    #   Decision to upwind is made based on the shape factor change

    Hk1, Hk1_U1 = get_Hk(U1, param)
    Hk2, Hk2_U2 = get_Hk(U2, param)
    Z = np.zeros(Hk1_U1.shape)
    Hut = 1.0  # triggering constant for upwinding
    C = 1.0 if (wake) else 5.0
    Huc = C * Hut / Hk2**2  # only depends on U2
    Huc_U = np.concatenate((Z, -2 * Huc / Hk2 * Hk2_U2))
    aa = (Hk2 - 1.0) / (Hk1 - 1.0)
    sga = np.sign(aa)
    la = np.log(sga * aa)
    la_U = np.concatenate((-1.0 / (Hk1 - 1.0) * Hk1_U1, 1.0 / (Hk2 - 1.0) * Hk2_U2))
    Hls = la**2
    Hls_U = 2 * la * la_U
    if Hls > 15:
        Hls, Hls_U = 15, Hls_U * 0.0
    upw = 1.0 - 0.5 * np.exp(-Hls * Huc)
    upw_U = -0.5 * np.exp(-Hls * Huc) * (-Hls_U * Huc - Hls * Huc_U)

    return upw, upw_U


def upwind(upw, upw_U, f1, f1_U1, f2, f2_U2):
    # calculates an upwind average (and derivatives) of two scalars
    # INPUT
    #   upw, upw_U : upwind scalar and its linearization w.r.t. U1,U2
    #   f1, f1_U   : first scalar and its linearization w.r.t. U1
    #   f2, f2_U   : second scalar and its linearization w.r.t. U2
    # OUTPUT
    #   f    : averaged scalar
    #   f_U  : linearization of f w.r.t. both states, [f_U1, f_U2]

    f = (1 - upw) * f1 + upw * f2
    f_U = (-upw_U) * f1 + upw_U * f2 + np.concatenate(((1 - upw) * f1_U1, upw * f2_U2))

    return f, f_U


def get_uq(ds, ds_U, cf, cf_U, Hk, Hk_U, Ret, Ret_U, param: Param, wake: bool):
    # calculates the equilibrium 1/ue*due/dx
    # INPUT
    #   ds, ds_U   : delta star and linearization (1x4)
    #   cf, cf_U   : skin friction and linearization (1x4)
    #   Hk, Hk_U   : kinematic shape parameter and linearization (1x4)
    #   Ret, Ret_U : theta Reynolds number and linearization (1x4)
    #   param      : parameter structure
    # OUTPUT
    #   uq, uq_U   : equilibrium 1/ue*due/dx and linearization w.r.t. state (1x4)

    beta, A, C = param.GB, param.GA, param.GC
    if wake:
        A, C = A * param.Dlr, 0.0
    # limit Hk (TODO smooth/eliminate)
    if (wake) and (Hk < 1.00005):
        Hk, Hk_U = 1.00005, Hk_U * 0.0
    if (not wake) and (Hk < 1.05):
        Hk, Hk_U = 1.05, Hk_U * 0.0
    Hkc = Hk - 1.0 - C / Ret
    Hkc_U = Hk_U + C / Ret**2 * Ret_U

    if Hkc < 0.01:
        Hkc, Hkc_U = 0.01, Hkc_U * 0.0
    ut = 0.5 * cf - (Hkc / (A * Hk)) ** 2
    ut_U = 0.5 * cf_U - 2 * (Hkc / (A * Hk)) * (Hkc_U / (A * Hk) - Hkc / (A * Hk**2) * Hk_U)
    uq = ut / (beta * ds)
    uq_U = ut_U / (beta * ds) - uq / ds * ds_U

    return uq, uq_U


def get_cttr(U, param: Param, turb: bool):
    # calculates root of the shear stress coefficient at transition
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   cttr, cttr_U : sqrt(shear stress coeff) and its lin w.r.t. U (1x4)
    # DETAILS
    #   used to initialize the first turb station after transition

    wake = False  # transition happens just before the wake starts
    cteq, cteq_U = get_cteq(U, param, turb, wake)
    Hk, Hk_U = get_Hk(U, param)
    if Hk < 1.05:
        Hk, Hk_U = 1.05, Hk_U * 0.0
    C, E = param.CtauC, param.CtauE
    c = C * np.exp(-E / (Hk - 1.0))
    c_U = c * E / (Hk - 1) ** 2 * Hk_U
    cttr = c * cteq
    cttr_U = c_U * cteq + c * cteq_U

    return cttr, cttr_U


def get_cteq(U, param: Param, turb: bool, wake: bool):
    # calculates root of the equilibrium shear stress coefficient: sqrt(ctau_eq)
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   cteq, cteq_U : sqrt(equilibrium shear stress) and its lin w.r.t. U (1x4)
    # DETAILS
    #   uses equilibrium shear stress correlations
    CC, C = 0.5 / (param.GA**2 * param.GB), param.GC
    Hk, Hk_U = get_Hk(U, param)
    Hs, Hs_U = get_Hs(U, param, turb, wake)
    H, H_U = get_H(U)
    Ret, Ret_U = get_Ret(U, param)
    Us, Us_U = get_Us(U, param, turb, wake)
    if wake:
        if Hk < 1.00005:
            Hk, Hk_U = 1.00005, Hk_U * 0.0
        Hkc = Hk - 1.0
        Hkc_U = Hk_U
    else:
        if Hk < 1.05:
            Hk, HK_U = 1.05, Hk_U * 0.0  # noqa F841
        Hkc = Hk - 1.0 - C / Ret
        Hkc_U = Hk_U + C / Ret**2 * Ret_U
        if Hkc < 0.01:
            Hkc, Hkc_U = 0.01, Hkc_U * 0.0

    num = CC * Hs * (Hk - 1) * Hkc**2
    num_U = CC * (Hs_U * (Hk - 1) * Hkc**2 + Hs * Hk_U * Hkc**2 + Hs * (Hk - 1) * 2 * Hkc * Hkc_U)
    den = (1 - Us) * H * Hk**2
    den_U = (-Us_U) * H * Hk**2 + (1 - Us) * H_U * Hk**2 + (1 - Us) * H * 2 * Hk * Hk_U
    cteq = np.sqrt(num / den)
    cteq_U = 0.5 / cteq * (num_U / den - num / den**2 * den_U)

    return cteq, cteq_U


def get_Hs(U, param: Param, turb: bool, wake: bool):
    # calculates Hs = Hstar = K.E. shape parameter, from U
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   Hs, Hs_U : Hstar and its lin w.r.t. U (1x4)
    # DETAILS
    #   Hstar is the ratio theta*/theta, where theta* is the KE thicknes
    Hk, Hk_U = get_Hk(U, param)

    # limit Hk (TODO smooth/eliminate)
    if (wake) and (Hk < 1.00005):
        Hk, Hk_U = 1.00005, Hk_U * 0.0
    if (not wake) and (Hk < 1.05):
        Hk, Hk_U = 1.05, Hk_U * 0.0

    if turb:  # turbulent
        Hsmin, dHsinf = 1.5, 0.015
        Ret, Ret_U = get_Ret(U, param)
        # limit Re_theta and dependence
        Ho = 4.0
        Ho_U = 0.0
        if Ret > 400:
            Ho, Ho_U = 3 + 400.0 / Ret, -400.0 / Ret**2 * Ret_U
        Reb, Reb_U = Ret, Ret_U
        if Ret < 200:
            Reb, Reb_U = 200, Reb_U * 0.0
        if Hk < Ho:  # attached branch
            Hr = (Ho - Hk) / (Ho - 1)
            Hr_U = (Ho_U - Hk_U) / (Ho - 1) - (Ho - Hk) / (Ho - 1) ** 2 * Ho_U
            aa = (2 - Hsmin - 4 / Reb) * Hr**2
            aa_U = (4 / Reb**2 * Reb_U) * Hr**2 + (2 - Hsmin - 4 / Reb) * 2 * Hr * Hr_U
            Hs = Hsmin + 4 / Reb + aa * 1.5 / (Hk + 0.5)
            Hs_U = -4 / Reb**2 * Reb_U + aa_U * 1.5 / (Hk + 0.5) - aa * 1.5 / (Hk + 0.5) ** 2 * Hk_U
        else:  # separated branch
            lrb = np.log(Reb)
            lrb_U = 1 / Reb * Reb_U
            aa = Hk - Ho + 4 / lrb
            aa_U = Hk_U - Ho_U - 4 / lrb**2 * lrb_U
            bb = 0.007 * lrb / aa**2 + dHsinf / Hk
            bb_U = 0.007 * (lrb_U / aa**2 - 2 * lrb / aa**3 * aa_U) - dHsinf / Hk**2 * Hk_U
            Hs = Hsmin + 4 / Reb + (Hk - Ho) ** 2 * bb
            Hs_U = -4 / Reb**2 * Reb_U + 2 * (Hk - Ho) * (Hk_U - Ho_U) * bb + (Hk - Ho) ** 2 * bb_U
        # slight Mach number correction
        M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
        den = 1 + 0.014 * M2
        den_M2 = 0.014
        Hs = (Hs + 0.028 * M2) / den
        Hs_U = (Hs_U + 0.028 * M2_U) / den - Hs / den * den_M2 * M2_U
    else:  # laminar
        a = Hk - 4.35
        if Hk < 4.35:
            num = 0.0111 * a**2 - 0.0278 * a**3
            Hs = num / (Hk + 1) + 1.528 - 0.0002 * (a * Hk) ** 2
            Hs_Hk = (0.0111 * 2 * a - 0.0278 * 3 * a**2) / (Hk + 1) - num / (Hk + 1) ** 2 - 0.0002 * 2 * a * Hk * (Hk + a)
        else:
            Hs = 0.015 * a**2 / Hk + 1.528
            Hs_Hk = 0.015 * 2 * a / Hk - 0.015 * a**2 / Hk**2
        Hs_U = Hs_Hk * Hk_U

    return Hs, Hs_U


def get_cp(u, param: Param):
    # calculates pressure coefficient from speed, with compressibility correction
    # INPUT
    #   u     : speed
    #   param : parameter structure
    # OUTPUT
    #   cp, cp_U : pressure coefficient and its linearization w.r.t. u
    # DETAILS
    #   Karman-Tsien correction is included

    Vinf = param.Vinf
    cp = 1 - (u / Vinf) ** 2
    cp_u = -2 * u / Vinf**2
    if param.Minf > 0:
        KTl, b = param.KTl, param.KTb
        den = b + 0.5 * KTl * (1 + b) * cp
        den_cp = 0.5 * KTl * (1 + b)
        cp /= den
        cp_u *= (1 - cp * den_cp) / den

    return cp, cp_u


def get_uk(u, param: Param):
    # calculates Karman-Tsien corrected speed
    # INPUT
    #   u     : incompressible speed
    #   param : parameter structure
    # OUTPUT
    #   uk, uk_u : compressible speed and its linearization w.r.t. u
    # DETAILS
    #   Uses the Karman-Tsien correction, Minf from param

    if param.Minf > 0:
        KTl, Vinf = param.KTl, param.Vinf
        den = 1 - KTl * (u / Vinf) ** 2
        den_u = -2 * KTl * u / Vinf**2
        uk = u * (1 - KTl) / den
        uk_u = (1 - KTl) / den - (uk / den) * den_u
    else:
        uk, uk_u = u, 1.0

    return uk, uk_u


def get_Mach2(U, param: Param):
    # calculates squared Mach number
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   M2, M2_U : squared Mach number and its linearization w.r.t. U (1x4)
    # DETAILS
    #   Uses constant total enthalpy from param.H0
    #   The speed of sound varies; depends on enthalpy, which depends on speed
    #   The compressible edge speed must be used

    if param.Minf > 0:
        H0, g = param.H0, param.gam
        uk, uk_u = get_uk(U[3], param)
        c2 = (g - 1) * (H0 - 0.5 * uk**2)
        c2_uk = (g - 1) * (-uk)  # squared speed of sound
        M2 = uk**2 / c2
        M2_uk = 2 * uk / c2 - M2 / c2 * c2_uk
        M2_U = np.array([0, 0, 0, M2_uk * uk_u])
    else:
        M2 = 0.0
        M2_U = np.zeros(4)

    return M2, M2_U


def get_H(U):
    # calculates H = shape parameter = delta*/theta, from U
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   H, H_U : shape parameter and its linearization w.r.t. U (1x4)
    # DETAILS
    #   H is the ratio of the displacement thickness to the momentum thickness
    #   In U, the ds entry should be (delta*-wgap) ... i.e wake gap taken out
    #   When the real H is needed with wake gap, Hw is calculated and added

    H = U[1] / U[0]
    H_U = np.array([-H / U[0], 1 / U[0], 0, 0])

    return H, H_U


def get_Hw(U, wgap):
    # calculates Hw = wake gap shape parameter = wgap/theta
    # INPUT
    #   U    : state vector [th; ds; sa; ue]
    #   wgap : wake gap
    # OUTPUT
    #   Hw, Hw_U : wake gap shape parameter and its linearization w.r.t. U (1x4)
    # DETAILS
    #   Hw is the ratio of the wake gap to the momentum thickness
    #   The wake gap is the TE gap extrapolated into the wake (dead air region)

    Hw = wgap / U[0]  # wgap/th
    Hw_U = np.array([-Hw / U[0], 0, 0, 0])

    return Hw, Hw_U


def get_Hk(U, param: Param):
    # calculates Hk = kinematic shape parameter, from U
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   Hk, Hk_U : kinematic shape parameter and its linearization w.r.t. U (1x4)
    # DETAILS
    #   Hk is like H but with no density in the integrals defining th and ds
    #   So it is exactly the same when density is constant (= freestream)
    #   Here, it is computed from H with a correlation using the Mach number

    H, H_U = get_H(U)

    if param.Minf > 0:
        M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
        den = 1 + 0.113 * M2
        den_M2 = 0.113
        Hk = (H - 0.29 * M2) / den
        Hk_U = (H_U - 0.29 * M2_U) / den - Hk / den * den_M2 * M2_U
    else:
        Hk, Hk_U = H, H_U

    return Hk, Hk_U


def get_Hss(U, param: Param):
    # calculates Hss = density shape parameter, from U
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   Hss, Hss_U : density shape parameter and its linearization w.r.t. U (1x4)
    # DETAILS

    M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
    Hk, Hk_U = get_Hk(U, param)
    num = 0.064 / (Hk - 0.8) + 0.251
    num_U = -0.064 / (Hk - 0.8) ** 2 * Hk_U
    Hss = M2 * num
    Hss_U = M2_U * num + M2 * num_U

    return Hss, Hss_U


def get_de(U, param: Param):
    # calculates simplified BL thickness measure
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   de, de_U : BL thickness "delta" and its linearization w.r.t. U (1x4)
    # DETAILS
    #   delta is delta* incremented with a weighted momentum thickness, theta
    #   The weight on theta depends on Hk, and there is an overall cap

    Hk, Hk_U = get_Hk(U, param)
    aa = 3.15 + 1.72 / (Hk - 1)
    aa_U = -1.72 / (Hk - 1) ** 2 * Hk_U
    de = U[0] * aa + U[1]
    de_U = np.array([aa, 1, 0, 0]) + U[0] * aa_U
    dmx = 12.0
    if de > dmx * U[0]:
        de, de_U = dmx * U[0], np.array([dmx, 0, 0, 0])

    return de, de_U


def get_rho(U, param: Param):
    # calculates the density (useful if compressible)
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   rho, rho_U : density and linearization
    # DETAILS
    #   If compressible, rho is calculated from stag rho + isentropic relations

    if param.Minf > 0:
        M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
        # uk, uk_u = get_uk(U[3], param)  # corrected speed
        gmi = param.gam - 1
        den = 1 + 0.5 * gmi * M2
        den_M2 = 0.5 * gmi
        rho = param.rho0 / den ** (1 / gmi)
        rho_U = (-1 / gmi) * rho / den * den_M2 * M2_U
    else:
        rho = param.rho0
        rho_U = np.zeros(4)

    return rho, rho_U


def get_Ret(U, param: Param):
    # calculates theta Reynolds number, Re_theta, from U
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   Ret, Ret_U : Reynolds number based on the momentum thickness, linearization
    # DETAILS
    #   Re_theta = rho*ue*theta/mu
    #   If compressible, rho is calculated from stag rho + isentropic relations
    #   ue is the edge speed and must be comressibility corrected
    #   mu is the dynamic viscosity, from Sutherland's law if compressible

    if param.Minf > 0:
        M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
        uk, uk_u = get_uk(U[3], param)  # corrected speed
        H0, gmi, Ts = param.H0, param.gam - 1, param.Tsrat
        Tr = 1 - 0.5 * uk**2 / H0
        Tr_uk = -uk / H0  # edge/stagnation temperature ratio
        f = Tr**1.5 * (1 + Ts) / (Tr + Ts)
        f_Tr = 1.5 * f / Tr - f / (Tr + Ts)  # Sutherland's ratio
        mu = param.mu0 * f
        mu_uk = param.mu0 * f_Tr * Tr_uk  # local dynamic viscosity
        den = 1 + 0.5 * gmi * M2
        den_M2 = 0.5 * gmi
        rho = param.rho0 / den ** (1 / gmi)
        rho_U = (-1 / gmi) * rho / den * den_M2 * M2_U  # density
        Ret = rho * uk * U[0] / mu
        Ret_U = rho_U * uk * U[0] / mu + (rho * U[0] / mu - Ret / mu * mu_uk) * np.array([0, 0, 0, uk_u]) + rho * uk / mu * np.array([1, 0, 0, 0])
    else:
        Ret = param.rho0 * U[0] * U[3] / param.mu0
        Ret_U = np.array([U[3], 0, 0, U[0]]) / param.mu0

    return Ret, Ret_U


def get_cf(U, param: Param, turb: bool, wake: bool):
    # calculates cf = skin friction coefficient, from U
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   cf, cf_U : skin friction coefficient and its linearization w.r.t. U (1x4)
    # DETAILS
    #   cf is the local skin friction coefficient = tau/(0.5*rho*ue^2)
    #   Correlations are used based on Hk and Re_theta

    if wake:
        return 0, np.zeros(4)  # zero cf in wake
    Hk, Hk_U = get_Hk(U, param)
    Ret, Ret_U = get_Ret(U, param)

    # TODO: limit Hk

    if turb:  # turbulent cf
        M2, M2_U = get_Mach2(U, param)  # squared edge Mach number
        Fc = np.sqrt(1 + 0.5 * (param.gam - 1) * M2)
        Fc_U = 0.5 / Fc * 0.5 * (param.gam - 1) * M2_U
        aa = -1.33 * Hk
        aa_U = -1.33 * Hk_U
        # if (aa < -20), aa = -20; aa_U = aa_U*0; warning('aa in cfturb'); end
        # smooth limiting of aa
        if aa < -17:
            aa = -20 + 3 * np.exp((aa + 17) / 3)
            aa_U = (aa + 20) / 3 * aa_U  # TODO: ping me
        bb = np.log(Ret / Fc)
        bb_U = Ret_U / Ret - Fc_U / Fc
        if bb < 3:
            bb, bb_U = 3, bb_U * 0
        bb /= np.log(10)
        bb_U /= np.log(10)
        cc = -1.74 - 0.31 * Hk
        cc_U = -0.31 * Hk_U
        dd = np.tanh(4.0 - Hk / 0.875)
        dd_U = (1 - dd**2) * (-Hk_U / 0.875)
        cf0 = 0.3 * np.exp(aa) * bb**cc
        cf0_U = cf0 * aa_U + 0.3 * np.exp(aa) * cc * bb ** (cc - 1) * bb_U + cf0 * np.log(bb) * cc_U
        cf = (cf0 + 1.1e-4 * (dd - 1)) / Fc
        cf_U = (cf0_U + 1.1e-4 * dd_U) / Fc - cf / Fc * Fc_U
    else:  # laminar cf
        if Hk < 5.5:
            num = 0.0727 * (5.5 - Hk) ** 3 / (Hk + 1) - 0.07
            num_Hk = 0.0727 * (3 * (5.5 - Hk) ** 2 / (Hk + 1) * (-1) - (5.5 - Hk) ** 3 / (Hk + 1) ** 2)
        else:
            num = 0.015 * (1 - 1.0 / (Hk - 4.5)) ** 2 - 0.07
            num_Hk = 0.015 * 2 * (1 - 1.0 / (Hk - 4.5)) / (Hk - 4.5) ** 2
        cf = num / Ret
        cf_U = num_Hk / Ret * Hk_U - num / Ret**2 * Ret_U

    return cf, cf_U


def get_cfxt(U, x, param: Param, turb: bool, wake: bool):
    # calculates cf*x/theta from the state
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   x     : distance along wall (xi)
    #   param : parameter structure
    # OUTPUT
    #   cfxt,  : the combination cf*x/theta (calls cf function)
    #   cfxt_U : linearization w.r.t. U (1x4)
    #   cfxt_x : linearization w.r.t x (scalar)
    # DETAILS
    #   This combination appears in the momentum and shape parameter equations

    cf, cf_U = get_cf(U, param, turb, wake)
    cfxt = cf * x / U[0]
    cfxt_U = cf_U * x / U[0]
    cfxt_U[0] = cfxt_U[0] - cfxt / U[0]
    cfxt_x = cf / U[0]

    return cfxt, cfxt_U, cfxt_x


def get_cfutstag(Uin, param: Param):
    # calculates cf*ue*theta, used in stagnation station calculations
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   F, F_U : value and linearization of cf*ue*theta
    # DETAILS
    #   Only for stagnation and laminar

    U = Uin.copy()
    U[3] = 0
    Hk, Hk_U = get_Hk(U, param)

    if Hk < 5.5:
        num = 0.0727 * (5.5 - Hk) ** 3 / (Hk + 1) - 0.07
        num_Hk = 0.0727 * (3 * (5.5 - Hk) ** 2 / (Hk + 1) * (-1) - (5.5 - Hk) ** 3 / (Hk + 1) ** 2)
    else:
        num = 0.015 * (1 - 1.0 / (Hk - 4.5)) ** 2 - 0.07
        num_Hk = 0.015 * 2 * (1 - 1.0 / (Hk - 4.5)) / (Hk - 4.5) ** 2
    nu = param.mu0 / param.rho0
    F = nu * num
    F_U = nu * num_Hk * Hk_U

    return F, F_U


def get_cdutstag(Uin, param: Param):
    # calculates cDi*ue*theta, used in stagnation station calculations
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   D, D_U : value and linearization of cDi*ue*theta
    # DETAILS
    #   Only for stagnation and laminar

    U = Uin.copy()
    U[3] = 0.0
    Hk, Hk_U = get_Hk(U, param)

    if Hk < 4:
        num = 0.00205 * (4 - Hk) ** 5.5 + 0.207
        num_Hk = 0.00205 * 5.5 * (4 - Hk) ** 4.5 * (-1)
    else:
        Hk1 = Hk - 4
        num = -0.0016 * Hk1**2 / (1 + 0.02 * Hk1**2) + 0.207
        num_Hk = -0.0016 * (2 * Hk1 / (1 + 0.02 * Hk1**2) - Hk1**2 / (1 + 0.02 * Hk1**2) ** 2 * 0.02 * 2 * Hk1)

    nu = param.mu0 / param.rho0
    D = nu * num
    D_U = nu * num_Hk * Hk_U

    return D, D_U


def get_cDixt(U, x, param: Param, turb: bool, wake: bool):
    # calculates cDi*x/theta from the state
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   x     : distance along wall (xi)
    #   param : parameter structure
    # OUTPUT
    #   cDixt,  : the combination cDi*x/theta (calls cDi function)
    #   cDixt_U : linearization w.r.t. U (1x4)
    #   cDixt_x : linearization w.r.t x (scalar)
    # DETAILS
    #   cDi is the dissipation function

    cDi, cDi_U = get_cDi(U, param, turb, wake)
    cDixt = cDi * x / U[0]
    cDixt_U = cDi_U * x / U[0]
    cDixt_U[0] = cDixt_U[0] - cDixt / U[0]
    cDixt_x = cDi / U[0]

    return cDixt, cDixt_U, cDixt_x


def get_cDi(U, param: Param, turb: bool, wake):
    # calculates cDi = dissipation function = 2*cD/H*, from the state
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   cDi, cDi_U : dissipation function and its linearization w.r.t. U (1x4)
    # DETAILS
    #   cD is the dissipation coefficient, int(tau*du/dn*dn)/(rho*ue^3)
    #   The combination with H* appears in the shape parameter equation

    if turb:  # turbulent includes wake
        # initialize to 0; will add components that are needed
        cDi, cDi_U = 0, np.zeros(4)

        if not wake:
            # turbulent wall contribution (0 in the wake)
            cDi0, cDi0_U = get_cDi_turbwall(U, param, wake)
            cDi = cDi + cDi0
            cDi_U = cDi_U + cDi0_U
            cDil, cDil_U = get_cDi_lam(U, param)  # for max check
        else:
            cDil, cDil_U = get_cDi_lamwake(U, param)  # for max check

        # outer layer contribution
        cDi0, cDi0_U = get_cDi_outer(U, param, turb, wake)
        cDi = cDi + cDi0
        cDi_U = cDi_U + cDi0_U

        # laminar stress contribution
        cDi0, cDi0_U = get_cDi_lamstress(U, param, turb, wake)
        cDi = cDi + cDi0
        cDi_U = cDi_U + cDi0_U

        # maximum check
        if cDil > cDi:
            cDi, cDi_U = cDil, cDil_U

        # double dissipation in the wake
        if wake:
            cDi, cDi_U = 2 * cDi, 2 * cDi_U
    else:
        # just laminar dissipation
        [cDi, cDi_U] = get_cDi_lam(U, param)

    return cDi, cDi_U


def get_cDi_turbwall(U, param: Param, wake: bool):
    # calculates the turbulent wall contribution to cDi
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   cDi, cDi_U : dissipation function and its linearization w.r.t. U (1x4)
    # DETAILS
    #   This is one contribution to the dissipation function cDi = 2*cD/H*
    if wake:
        return 0, np.zeros(4)  # for pinging
    turb = True
    # get cf, Hk, Hs, Us
    cf, cf_U = get_cf(U, param, turb, wake)
    Hk, Hk_U = get_Hk(U, param)
    Hs, Hs_U = get_Hs(U, param, turb, wake)
    Us, Us_U = get_Us(U, param, turb, wake)
    Ret, Ret_U = get_Ret(U, param)

    lr = np.log(Ret)
    lr_U = Ret_U / Ret
    Hmin = 1 + 2.1 / lr
    Hmin_U = -2.1 / lr**2 * lr_U
    aa = np.tanh((Hk - 1) / (Hmin - 1))
    fac = 0.5 + 0.5 * aa
    fac_U = 0.5 * (1 - aa**2) * (Hk_U / (Hmin - 1) - (Hk - 1) / (Hmin - 1) ** 2 * Hmin_U)

    cDi = 0.5 * cf * Us * (2 / Hs) * fac
    cDi_U = cf_U * Us / Hs * fac + cf * Us_U / Hs * fac - cDi / Hs * Hs_U + cf * Us / Hs * fac_U

    return cDi, cDi_U


def get_cDi_lam(U, param: Param):
    # calculates the laminar dissipation function cDi
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   cDi, cDi_U : dissipation function and its linearization w.r.t. U (1x4)
    # DETAILS
    #   This is one contribution to the dissipation function cDi = 2*cD/H*

    # first get Hk and Ret
    Hk, Hk_U = get_Hk(U, param)
    Ret, Ret_U = get_Ret(U, param)

    if Hk < 4:
        num = 0.00205 * (4 - Hk) ** 5.5 + 0.207
        num_Hk = 0.00205 * 5.5 * (4 - Hk) ** 4.5 * (-1)
    else:
        Hk1 = Hk - 4
        num = -0.0016 * Hk1**2 / (1 + 0.02 * Hk1**2) + 0.207
        num_Hk = -0.0016 * (2 * Hk1 / (1 + 0.02 * Hk1**2) - Hk1**2 / (1 + 0.02 * Hk1**2) ** 2 * 0.02 * 2 * Hk1)

    cDi = num / Ret
    cDi_U = num_Hk / Ret * Hk_U - num / Ret**2 * Ret_U

    return cDi, cDi_U


def get_cDi_lamwake(U, param: Param):
    # laminar wake dissipation function cDi
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   cDi, cDi_U : dissipation function and its linearization w.r.t. U (1x4)
    # DETAILS
    #   This is one contribution to the dissipation function cDi = 2*cD/H*

    turb = False  # force laminar
    wake = True

    # dependencies
    Hk, Hk_U = get_Hk(U, param)
    Hs, Hs_U = get_Hs(U, param, turb, wake)
    Ret, Ret_U = get_Ret(U, param)
    HsRet = Hs * Ret
    HsRet_U = Hs_U * Ret + Hs * Ret_U

    num = 2 * 1.1 * (1 - 1 / Hk) ** 2 * (1 / Hk)
    num_Hk = 2 * 1.1 * (2 * (1 - 1 / Hk) * (1 / Hk**2) * (1 / Hk) + (1 - 1 / Hk) ** 2 * (-1 / Hk**2))
    cDi = num / HsRet
    cDi_U = num_Hk * Hk_U / HsRet - num / HsRet**2 * HsRet_U

    return cDi, cDi_U


def get_cDi_outer(U, param: Param, turb: bool, wake: bool):
    # turbulent outer layer contribution to dissipation function cDi
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   cDi, cDi_U : dissipation function and its linearization w.r.t. U (1x4)
    # DETAILS
    #   This is one contribution to the dissipation function cDi = 2*cD/H*
    if not turb:
        return 0, np.zeros(4)  # for pinging

    # first get Hs, Us
    [Hs, Hs_U] = get_Hs(U, param, turb, wake)
    [Us, Us_U] = get_Us(U, param, turb, wake)

    # shear stress: note, state stores ct^.5
    ct = U[2] ** 2
    ct_U = np.array([0, 0, 2 * U[2], 0])

    cDi = ct * (0.995 - Us) * 2 / Hs
    cDi_U = ct_U * (0.995 - Us) * 2 / Hs + ct * (-Us_U) * 2 / Hs - ct * (0.995 - Us) * 2 / Hs**2 * Hs_U

    return cDi, cDi_U


def get_cDi_lamstress(U, param: Param, turb: bool, wake: bool):
    # laminar stress contribution to dissipation function cDi
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   cDi, cDi_U : dissipation function and its linearization w.r.t. U (1x4)
    # DETAILS
    #   This is one contribution to the dissipation function cDi = 2*cD/H*

    # first get Hs, Us, and Ret
    Hs, Hs_U = get_Hs(U, param, turb, wake)
    Us, Us_U = get_Us(U, param, turb, wake)
    Ret, Ret_U = get_Ret(U, param)
    HsRet = Hs * Ret
    HsRet_U = Hs_U * Ret + Hs * Ret_U

    num = 0.15 * (0.995 - Us) ** 2 * 2
    num_Us = 0.15 * 2 * (0.995 - Us) * (-1) * 2
    cDi = num / HsRet
    cDi_U = num_Us * Us_U / HsRet - num / HsRet**2 * HsRet_U

    return cDi, cDi_U


def get_Us(U, param: Param, turb: bool, wake: bool):
    # calculates the normalized wall slip velocity Us
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   Us, Us_U : normalized wall slip velocity and its linearization w.r.t. U (1x4)

    [Hs, Hs_U] = get_Hs(U, param, turb, wake)
    [Hk, Hk_U] = get_Hk(U, param)
    [H, H_U] = get_H(U)

    # limit Hk (TODO smooth/eliminate)
    if (wake) and (Hk < 1.00005):
        Hk, Hk_U = 1.00005, Hk_U * 0
    if (not wake) and (Hk < 1.05):
        Hk, Hk_U = 1.05, Hk_U * 0

    beta = param.GB
    bi = 1.0 / beta
    Us = 0.5 * Hs * (1 - bi * (Hk - 1) / H)
    Us_U = 0.5 * Hs_U * (1 - bi * (Hk - 1) / H) + 0.5 * Hs * (-bi * (Hk_U) / H + bi * (Hk - 1) / H**2 * H_U)
    # limits
    if (not wake) and (Us > 0.95):
        Us, Us_U = 0.98, Us_U * 0
    if (not wake) and (Us > 0.99995):
        Us, Us_U = 0.99995, Us_U * 0

    return Us, Us_U


def get_damp(U, param: Param):
    # calculates the amplification rate, dn/dx, used in predicting transition
    # INPUT
    #   U     : state vector [th; ds; sa; ue]
    #   param : parameter structure
    # OUTPUT
    #   damp, damp_U : amplification rate and its linearization w.r.t. U (1x4)
    # DETAILS
    #   damp = dn/dx is used in the amplification equation, prior to transition

    [Hk, Hk_U] = get_Hk(U, param)
    [Ret, Ret_U] = get_Ret(U, param)
    th = U[0]

    # limit Hk (TODO smooth/eliminate)
    if Hk < 1.05:
        Hk, Hk_U = 1.05, Hk_U * 0

    Hmi = 1.0 / (Hk - 1)
    Hmi_U = -(Hmi**2) * Hk_U
    aa = 2.492 * Hmi**0.43
    aa_U = 0.43 * aa / Hmi * Hmi_U
    bb = np.tanh(14 * Hmi - 9.24)
    bb_U = (1 - bb**2) * 14 * Hmi_U
    lrc = aa + 0.7 * (bb + 1)
    lrc_U = aa_U + 0.7 * bb_U
    lten = np.log(10)
    lr = np.log(Ret) / lten
    lr_U = (1 / Ret) * Ret_U / lten
    dl = 0.1  # changed from .08 to make smoother
    damp = 0
    damp_U = np.zeros(len(U))  # default no amplification
    if lr >= lrc - dl:
        rn = (lr - (lrc - dl)) / (2 * dl)
        rn_U = (lr_U - lrc_U) / (2 * dl)
        if rn >= 1:
            rf = 1
            rf_U = np.zeros(len(U))
        else:
            rf = 3 * rn**2 - 2 * rn**3
            rf_U = (6 * rn - 6 * rn**2) * rn_U
        ar = 3.87 * Hmi - 2.52
        ar_U = 3.87 * Hmi_U
        ex = np.exp(-(ar**2))
        ex_U = ex * (-2 * ar * ar_U)
        da = 0.028 * (Hk - 1) - 0.0345 * ex
        da_U = 0.028 * Hk_U - 0.0345 * ex_U
        af = -0.05 + 2.7 * Hmi - 5.5 * Hmi**2 + 3 * Hmi**3 + 0.1 * np.exp(-20 * Hmi)
        af_U = (2.7 - 11 * Hmi + 9 * Hmi**2 - 1 * np.exp(-20 * Hmi)) * Hmi_U
        damp = rf * af * da / th
        damp_U = (rf_U * af * da + rf * af_U * da + rf * af * da_U) / th - damp / th * np.array([1, 0, 0, 0])

    # extra amplification to ensure dn/dx > 0 near ncrit
    ncrit = param.ncrit

    Cea = 5
    nx = Cea * (U[2] - ncrit)
    nx_U = Cea * np.array([0, 0, 1, 0])
    eex = 1 + np.tanh(nx)
    eex_U = (1 - np.tanh(nx) ** 2) * nx_U

    ed = eex * 0.001 / th
    ed_U = eex_U * 0.001 / th - ed / th * np.array([1, 0, 0, 0])
    damp = damp + ed
    damp_U = damp_U + ed_U

    return damp, damp_U
