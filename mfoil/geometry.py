import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from mfoil.utils import cosd, sind, atan2, norm2, dist


class Geom:  # geometry
    def __init__(self):
        self.chord = 1.0  # chord length
        self.wakelen = 1.0  # wake extent length, in chords
        self.npoint = 1  # number of geometry representation points
        self.name = "noname"  # airfoil name, e.g. NACA XXXX
        self.xpoint = []  # point coordinates, [2 x npoint]
        self.xref = np.array([0.25, 0])  # moment reference point


# ============ PANELING  ==============
class Panel:  # paneling
    def __init__(self):
        self.N = 0  # number of nodes
        self.x = []  # node coordinates, [2 x N]
        self.s = []  # arclength values at nodes
        self.t = []  # dx/ds, dy/ds tangents at nodes


def make_panels(geom: Geom, npanel: int, stgt=None) -> Panel:
    # places panels on the current airfoil, as described by geom.xpoint
    # INPUT
    #   M      : mfoil class
    #   npanel : number of panels
    #   stgt   : optional target s values (e.g. for adaptation), or None
    # OUTPUT
    #   M.foil.N : number of panel points
    #   M.foil.x : coordinates of panel nodes (2xN)
    #   M.foil.s : arclength values at nodes (1xN)
    #   M.foil.t : tangent vectors, not normalized, dx/ds, dy/ds (2xN)
    # DETAILS
    #   Uses curvature-based point distribution on a spline of the points
    foil = Panel()
    Ufac = 2  # uniformity factor (higher, more uniform paneling)
    TEfac = 0.1  # Trailing-edge factor (higher, more TE resolution)
    foil.x, foil.s, foil.t = spline_curvature(geom.xpoint, npanel + 1, Ufac, TEfac, stgt)
    foil.N = foil.x.shape[1]
    return foil


def TE_info(X):
    # returns trailing-edge information for an airfoil with node coords X
    # INPUT
    #   X : node coordinates, ordered clockwise (2xN)
    # OUTPUT
    #   t    : bisector vector = average of upper/lower tangents, normalized
    #   hTE  : trailing edge gap, measured as a cross-section
    #   dtdx : thickness slope = d(thickness)/d(wake x)
    #   tcp  : |t cross p|, used for setting TE source panel strength
    #   tdp  : t dot p, used for setting TE vortex panel strength
    # DETAILS
    #   p refers to the unit vector along the TE panel (from lower to upper)

    t1 = X[:, 0] - X[:, 1]
    t1 = t1 / norm2(t1)  # lower tangent vector
    t2 = X[:, -1] - X[:, -2]
    t2 = t2 / norm2(t2)  # upper tangent vector
    t = 0.5 * (t1 + t2)
    t = t / norm2(t)  # average tangent; gap bisector
    s = X[:, -1] - X[:, 0]  # lower to upper connector vector
    hTE = -s[0] * t[1] + s[1] * t[0]  # TE gap
    dtdx = t1[0] * t2[1] - t2[0] * t1[1]  # sin(theta between t1,t2) approx dt/dx
    p = s / norm2(s)  # unit vector along TE panel
    tcp = abs(t[0] * p[1] - t[1] * p[0])
    tdp = np.dot(t, p)

    return t, hTE, dtdx, tcp, tdp


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
    d = dist(xj2 - xj1, zj2 - zj1)  # panel length
    r1 = dist(x, z)  # left edge to control point
    r2 = dist(x - d, z)  # right edge to control point
    theta1 = atan2(z, x)  # left angle
    theta2 = atan2(z, x - d)  # right angle

    return t, n, x, z, d, r1, r2, theta1, theta2

# ============ GEOMETRY ==============
def mgeom_flap(geom: Geom, npanel: int, xzhinge, eta) -> Panel:
    # deploys a flap at hinge location xzhinge, with flap angle eta
    # INPUTS
    #   M       : mfoil class containing an airfoil
    #   xzhinge : flap hinge location (x,z) as numpy array
    #   eta     : flap angle, positive = down, degrees
    # OUTPUTS
    #   M.foil.x : modified airfoil coordinates

    X = geom.xpoint
    N = X.shape[1]  # airfoil points
    xh = xzhinge[0]  # x hinge location

    # identify points on flap
    If = np.nonzero(X[0, :] > xh)[0]

    # rotate flap points
    R = np.array([[cosd(eta), sind(eta)], [-sind(eta), cosd(eta)]])
    for i in range(len(If)):
        X[:, If[i]] = xzhinge + R @ (X[:, If[i]] - xzhinge)

    # remove flap points to left of hinge
    idx = If[X[0, If] < xh]
    idx = np.setdiff1d(np.arange(N), idx)

    # re-assemble the airfoil; note, chord length is *not* redefined
    geom.xpoint = X[:, idx]
    geom.npoint = geom.xpoint.shape[1]

    # repanel
    return make_panels(geom, npanel)


def mgeom_addcamber(geom: Geom, npanel: int, xzcamb) -> Panel:
    # adds camber to airfoil from given coordinates
    # INPUTS
    #   M       : mfoil class containing an airfoil
    #   xzcamb  : (x,z) points on camberline increment, 2 x Nc
    # OUTPUTS
    #   M.foil.x : modified airfoil coordinates

    if xzcamb.shape[0] > xzcamb.shape[1]:
        xzcamb = np.transpose(xzcamb)

    X = geom.xpoint  # airfoil points

    # interpolate camber delta, add to X
    dz = interp1d(xzcamb[0, :], xzcamb[1, :], "cubic")(X[0, :])
    X[1, :] += dz

    # store back in M.geom
    geom.xpoint = X
    geom.npoint = geom.xpoint.shape[1]

    # repanel
    return make_panels(geom, npanel)


def mgeom_derotate(geom: Geom, npanel: int) -> Panel:
    # derotates airfoil about leading edge to make chordline horizontal
    # INPUTS
    #   M       : mfoil class containing an airfoil
    # OUTPUTS
    #   M.foil.x : modified airfoil coordinates

    X = geom.xpoint
    N = X.shape[1]  # airfoil points

    xLE = X[:, np.argmin(X[0, :])]  # LE point
    xTE = 0.5 * (X[:, 0] + X[:, N - 1])  # TE "point"

    theta = atan2(xTE[1] - xLE[1], xTE[0] - xLE[0])  # rotation angle
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    for i in range(N):
        X[:, i] = xLE + R @ (X[:, i] - xLE)

    # store back in M.geom
    geom.xpoint = X
    geom.npoint = geom.xpoint.shape[1]

    # repanel
    return make_panels(geom, npanel)


def space_geom(dx0, L, Np):
    # spaces Np points geometrically from [0,L], with dx0 as first interval
    # INPUTS
    #   dx0 : first interval length
    #   L   : total domain length
    #   Np  : number of points, including endpoints at 0,L
    # OUTPUTS
    #   x   : point locations (1xN)

    assert Np > 1, "Need at least two points for spacing."
    N = Np - 1  # number of intervals
    # L = dx0 * (1 + r + r^2 + ... r^{N-1}) = dx0*(r^N-1)/(r-1)
    # let d = L/dx0, and for a guess, consider r = 1 + s
    # The equation to solve becomes d*s  = (1+s)^N - 1
    # Initial guess: (1+s)^N ~ 1 + N*s + N*(N-1)*s^2/2 + N*(N-1)*(N-2)*s^3/3
    d = L / dx0
    a = N * (N - 1.0) * (N - 2.0) / 6.0
    b = N * (N - 1.0) / 2.0
    c = N - d
    disc = max(b * b - 4.0 * a * c, 0.0)
    r = 1 + (-b + np.sqrt(disc)) / (2 * a)
    for k in range(10):
        R = r**N - 1 - d * (r - 1)
        R_r = N * r ** (N - 1) - d
        dr = -R / R_r
        if abs(dr) < 1e-6:
            break
        r -= R / R_r
    return np.r_[0, np.cumsum(dx0 * r ** (np.array(range(N))))]


def set_coords(geom: Geom, X):
    # sets geometry from coordinate matrix
    # INPUTS
    #   M : mfoil class
    #   X : matrix whose rows or columns are (x,z) points, CW or CCW
    # OUTPUTS
    #   M.geom.npoint : number of points
    #   M.geom.xpoint : point coordinates (2 x npoint)
    #   M.geom.chord  : chord length
    # DETAILS
    #   Coordinates should start and end at the trailing edge
    #   Trailing-edge point must be repeated if sharp
    #   Points can be clockwise or counter-clockwise (will detect and make CW)

    if X.shape[0] > X.shape[1]:
        X = X.transpose()

    # ensure CCW
    A = 0.0
    for i in range(X.shape(1)):
        A += (X[0, i] - X[0, i - 1]) * (X[1, i] + X[1, i - 1])
    if A < 0:
        X = np.fliplr(X)

    # store points in M
    geom.npoint = X.shape(1)
    geom.xpoint = X
    geom.chord = max(X[0, :]) - min(X[0, :])


def naca_points(digits: str) -> Geom:
    # calculates coordinates of a NACA 4-digit airfoil, stores in M.geom
    # INPUTS
    #   M      : mfoil class
    #   digits : character array containing NACA digits
    # OUTPUTS
    #   M.geom.npoint : number of points
    #   M.geom.xpoint : point coordinates (2 x npoint)
    #   M.geom.chord  : chord length
    # DETAILS
    #   Uses analytical camber/thickness formulas
    geom = Geom()
    geom.name = "NACA " + digits
    N, te = 100, 1.5  # points per side and trailing-edge bunching factor
    f = np.linspace(0, 1, N + 1)  # linearly-spaced points between 0 and 1
    x = 1 - (te + 1) * f * (1 - f) ** te - (1 - f) ** (te + 1)  # bunched points, x, 0 to 1

    # normalized thickness, gap at trailing edge (use -.1035*x**4 for no gap)
    t = 0.2969 * np.sqrt(x) - 0.126 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    tmax = float(digits[-2:]) * 0.01  # max thickness
    t = t * tmax / 0.2

    if len(digits) == 4:
        # 4-digit series
        m, p = float(digits[0]) * 0.01, float(digits[1]) * 0.1
        c = m / (1 - p) ** 2 * ((1 - 2.0 * p) + 2.0 * p * x - x**2)
        for i in range(len(x)):
            if x[i] < p:
                c[i] = m / p**2 * (2 * p * x[i] - x[i] ** 2)
    elif len(digits) == 5:
        # 5-digit series
        n = float(digits[1])
        valid = digits[0] == "2" and digits[2] == "0" and n > 0 and n < 6
        assert valid, "5-digit NACA must begin with 2X0, X in 1-5"
        mv = [0.058, 0.126, 0.2025, 0.29, 0.391]
        m = mv(n)
        cv = [361.4, 51.64, 15.957, 6.643, 3.23]
        cc = cv(n)
        c = (cc / 6.0) * (x**3 - 3 * m * x**2 + m**2 * (3 - m) * x)
        for i in range(len(x)):
            if x[i] > m:
                c[i] = (cc / 6.0) * m**3 * (1 - x(i))
    else:
        raise ValueError("Provide 4 or 5 NACA digits")

    zu = c + t
    zl = c - t  # upper and lower surfaces
    xs = np.concatenate((np.flip(x), x[1:]))  # x points
    zs = np.concatenate((np.flip(zl), zu[1:]))  # z points

    # store points in M
    geom.npoint = len(xs)
    geom.xpoint = np.vstack((xs, zs))
    geom.chord = max(xs) - min(xs)
    return geom


def spline_curvature(Xin, N, Ufac, TEfac, stgt):
    # Splines 2D points in Xin and samples using curvature-based spacing
    # INPUT
    #   Xin   : points to spline
    #   N     : number of points = one more than the number of panels
    #   Ufac  : uniformity factor (1 = normal; > 1 means more uniform distribution)
    #   TEfac : trailing-edge resolution factor (1 = normal; > 1 = high; < 1 = low)
    #   stgt  : optional target s values
    # OUTPUT
    #   X  : new points (2xN)
    #   S  : spline s values (N)
    #   XS : spline tangents (2xN)

    # min/max of given points (x-coordinate)
    xmin, xmax = min(Xin[0, :]), max(Xin[0, :])

    # spline given points
    PP = spline2d(Xin)

    # curvature-based spacing on geom
    nfine = 501
    s = np.linspace(0, PP["X"].x[-1], nfine)
    xyfine = splineval(PP, s)
    PPfine = spline2d(xyfine)

    if stgt is None:
        s = PPfine["X"].x
        sk = np.zeros(nfine)
        xq, wq = quadseg()
        for i in range(nfine - 1):
            ds = s[i + 1] - s[i]
            st = xq * ds
            px = PPfine["X"].c[:, i]
            xss = 6.0 * px[0] * st + 2.0 * px[1]
            py = PPfine["Y"].c[:, i]
            yss = 6.0 * py[0] * st + 2.0 * py[1]
            skint = 0.01 * Ufac + 0.5 * np.dot(wq, np.sqrt(xss * xss + yss * yss)) * ds

            # force TE resolution
            xx = (0.5 * (xyfine[0, i] + xyfine[0, i + 1]) - xmin) / (xmax - xmin)  # close to 1 means at TE
            skint = skint + TEfac * 0.5 * np.exp(-100 * (1.0 - xx))

            # increment sk
            sk[i + 1] = sk[i] + skint

        # offset by fraction of average to avoid problems with zero curvature
        sk = sk + 2.0 * sum(sk) / nfine

        # arclength values at points
        skl = np.linspace(min(sk), max(sk), N)
        s = interp1d(sk, s, "cubic")(skl)
    else:
        s = stgt

    # new points
    X, S, XS = splineval(PPfine, s), s, splinetan(PPfine, s)

    return X, S, XS


def spline2d(X):
    # splines 2d points
    # INPUT
    #   X : points to spline (2xN)
    # OUTPUT
    #   PP : two-dimensional spline structure

    N = X.shape[1]
    S, Snew = np.zeros(N), np.zeros(N)

    # estimate the arclength and spline x, y separately
    for i in range(1, N):
        S[i] = S[i - 1] + norm2(X[:, i] - X[:, i - 1])
    PPX = CubicSpline(S, X[0, :])
    PPY = CubicSpline(S, X[1, :])

    # re-integrate to true arclength via several passes
    xq, wq = quadseg()
    for ipass in range(10):
        serr = 0
        Snew[0] = S[0]
        for i in range(N - 1):
            ds = S[i + 1] - S[i]
            st = xq * ds
            px = PPX.c[:, i]
            xs = 3.0 * px[0] * st * st + 2.0 * px[1] * st + px[2]
            py = PPY.c[:, i]
            ys = 3.0 * py[0] * st * st + 2.0 * py[1] * st + py[2]
            sint = np.dot(wq, np.sqrt(xs * xs + ys * ys)) * ds
            serr = max(serr, abs(sint - ds))
            Snew[i + 1] = Snew[i] + sint
        S[:] = Snew
        PPX = CubicSpline(S, X[0, :])
        PPY = CubicSpline(S, X[1, :])

    return {"X": PPX, "Y": PPY}


def splineval(PP, S):
    # evaluates 2d spline at given S values
    # INPUT
    #   PP : two-dimensional spline structure
    #   S  : arclength values at which to evaluate the spline
    # OUTPUT
    #   XY : coordinates on spline at the requested s values (2xN)

    return np.vstack((PP["X"](S), PP["Y"](S)))


def splinetan(PP, S):
    # evaluates 2d spline tangent (not normalized) at given S values
    # INPUT
    #   PP  : two-dimensional spline structure
    #   S   : arclength values at which to evaluate the spline tangent
    # OUTPUT
    #   XYS : dX/dS and dY/dS values at each point (2xN)

    DPX = PP["X"].derivative()
    DPY = PP["Y"].derivative()
    return np.vstack((DPX(S), DPY(S)))


def quadseg():
    # Returns quadrature points and weights for a [0,1] line segment
    # INPUT
    # OUTPUT
    #   x : quadrature point coordinates (1d)
    #   w : quadrature weights

    x = np.array([0.046910077030668, 0.230765344947158, 0.500000000000000, 0.769234655052842, 0.953089922969332])
    w = np.array([0.118463442528095, 0.239314335249683, 0.284444444444444, 0.239314335249683, 0.118463442528095])

    return x, w
