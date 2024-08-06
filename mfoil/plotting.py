import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mfoil.solver import Mfoil
from mfoil.utils import norm2


# ============ PLOTTING AND POST-PROCESSING  ==============


# -------------------------------------------------------------------------------
def plot_cpplus(ax, m):
    # makes a cp plot with outputs printed
    # INPUT
    #   M : mfoil structure
    # OUTPUT
    #   cp plot on current axes

    chord = m.geom.chord
    x = m.foil.x[0, :].copy()
    xrng = np.array([-0.1, 1.4]) * chord
    if m.oper.viscous:
        x = np.concatenate((x, m.wake.x[0, :]))
        colors = ["red", "blue", "black"]
        for si in range(3):
            Is = m.vsol.Is[si]
            ax.plot(x[Is], m.post.cp[Is], "-", color=colors[si], linewidth=2)
            ax.plot(x[Is], m.post.cpi[Is], "--", color=colors[si], linewidth=2)
    else:
        ax.plot(x, m.post.cp, "-", color="blue", linewidth=2)

    if (m.oper.Ma > 0) and (m.param.cps > (min(m.post.cp) - 0.2)):
        ax.plot([xrng(1), chord], m.param.cps * [1, 1], "--", color="black", linewidth=2)
        ax.text(0.8 * chord, m.param.cps - 0.1, r"sonic $c_p$", fontsize=18)

    ax.set_xlim(xrng)
    ax.invert_yaxis()
    ax.set_ylabel(r"$c_p$", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.spines["bottom"].set_position("zero")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # output text box
    textstr = "\n".join(
        (
            r"\underline{%s}" % (m.geom.name),
            r"$\textrm{Ma} = %.4f$" % (m.oper.Ma),
            r"$\alpha = %.2f^{\circ}$" % (m.oper.alpha),
            r"$c_{\ell} = %.4f$" % (m.post.cl),
            r"$c_{m} = %.4f$" % (m.post.cm),
            r"$c_{d} = %.6f$" % (m.post.cd),
        )
    )
    ax.text(
        0.74,
        0.97,
        textstr,
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top",
    )

    if m.oper.viscous:
        textstr = "\n".join(
            (
                r"$\textrm{Re} = %.1e$" % (m.oper.Re),
                r"$c_{df} = %.5f$" % (m.post.cdf),
                r"$c_{dp} = %.5f$" % (m.post.cdp),
            )
        )
        ax.text(
            0.74,
            0.05,
            textstr,
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="top",
        )


# -------------------------------------------------------------------------------
def plot_airfoil(ax, m: Mfoil):
    """
    Makes an airfoil plot

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to plot
    M : Mfoil
        Mfoil structure
    """

    chord = m.geom.chord
    xz = m.foil.x.copy()
    if m.oper.viscous:
        xz = np.hstack((xz, m.wake.x))
    xrng = np.array([-0.1, 1.4]) * chord
    ax.plot(xz[0, :], xz[1, :], "-", color="black", linewidth=1)
    ax.axis("equal")
    ax.set_xlim(xrng)
    ax.axis("off")


# -------------------------------------------------------------------------------
def mplot_boundary_layer(ax, m: Mfoil):
    """
    Makes a plot of the boundary layer

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to plot
    M : Mfoil
        Mfoil structure
    """
    if not m.oper.viscous:
        return
    xz = np.hstack((m.foil.x, m.wake.x))
    N = m.foil.N
    ds = m.post.ds  # displacement thickness
    rl = 0.5 * (1 + (ds[0] - ds[N - 1]) / ds[N])
    ru = 1 - rl
    t = np.hstack((m.foil.t, m.wake.t))  # tangents
    n = np.vstack((-t[1, :], t[0, :]))  # outward normals
    for i in range(n.shape[1]):
        n[:, i] /= norm2(n[:, i])
    xzd = xz + n * ds  # airfoil + delta*
    ctype = ["red", "blue", "black"]
    for i in range(4):
        si = i
        if si == 2:
            xzd = xz + n * ds * ru
        if si == 3:
            xzd, si = xz - n * ds * rl, 2
        Is = m.vsol.Is[si]
        ax.plot(xzd[0, Is], xzd[1, Is], "-", color=ctype[si], linewidth=2)


# -------------------------------------------------------------------------------
def plot_results(m: Mfoil):
    """
    Makes a summary results plot with cp, airfoil, BL delta, outputs

    Parameters
    ----------
    M : Mfoil
        Mfoil structure
    """

    assert m.post.cp is not None, "no cp for results plot"

    # figure parameters
    plt.rcParams["figure.figsize"] = [8, 7]
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["text.usetex"] = True

    # figure
    f = plt.figure()
    ax1 = f.add_subplot(111)
    gs = gridspec.GridSpec(4, 1)
    ax1.set_position(gs[0:3].get_position(f))
    ax1.set_subplotspec(gs[0:3])
    ax2 = f.add_subplot(gs[3])
    f.tight_layout()
    plt.show(block=m.post.rfile is None)

    # cp plot
    plot_cpplus(ax1, m)

    # airfoil plot
    plot_airfoil(ax2, m)

    # # BL thickness
    mplot_boundary_layer(ax2, m)

    if m.post.rfile is not None:
        plt.savefig(m.post.rfile)
    
    plt.show()
    plt.close()


# -------------------------------------------------------------------------------
def main():
    m = Mfoil(naca="2412", npanel=199)
    m.param.verb = 1
    print("NACA geom name =", m.geom.name, "  num. panels =", m.foil.N)
    # add camber
    # m.geom_addcamber(np.array([[0,0.3,0.7,1],[0,-.03,.01,0]]))
    # set up a compressible viscous run (note, alpha is in degrees)
    # m.geom_flap([0.85, 0], 10)
    m.setoper(alpha=2, Re=5e6, Ma=0.0)
    # request plotting, specify the output file for the plot
    m.post.rfile = "results.pdf"
    # run the solver
    print("Running the solver.")
    m.solve()
    plot_results(m)
    m.setoper(alpha=2.2, Re=5e6, Ma=0.0)
    m.oper.initbl = False
    m.solve()
    plot_results(m)
    # run the derivative ping check
    # print('Derivative ping check.')
    # m.ping()


if __name__ == "__main__":
    main()
