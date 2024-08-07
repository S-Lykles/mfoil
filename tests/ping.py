from mfoil.solver import Mfoil
import mfoil.solver as mfs
from mfoil.utils import norm2, vprint
import numpy as np

# -------------------------------------------------------------------------------
def check_ping(ep, v, v_u, sname):
    # checks convergence of 3 values/derivatives
    # INPUT
    #   v     : list of three function evaluations at 0,+ep,+2*ep
    #   v_u   : list of three derivative evaluations at 0,+ep,+2*ep
    #   sname : descriptive name of where values came from for printing
    # OUTPUT
    #   E     : error values for two finite-difference comparisons
    #   rate  : convergence rate, also printed

    E = np.zeros(2)
    for i in range(2):
        E[i] = norm2((v[1 + i] - v[0]) / (ep * (i + 1.0)) - 0.5 * (v_u[0] + v_u[1 + i]))
    rate = np.log2(E[1] / E[0])
    print("%s ping error convergence rate = %.4f" % (sname, rate))
    return E, rate


# -------------------------------------------------------------------------------
def ping_test(M: Mfoil):
    # checks derivatives of various functions through finite-difference pinging
    # INPUT
    #   M : mfoil class
    # OUTPUT
    #   printouts of rates (2 = second order expected).

    M.oper.alpha = 3  # angle, in degrees
    M.oper.Ma = 0.4  # Mach number
    M.oper.viscous = True  # tests are viscous
    np.random.seed(17)  # for consistent pseudo random numbers
    M.param.verb = 2  # to minimize prints to screen

    # freestream Reynolds numbers
    Rev = np.r_[2e3, 1e5]

    # laminar/turbulent test states: th, ds, sa, ue
    Uv = [np.r_[0.01, 0.02, 8.4, 0.9], np.r_[0.023, 0.05, 0.031, 1.1]]

    # functions to test
    fv = [
        mfs.get_Hk,
        mfs.get_Ret,
        mfs.get_cf,
        mfs.get_cDi,
        mfs.get_Hs,
        mfs.get_Us,
        mfs.get_cDi_turbwall,
        mfs.get_cDi_lam,
        mfs.get_cDi_lamwake,
        mfs.get_cDi_outer,
        mfs.get_cDi_lamstress,
        mfs.get_cteq,
        mfs.get_cttr,
        mfs.get_de,
        mfs.get_damp,
        mfs.get_Mach2,
        mfs.get_Hss,
        mfs.residual_station,
    ]
    param = M.param
    # ping tests
    sturb = ["lam", "turb", "wake"]
    for iRe in range(len(Rev)):  # loop over Reynolds numbers
        M.oper.Re = Rev[iRe]
        mfs.init_thermo(M)
        turb, wake, simi = False, False, False
        for it in range(3):  # loop over lam, turb, wake
            turb, wake = (it > 0), (it == 2)
            for ih in range(len(fv)):  # loop over functions
                U, srates, smark, serr, f = Uv[min(it, 1)], "", "", "", fv[ih]
                if f == mfs.residual_station:
                    U = np.concatenate((U, U * np.r_[1.1, 0.8, 0.9, 1.2]))
                for k in range(len(U)):  # test all state component derivatives
                    ep, E = 1e-2 * U[k], np.zeros(2)
                    if f == mfs.residual_station:
                        xi, Aux, dx = (
                            np.r_[0.7, 0.8],
                            np.r_[0.002, 0.0018],
                            np.r_[-0.2, 0.3],
                        )
                        v0, v_U0, v_x0 = f(param, xi, np.stack((U[0:4], U[4:8]), axis=-1), Aux, turb, wake, simi)
                        for iep in range(2):  # test with two epsilons
                            U[k] += ep
                            xi += ep * dx
                            v1, v_U1, v_x1 = f(param, xi, np.stack((U[0:4], U[4:8]), axis=-1), Aux, turb, wake, simi)
                            U[k] -= ep
                            xi -= ep * dx
                            E[iep] = norm2((v1 - v0) / ep - 0.5 * (v_U1[:, k] + v_U0[:, k] + np.dot(v_x0 + v_x1, dx)))
                            ep /= 2
                    else:
                        [v0, v_U0] = f(U, param)
                        for iep in range(2):  # test with two epsilons
                            U[k] += ep
                            v1, v_U1 = f(U, param)
                            U[k] -= ep
                            E[iep] = abs((v1 - v0) / ep - 0.5 * (v_U1[k] + v_U0[k]))
                            ep /= 2
                    srate = " N/A"
                    skip = False
                    if (not skip) and (E[0] > 5e-11) and (E[1] > 5e-11):
                        m = np.log2(E[0] / E[1])
                        srate = "%4.1f" % (m)
                        if m < 1.5:
                            smark = "<==="
                    srates += " " + srate
                    serr += " %.2e->%.2e" % (E[0], E[1])
                vprint(
                    param.verb,
                    0,
                    "%-18s %-5s err=[%s]  rates=[%s] %s" % (f.__name__, sturb[it], serr, srates, smark),
                )

    # transition residual ping
    M.oper.Re = 2e6
    mfs.init_thermo(M)
    turb, wake, simi = False, False, False
    U, x, Aux = (
        np.transpose(np.array([[0.01, 0.02, 8.95, 0.9], [0.013, 0.023, 0.028, 0.85]])),
        np.r_[0.7, 0.8],
        np.r_[0, 0],
    )
    dU, dx, ep, v, v_u = np.random.rand(4, 2), np.random.rand(2), 1e-4, [], []
    for ie in range(3):
        R, R_U, R_x = mfs.residual_transition(M, param, x, U, Aux, wake, simi)
        v.append(R)
        v_u.append(np.dot(R_U, np.reshape(dU, 8, order="F")) + np.dot(R_x, dx))
        U += ep * dU
        x += ep * dx
    check_ping(ep, v, v_u, "transition residual")

    # stagnation residual ping
    M.oper.Re = 1e6
    M.oper.alpha = 1
    mfs.init_thermo(M)
    U, x, Aux = (
        np.array([0.00004673616, 0.000104289, 0, 0.11977917547]),
        4.590816441485401e-05,
        [0, 0],
    )
    dU, dx, ep, v, v_u = np.random.rand(4), np.random.rand(1), 1e-6, [], []
    for ie in range(3):
        simi = True
        R, R_U, R_x = mfs.residual_station(param, np.r_[x, x], np.stack((U, U), axis=-1), Aux, wake, simi)
        simi = False
        v.append(R)
        v_u.append(np.dot(R_U[:, range(0, 4)] + R_U[:, range(4, 8)], dU) + (R_x[:, 0] + R_x[:, 1]) * dx[0])
        U += ep * dU
        x += ep * dx[0]
    check_ping(ep, v, v_u, "stagnation residual")

    # need a viscous solution for the next tests
    mfs.solve_viscous(M)
    # entire system ping
    # M.param.niglob = 10
    Nsys = M.glob.U.shape[1]
    dU, dx, ep = np.random.rand(4, Nsys), 0.1 * np.random.rand(Nsys), 1e-6
    for ix in range(2):  # ping with explicit and implicit (baked-in) R_x effects
        if ix == 1:
            dx *= 0
            mfs.stagpoint_move(M)  # baked-in check
        v, v_u = [], []
        for ie in range(3):
            mfs.build_glob_sys(M)
            v.append(M.glob.R.copy())
            v_u.append(M.glob.R_U @ np.reshape(dU, 4 * Nsys, order="F") + M.glob.R_x @ dx)
            M.glob.U += ep * dU
            M.isol.xi += ep * dx
            if ix == 1:
                mfs.stagpoint_move(M)  # baked-in check: stagnation point moves
        M.glob.U -= 3 * ep * dU
        M.isol.xi -= 3 * ep * dx
        check_ping(ep, v, v_u, "global system, ix=%d" % (ix))

    # wake system ping
    dU, ep, v, v_u = np.random.rand(M.glob.U.shape[0], M.glob.U.shape[1]), 1e-5, [], []
    for ie in range(3):
        R, R_U, J = mfs.wake_sys(M, param)
        v.append(R)
        v_u.append(np.dot(R_U, np.reshape(dU[:, J], 4 * len(J), order="F")))
        M.glob.U += ep * dU
    M.glob.U -= 2 * ep * dU
    check_ping(ep, v, v_u, "wake system")

    # stagnation state ping
    M.oper.Re = 5e5
    mfs.init_thermo(M)
    (
        U,
        x,
    ) = (
        np.transpose(np.array([[5e-5, 1.1e-4, 0, 0.0348], [4.9e-5, 1.09e-4, 0, 0.07397]])),
        np.r_[5.18e-4, 1.1e-3],
    )
    dU, dx, ep, v, v_u = np.random.rand(4, 2), np.random.rand(2), 1e-6, [], []
    for ie in range(3):
        Ust, Ust_U, Ust_x, xst = mfs.stagnation_state(U, x)
        v.append(Ust)
        v_u.append(np.dot(Ust_U, np.reshape(dU, 8, order="F")) + np.dot(Ust_x, dx))
        U += ep * dU
        x += ep * dx
    check_ping(ep, v, v_u, "stagnation state")

    # force calculation ping
    Nsys, N, v, v_u = M.glob.U.shape[1], M.foil.N, [], []
    due = np.random.rand(N)
    dU = np.zeros((4, Nsys))
    dU[3, 0:N] = due
    da = 10
    ep = 1e-2
    for ie in range(3):
        mfs.calc_force(M)
        v.append(np.array([M.post.cl]))
        v_u.append(np.array([np.dot(M.post.cl_ue, due) + M.post.cl_alpha * da]))
        M.glob.U += ep * dU
        M.oper.alpha += ep * da
    M.glob.U -= 3 * ep * dU
    M.oper.alpha -= 3 * ep * da
    check_ping(ep, v, v_u, "lift calculation")
