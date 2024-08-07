import numpy as np
from mfoil.utils import norm2, dist, atan2
from mfoil.geometry import panel_info


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
    P2 = x * P1 + (0.5 / np.pi) * (
        0.5 * r2**2 * logr2 - 0.5 * r1**2 * logr1 - r2**2 / 4 + r1**2 / 4
    )

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
    logr1, theta1, theta2 = (
        (0, np.pi, np.pi) if (r1 < ep) else (np.log(r1), theta1, theta2)
    )
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
    logr1, theta1, theta2 = (
        (0, np.pi, np.pi) if (r1 < ep) else (np.log(r1), theta1, theta2)
    )
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
    logr1, theta1, theta2 = (
        (0, np.pi, np.pi) if (r1 < ep) else (np.log(r1), theta1, theta2)
    )
    logr2, theta1, theta2 = (0, 0, 0) if (r2 < ep) else (np.log(r2), theta1, theta2)

    # streamfunction components
    P1 = (0.5 / np.pi) * (x * (theta1 - theta2) + theta2 * d + z * logr1 - z * logr2)
    P2 = x * P1 + (0.5 / np.pi) * (
        0.5 * r2**2 * theta2 - 0.5 * r1**2 * theta1 - 0.5 * z * d
    )

    # influence coefficients
    a = P1 - P2 / d
    b = P2 / d

    return a, b