import pytest 
import numpy as np
from mfoil.solver import Mfoil
from mfoil_original import mfoil as Mfoil_original
from ping import ping_test


@pytest.mark.parametrize("airfoil", ["0012", '2312'])
@pytest.mark.parametrize("alpha", [1, 3])
@pytest.mark.parametrize("Re", [1e6])
def test_viscous(airfoil, alpha, Re):
    mfoil = Mfoil(naca=airfoil)
    mfoil.setoper(Re=Re, alpha=alpha)
    mfoil_original = Mfoil_original(naca=airfoil)
    mfoil_original.param.doplot = False
    mfoil_original.setoper(Re=Re, alpha=alpha)
    mfoil.param.verb = 0
    mfoil_original.param.verb = 0
    mfoil.solve()
    mfoil_original.solve()
    # ping_test(mfoil)
    if mfoil.glob.conv:
        assert np.allclose(mfoil.post.cp, mfoil_original.post.cp)
    else:
        assert not mfoil_original.glob.conv


@pytest.mark.parametrize("airfoil", ["0012"])
@pytest.mark.parametrize("cl", [0.4])
@pytest.mark.parametrize("Re", [1e6])
def test_viscous_target_cl(airfoil, cl, Re):
    mfoil = Mfoil(naca=airfoil)
    mfoil.setoper(Re=Re, cl=cl)
    mfoil_original = Mfoil_original(naca=airfoil)
    mfoil_original.param.doplot = False
    mfoil_original.setoper(Re=Re, cl=cl)
    mfoil.param.verb = 0
    mfoil_original.param.verb = 0
    mfoil.solve()
    mfoil_original.solve()
    if mfoil.glob.conv:
        assert np.allclose(mfoil.post.cp, mfoil_original.post.cp)
    else:
        assert not mfoil_original.glob.conv
        
@pytest.mark.parametrize("airfoil", ["0012", '2312'])
@pytest.mark.parametrize("alpha", range(0, 10, 2))
def test_inviscid(airfoil, alpha):
    mfoil = Mfoil(naca=airfoil)
    mfoil.setoper(alpha=alpha)
    mfoil_original = Mfoil_original(naca=airfoil)
    mfoil_original.param.doplot = False
    mfoil_original.setoper(alpha=alpha)
    mfoil.param.verb = 0
    mfoil_original.param.verb = 0
    mfoil.solve()
    mfoil_original.solve()

    if mfoil.glob.conv:
        assert np.allclose(mfoil.post.cp, mfoil_original.post.cp)
    else:
        assert not mfoil_original.glob.conv
        
if __name__ == "__main__":
    pytest.main()

