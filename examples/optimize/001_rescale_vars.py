import xobjects as xo
import xdeps as xd


def my_function(x):
    return x

x0 = [0., 0., 0.]

opt = xd.Optimize.from_callable(my_function, x0=x0, steps=[1e-6, 1e-6, 1e-6],
                        tar=[0., 0., 0.], tols=[1e-12, 1e-12, 1e-12],
                        limits=[[-1, 2], [-1, 4], [-2, 2]])
opt.solve()

mf = opt.get_merit_function()
jmf = mf.get_jacobian([0.5, 2 , -1])

xo.assert_allclose(jmf, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], atol=1e-6, rtol=0)
