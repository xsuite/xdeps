import xobjects as xo
import xdeps as xd
import numpy as np


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

smf = opt.get_merit_function(rescale_x=(0, 1))
x = [0, 1, 0.5]
xo.assert_allclose(smf(x), [-1, 4, 0], atol=1e-10, rtol=0)

# Compute jacobian of rescaled merit function using finite differences
x0 = [0.1, 0.2, 0.3]
jsmf_ref = np.zeros((3, 3))
for ii in range(3):
    dx = 1e-6
    x1 = np.array(x0)
    x1[ii] += dx
    jsmf_ref[:, ii] = (smf(x1) - smf(x0))/dx

jsmf = smf.get_jacobian(x0)