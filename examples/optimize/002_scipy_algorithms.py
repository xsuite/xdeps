import xobjects as xo
import xdeps as xd
import numpy as np

def my_function(x):
    return [(x[0]-0.0001)**2, (x[1]-0.0003)**2, (x[2]+0.0005)**2, 3.0]

def scalar_func(x):
    return np.sum(np.array(my_function(x))**2)

x0 = [0., 0., 0.]
limits = [[-1, 1], [-1, 1], [-1, 1]]
targets = [0., 0., 0., 3.]
steps = [1e-6, 1e-6, 1e-6, 1e-6]
tols = [1e-12, 1e-12, 1e-12, 1e-12]

opt = xd.Optimize.from_callable(my_function, x0=x0, steps=steps, tar=targets, tols=tols, limits=limits)
opt.run_bfgs()
xo.assert_allclose(opt.get_merit_function().get_x(), [0.0001, 0.0003, -0.0005], atol=1e-6, rtol=0)

opt.reload(0)
opt.run_l_bfgs_b()
xo.assert_allclose(opt.get_merit_function().get_x(), [0.0001, 0.0003, -0.0005], atol=1e-6, rtol=0)

opt.reload(0)
opt.run_ls_trf()
xo.assert_allclose(opt.get_merit_function().get_x(), [0.0001, 0.0003, -0.0005], atol=1e-6, rtol=0)

opt.reload(0)
opt.run_ls_dogbox()
xo.assert_allclose(opt.get_merit_function().get_x(), [0.0001, 0.0003, -0.0005], atol=1e-6, rtol=0)

assert 'bfgs' in opt.log()['tag']
assert 'l-bfgs-b' in opt.log()['tag']
assert 'trf' in opt.log()['tag']
assert 'dogbox' in opt.log()['tag']