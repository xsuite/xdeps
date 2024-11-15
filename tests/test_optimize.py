import numpy as np
import xobjects as xo
import xdeps as xd


def test_mirit_function_view():

    def my_function(x):
        return x

    x0 = [0., 0., 0.]

    opt = xd.Optimize.from_callable(my_function, x0=x0, steps=[1e-6, 1e-6, 1e-6],
                                    tar=[0., 0., 0.], tols=[1e-12, 1e-12, 1e-12],
                                    limits=[[-1, 2], [-1, 4], [-2, 2]])
    opt.solve()

    mf = opt.get_merit_function()
    jmf = mf.get_jacobian([0.5, 2, -1])
    xo.assert_allclose(jmf, [[1, 0, 0], [0, 1, 0], [
                       0, 0, 1]], atol=1e-6, rtol=0)

    # Check scalar function
    scalar_mf = opt.get_merit_function(return_scalar=True)
    xo.assert_allclose(scalar_mf([0.5, 2, -1]),
                       (mf([0.5, 2, -1])**2).sum(), atol=1e-10, rtol=0)
    #  Jacobian with finite differences
    jmf_scalare_ref = np.zeros(3)
    for ii in range(3):
        dx = 1e-10
        x1 = np.array([0.5, 2, -1])
        x1[ii] += dx
        jmf_scalare_ref[ii] = (scalar_mf(x1) - scalar_mf([0.5, 2, -1]))/dx
    xo.assert_allclose(jmf_scalare_ref, scalar_mf.get_jacobian(
        [0.5, 2, -1]), atol=1e-6, rtol=0)

    # Check rescaling
    scaled_mf = opt.get_merit_function(rescale_x=(0, 1))
    x = [0, 1, 0.5]
    xo.assert_allclose(scaled_mf(x), [-1, 4, 0], atol=1e-10, rtol=0)

    # Compute jacobian of rescaled merit function using finite differences
    x0 = [0.1, 0.2, 0.3]
    jsmf_ref = np.zeros((3, 3))
    for ii in range(3):
        dx = 1e-6
        x1 = np.array(x0)
        x1[ii] += dx
        jsmf_ref[:, ii] = (scaled_mf(x1) - scaled_mf(x0))/dx

    jsmf = scaled_mf.get_jacobian(x0)
    xo.assert_allclose(jsmf, jsmf_ref, atol=1e-6, rtol=0)

    scalar_scaled_smf = opt.get_merit_function(
        rescale_x=(0, 1), return_scalar=True)
    xo.assert_allclose(scalar_scaled_smf(
        x), (scaled_mf(x)**2).sum(), atol=1e-10, rtol=0)

    jscarlar_scaled_smf_ref = np.zeros(3)
    for ii in range(3):
        dx = 1e-10
        x1 = np.array(x0)
        x1[ii] += dx
        jscarlar_scaled_smf_ref[ii] = (
            scalar_scaled_smf(x1) - scalar_scaled_smf(x0))/dx

    xo.assert_allclose(jscarlar_scaled_smf_ref,
                       scalar_scaled_smf.get_jacobian(x0), atol=1e-6, rtol=0)

def test_optimize_basics():

    def my_function(x):
        return [(x[0]-0.0001)**2, (x[1]-0.0003)**2, (x[2]+0.0005)**2, 3.0]

    x0 = [0., 0., 0.]

    opt = xd.Optimize.from_callable(my_function, x0=x0, steps=[1e-6, 1e-6, 1e-6, 1e-6],
                            tar=[0., 0., 0., 3.], tols=[1e-12, 1e-12, 1e-12, 1e-12])
    opt.solve()

    xo.assert_allclose(opt.get_merit_function().get_x(), [0.0001, 0.0003, -0.0005], atol=1e-6, rtol=0)

def test_optimize_scipy_algorithms():

    def my_function(x):
        return [(x[0]-0.0001)**2, (x[1]-0.0003)**2, (x[2]+0.0005)**2, 3.0]

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