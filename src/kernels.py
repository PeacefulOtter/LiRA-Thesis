
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, ExpSineSquared, RationalQuadratic, Matern, WhiteKernel


def constant_kernel(x, y):
    constant_value = (y + x) / 2
    constant_value_bounds = (x, y)
    return ConstantKernel(constant_value=constant_value, constant_value_bounds=constant_value_bounds)

def rbf_kernel(x, y):
    length_scale_bounds = (x, y) 
    return RBF(length_scale_bounds=length_scale_bounds)

def expsinesquared_kernel():
    length_scale = 1.0
    periodicity = 1.0
    return ExpSineSquared(length_scale=length_scale, periodicity=periodicity)

def rationalquad_kernel():
    length_scale = 1.0
    alpha = 0.1
    alpha_bounds = (1e-5, 1e15)
    return RationalQuadratic(length_scale=length_scale, alpha=alpha, alpha_bounds=alpha_bounds)

def white_kernel(noise=1.0, noise_level_bounds=(1e-05, 100000.0)):
    return WhiteKernel(noise_level=noise, noise_level_bounds=noise_level_bounds)

def matern_kernel():
    length_scale = 1.0
    length_scale_bounds = (1e-5, 1e5)
    nu = 1.5
    return Matern(length_scale=length_scale, length_scale_bounds=length_scale_bounds, nu=nu)


def default_kernel():
    return 1.0 * rbf_kernel() + white_kernel()