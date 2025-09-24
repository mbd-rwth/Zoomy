import numpy as np
from copy import deepcopy
import sympy
from sympy import Symbol, lambdify
from sympy import bspline_basis_set
from sympy.abc import x
from sympy import integrate, diff
from sympy import legendre
from sympy import lambdify
from sympy.functions.special.polynomials import chebyshevu


class Basisfunction:
    name = "Basisfunction"
    
    def bounds(self):
        return [0, 1]

    def basis_definition(self):
        x = Symbol("x")
        b = lambda k, x: x**k
        return [b(k, x) for k in range(self.level + 1)]
    
    def weight(self, z):
        return 1
    
    def weight_eval(self, z):
        x = Symbol("x")
        f = sympy.lambdify(x, self.weight(x))
        return f(z)

    def __init__(self, level=0, **kwargs):
        self.level = level
        self.basis = self.basis_definition(**kwargs)

    def get(self, k):
        return self.basis[k]

    def eval(self, k, z):
        return self.get(k).subs(x, z)
    
    def eval_psi(self, k, z):
        x = sympy.Symbol('x')
        psi = sympy.integrate(self.get(k), (x, self.bounds()[0], x))
        return psi.subs(x, z)

    def get_lambda(self, k):
        f = lambdify(x, self.get(k))

        def lam(z):
            if type(z) == int or type(z) == float:
                return f(z)
            elif type(z) == list or type(z) == np.ndarray:
                return np.array([f(xi) for xi in z])
            else:
                assert False

        return lam

    def plot(self, ax):
        X = np.linspace(self.bounds()[0], self.bounds()[1], 1000)
        for i in range(len(self.basis)):
            f = lambdify(x, self.get(i))
            y = np.array([f(xi) for xi in X])
            ax.plot(X, y, label=f"basis {i}")

    def reconstruct_velocity_profile(self, alpha, N=100):
        Z = np.linspace(self.bounds()[0], self.bounds()[1], N)
        u = np.zeros_like(Z)
        for i in range(len(self.basis)):
            b = lambdify(x, self.get(i))
            u[:] += alpha[i] * b(Z)
        return u
    
    def reconstruct_velocity_profile_at(self, alpha, z):
        u = 0
        for i in range(len(self.basis)):
            b = lambdify(x, self.get(i))
            u += alpha[i] * b(z)
        return u

    def reconstruct_alpha(self, velocities, z):
        n_basis = len(self.basis)
        alpha = np.zeros(n_basis)
        for i in range(n_basis):
            b = lambdify(x, self.get(i))
            nom = np.trapz(velocities * b(z) * self.weight(z), z)
            if type(b(z)) == int:
                den = b(z) ** 2
            else:
                den = np.trapz((b(z) * b(z)).reshape(z.shape), z)
            res = nom / den
            alpha[i] = res
        return alpha
    
    def project_onto_basis(self, Y):
        Z = np.linspace(self.bounds()[0], self.bounds()[1], Y.shape[0])
        n_basis = len(self.basis)
        alpha = np.zeros(n_basis)
        for i in range(n_basis):
            b = lambdify(x, self.get(i))
            alpha[i] = np.trapz(Y * b(Z) * self.weight_eval(Z), Z)
        return alpha

    def get_diff_basis(self):
        db = [diff(b, x) for i, b in enumerate(self.basis)]
        self.basis = db


class Monomials(Basisfunction):
    name = "Monomials"


class Legendre_shifted(Basisfunction):
    name = "Legendre_shifted"

    def basis_definition(self):
        x = Symbol("x")
        b = lambda k, x: legendre(k, 2 * x - 1) * (-1) ** (k)
        return [b(k, x) for k in range(self.level + 1)]
    
class Chebyshevu(Basisfunction):
    name = "Chebyshevu"
    
    def bounds(self):
        return [-1, 1]
    
    def weight(self, z):
        # do not forget to include the jacobian of the coordinate transformation in the weight
        return sympy.sqrt(1-z**2)

    def basis_definition(self):
        x = Symbol("x")
        b = lambda k, x: sympy.sqrt(2 / sympy.pi) * chebyshevu(k, x)
        return [b(k, x) for k in range(self.level + 1)]
    
class Legendre_DN(Basisfunction):
    name = "Legendre_DN - satifying no-slip and no-stress. This is a non-SWE basis"

    def bounds(self):
        return [-1, 1]

    def basis_definition(self):
        x = Symbol("x")
        def b(k, x):
            alpha = sympy.Rational((2*k+3), (k+2)**2)
            beta = -sympy.Rational((k+1),(k+2))**2
            return (legendre(k, x) ) + alpha * (legendre(k+1, x) ) + beta * (legendre(k+2, x))
        #normalizing makes no sence, as b(k, 0) = 0 by construction
        return [b(k, x) for k in range(self.level + 1)]


class Spline(Basisfunction):
    name = "Spline"

    def basis_definition(self, degree=1, knots=[0, 0, 0.001, 1, 1]):
        x = Symbol("x")
        basis = bspline_basis_set(degree, knots, x)
        return basis


class OrthogonalSplineWithConstant(Basisfunction):
    name = "OrthogonalSplineWithConstant"

    def basis_definition(self, degree=1, knots=[0, 0, 0.5, 1, 1]):
        x = Symbol("x")

        def prod(u, v):
            return integrate(u * v, (x, 0, 1))

        basis = bspline_basis_set(degree, knots, x)
        add_basis = [1]
        # add_basis = [sympy.Piecewise((0, x<0.1), (1, True))]
        basis = add_basis + basis[:-1]
        orth = deepcopy(basis)
        for i in range(1, len(orth)):
            for j in range(0, i):
                orth[i] -= prod(basis[i], orth[j]) / prod(orth[j], orth[j]) * orth[j]

        for i in range(len(orth)):
            orth[i] /= sympy.sqrt(prod(orth[i], orth[i]))

        return orth
