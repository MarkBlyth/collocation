import abc
import numpy as np
from finiteelements import KroghFiniteElements


class _Collocation:
    @property
    @abc.abstractmethod
    def _FEClass(self):
        pass

    def __init__(self, mesh, smoothness):
        self.low = mesh[0]
        self.high = mesh[-1]
        self._mesh = mesh
        self._smoothness = smoothness
        self._finite_elements = self._FEClass(mesh)

    def _internal_continuity_errors(self, coeffs):
        n_dims = 1 if coeffs.ndim == 1 else coeffs.shape[0]
        meshpoint_derivatives = self._finite_elements.meshpoint_derivatives(
            coeffs, self._smoothness
        )
        internal_errors = np.zeros((self._smoothness + 1, n_dims, self._mesh.size - 2))
        for row, (element1, element2) in enumerate(
            zip(meshpoint_derivatives[:-1], meshpoint_derivatives[1:])
        ):
            internal_errors[:, :, row] = element2[:, :, 0] - element1[:, :, 1]
        return internal_errors.ravel()

    def _periodic_boundary_errors(self, coeffs):
        meshpoint_derivatives = self._finite_elements.meshpoint_derivatives(
            coeffs, self._smoothness
        )
        return (
            meshpoint_derivatives[-1][:, :, 1] - meshpoint_derivatives[0][:, :, 0]
        ).ravel()

    def evaluate_solution(self, ts, coeffs):
        return self._finite_elements(ts, coeffs)

    def evaluate_derivatives(self, ts, coeffs, derivatives):
        return self._finite_elements.derivatives(ts, coeffs, derivatives)

    @abc.abstractmethod
    def _collocation_condition(self, coeffs, reference):
        pass

    @abc.abstractmethod
    def _continuity_errors(self, coeffs):
        pass

    def __call__(self, coeffs, reference):
        return np.r_[
            self._collocation_condition(coeffs, reference),
            self._continuity_errors(coeffs),
        ]


class _CBCCollocation(_Collocation):
    @staticmethod
    def _mean_over_subintervals(ts, ys, subintervals):
        ret = np.zeros(subintervals.size - 1)
        for i in range(ret.size):
            ret[i] = np.mean(
                ys[np.logical_and(ts > subintervals[i], ts <= subintervals[i + 1])]
            )
        return ret

    def _continuity_errors(self, coeffs):
        return np.r_[
            self._internal_continuity_errors(coeffs),
            self._periodic_boundary_errors(coeffs),
        ]

    def _get_submesh(self, n_subintervals):
        submesh_rows = np.zeros((len(self._finite_elements), n_subintervals))
        for i, element in enumerate(self._finite_elements):
            submesh_rows[i] = np.linspace(
                element.low, element.high, n_subintervals + 1
            )[1:]
        return np.r_[self._finite_elements[0].low, submesh_rows.ravel()]

    def _collocation_condition(self, coeffs, reference):
        ts, ys = reference
        residuals = self._finite_elements(ts, coeffs) - ys
        n_subintervals = coeffs.size // len(self._finite_elements) - (
            self._smoothness + 1
        )
        submesh = self._get_submesh(n_subintervals)
        return _CBCCollocation._mean_over_subintervals(ts, residuals, submesh)


class _OrthoCollocationBVP(_Collocation):
    def __init__(self, mesh):
        super().__init__(mesh, 0)

    def _get_gauss_points(self, n_coll):
        roots, _ = np.polynomial.legendre.leggauss(n_coll)
        roots = 0.5 * (roots + 1)
        gausspoints = np.zeros((self._finite_elements.mesh.size - 1, n_coll))
        for i, elem in enumerate(self._finite_elements):
            gausspoints[i, :] = (elem.high - elem.low) * roots + elem.low
        return gausspoints.ravel()

    def _continuity_errors(self, coeffs):
        return self._internal_continuity_errors(coeffs)

    def _collocation_condition(self, coeffs, reference):
        if coeffs.ndim == 1:
            n_coll = coeffs.size // len(self._finite_elements) - (self._smoothness + 1)
        else:
            n_coll = coeffs.shape[1] // len(self._finite_elements) - (
                self._smoothness + 1
            )
        gausspoints = self._get_gauss_points(n_coll)
        soln_derivatives = self._finite_elements.derivatives(gausspoints, coeffs, 1)[1]
        real_derivatives = reference(
            gausspoints, self.evaluate_solution(gausspoints, coeffs)
        )
        return (soln_derivatives - real_derivatives).ravel()

    def get_default_zero_problem(self, bvp, boundary_conds, ndim):
        lhs, rhs = self._finite_elements.mesh[0], self._finite_elements.mesh[-1]

        def zero_problem(coeffs):
            coeffs = coeffs.reshape((ndim, -1))
            start, end = self.evaluate_solution([lhs, rhs], coeffs)
            return np.r_[self.__call__(coeffs, bvp), boundary_conds(start, end)]

        return zero_problem


class KroghBVP(_OrthoCollocationBVP):
    _FEClass = KroghFiniteElements


class KroghCBCCollocation(_CBCCollocation):
    _FEClass = KroghFiniteElements

class KroghCustomCollocation(_Collocation):
    _FEClass = KroghFiniteElements

    @abc.abstractmethod
    def _collocation_condition(self, coeffs, reference):
        pass

    @abc.abstractmethod
    def _continuity_errors(self, coeffs):
        pass
