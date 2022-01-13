import abc
import scipy.interpolate
import numpy as np


class _Element(abc.ABC):
    """
    Coeff's can be 1- or 2-dimensional; each row gives the
    coefficients for a specific dimension. Return is of format
    arr[derivative, dimension, timepoint]
    """

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self._endpoints = np.r_[low, high]

    def contains(self, timepoints):
        timepointsarr = np.asarray(timepoints)
        return np.logical_and(timepointsarr >= self.low, timepointsarr <= self.high)

    def derivatives_endpoints(self, coeffs, derivatives):
        return self.derivatives(self._endpoints, coeffs, derivatives)

    @abc.abstractmethod
    def derivatives(self, ts, coeffs, derivatives):
        pass

    @abc.abstractmethod
    def __call__(self, ts, coeffs):
        pass


class ChebyshevNodes:
    @staticmethod
    def _chebyshev_nodes(order):
        indices = np.arange(1, order + 1)
        return np.cos(np.pi * (2 * indices - 1) / (2 * order))

    @staticmethod
    def _transform_nodes(nodes, low, high):
        nodes = 0.5 * (nodes + 1)
        return nodes * (high - low) + low

    @staticmethod
    def _get_nodes(order, low, high):
        basenodes = ChebyshevNodes._chebyshev_nodes(order)
        return ChebyshevNodes._transform_nodes(basenodes, low, high)


class _KroghElement(_Element, ChebyshevNodes):
    """
    Interpolating polynomial basis functions.
    For numerical stability, we take interpolations at (Chebyshev zeros, coefficients).
    """

    def derivatives(self, ts, coeffs, derivatives):
        return self._evaluate(ts, coeffs, derivatives)

    def __call__(self, ts, coeffs):
        return self._evaluate(ts, coeffs, 0)

    def _evaluate(self, ts, coeffs, derivatives):
        if ts.ndim > 1:
            raise ValueError("ts must be 0 or 1 dimensional")
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape((1, -1))
        coeffs = np.asarray(coeffs)
        nodes = _KroghElement._get_nodes(coeffs.shape[1], self.low, self.high)
        interpolator = scipy.interpolate.KroghInterpolator(nodes, coeffs.T)
        return interpolator.derivatives(ts, derivatives + 1).transpose((0, 2, 1))


class _FiniteElements(abc.ABC):
    @property
    @abc.abstractmethod
    def _element(self, low, high):
        pass

    def __init__(self, mesh):
        self.mesh = np.sort(mesh)
        self.elements = [
            self._element(low, high) for low, high in zip(mesh[:-1], mesh[1:])
        ]

    def __call__(self, ts, coeffs):
        return self._evaluate_at(ts, coeffs, self.elements)

    def __iter__(self):
        self._iter_index = -1
        return self

    def __next__(self):
        self._iter_index += 1
        if self._iter_index >= len(self.elements):
            raise StopIteration
        return self.elements[self._iter_index]

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def _evaluate_at(self, ts, coeffs, basisfuncs, n_derivatives=0):
        ts = np.asarray(ts)
        coeffs = np.asarray(coeffs)
        coeffs = coeffs.reshape((1, -1)) if coeffs.ndim == 1 else coeffs
        ret = np.zeros((n_derivatives + 1, coeffs.shape[0], ts.size))
        coeff_lists = coeffs.reshape(
            (coeffs.shape[0], self.mesh.size - 1, -1)
        ).transpose((1, 0, 2))
        element_membership = [element.contains(ts) for element in self.elements]
        for f, coeff, cond in zip(basisfuncs, coeff_lists, element_membership):
            ret[:, :, cond] = f(ts[cond], coeff)
        return ret.squeeze()

    def derivatives(self, ts, coeffs, derivatives):
        def basisfunc(element):
            def derivative(ts, coeffs):
                return element.derivatives(ts, coeffs, derivatives)

            return derivative

        basis_funcs = [basisfunc(e) for e in self.elements]
        return self._evaluate_at(ts, coeffs, basis_funcs, derivatives)

    def _evaluate(self, coeffs, basisfuncs):
        n_dim = 1 if coeffs.ndim == 1 else coeffs.shape[0]
        coeff_lists = coeffs.reshape((n_dim, self.mesh.size - 1, -1)).transpose(
            (1, 0, 2)
        )
        return np.array([x(coeff) for x, coeff in zip(basisfuncs, coeff_lists)])

    def meshpoint_derivatives(self, coeffs, derivatives):
        def basisfunc(element):
            def elem_derivatives(these_coeffs):
                return element.derivatives_endpoints(these_coeffs, derivatives)

            return elem_derivatives

        basis_funcs = [basisfunc(e) for e in self.elements]
        return self._evaluate(coeffs, basis_funcs)


class KroghFiniteElements(_FiniteElements):
    _element = _KroghElement
