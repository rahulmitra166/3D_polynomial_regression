import numpy as np
import itertools
import sympy as sp
import multiprocessing

class PolynomialRegression:
    
    def __init__(self, degree):
        self.degree = degree
    
    def create_polynomial(self):
        x, y, z = sp.symbols('x y z')
        px = sum(sp.Symbol(f'ax{i}') * x**i for i in range(self.degree + 1))
        py = sum(sp.Symbol(f'ay{i}') * y**i for i in range(self.degree + 1))
        pz = sum(sp.Symbol(f'az{i}') * z**i for i in range(self.degree + 1))
        self.polynomial = sp.expand(px * py * pz)

    def _compute_term(self, args):
        P, term = args
        coeff, monomial = term.as_coeff_Mul()
        coeffs = {str(s): monomial.as_coeff_exponent(s)[1] for s in monomial.free_symbols}
        return P[:, 0] ** coeffs.get('x', 0) * P[:, 1] ** coeffs.get('y', 0) * P[:, 2] ** coeffs.get('z', 0)

    def create_design_matrix(self, P):
        n = P.shape[0]
        X = np.ones((n, len(self.polynomial.as_ordered_terms())))

        with multiprocessing.Pool() as pool:
            results = pool.map(self._compute_term, [(P, term) for term in self.polynomial.as_ordered_terms()])

        for i, result in enumerate(results):
            X[:, i] = result

        return X

    def fit(self, P, V):
        self.P = P
        self.V = V
        self.create_polynomial()
        X = self.create_design_matrix(P)
        self.coefficients, _, _, _ = np.linalg.lstsq(X, V, rcond=None)
    
    def predict(self, P):
        X = self.create_design_matrix(P)
        return X @ self.coefficients
    
    def get_polynomial_string(self):
        terms = list(self.polynomial.as_ordered_terms())
        term_strings = []
        for coef, term in zip(self.coefficients, terms):
            term_string = f"{coef:.2f} * {term}"
            term_strings.append(term_string)
        return " + ".join(term_strings)
