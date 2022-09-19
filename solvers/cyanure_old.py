from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import scipy
    from cyanure import BinaryClassifier


class Solver(BaseSolver):
    name = 'cyanure'

    install_cmd = 'conda'
    requirements = ['mkl', 'pip:cyanure-mkl']

    parameters = {
        'solver': ['catalyst-miso', 'qning-miso', 'qning-ista',  'auto',  'acc-svrg'],
    }

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        if (scipy.sparse.issparse(self.X) and
                scipy.sparse.isspmatrix_csc(self.X)):
            self.X = scipy.sparse.csr_matrix(self.X)

        self.solver_instance = BinaryClassifier(loss='logistic', penalty='l1',
                                       fit_intercept=False)
        if self.solver is None:
            self.solver = "auto"

        self.solver_parameter = dict(
            lambd=self.lmbd / self.X.shape[0], it0=1000000,
            tol=1e-12, verbose=False, solver=self.solver
        )

    def run(self, n_iter):
        self.solver_instance.fit(self.X, self.y, max_epochs=n_iter,
                        **self.solver_parameter)

    def get_result(self):
        return self.solver_instance.get_weights()