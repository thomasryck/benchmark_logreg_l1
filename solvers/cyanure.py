import warnings
from sklearn.exceptions import ConvergenceWarning
from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    import scipy
    from cyanure.estimators import Classifier


class Solver(BaseSolver):
    name = 'cyanure.estimators'

    install_cmd = 'conda'
    requirements = ['cyanure']

    parameters = {
        'solver': ['catalyst-miso', 'qning-miso', 'qning-ista',  'auto',  'acc-svrg'],
    }

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        if (scipy.sparse.issparse(self.X) and
                scipy.sparse.isspmatrix_csc(self.X)):
            self.X = scipy.sparse.csr_matrix(self.X)

        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        if self.solver is None:
            self.solver = "auto"

        self.solver_parameter = dict(
        lambda_1=self.lmbd / self.X.shape[0], duality_gap_interval=10,
        tol=1e-12, verbose=False, solver=self.solver
        )

        self.solver_instance = Classifier(loss='logistic', penalty='l1',
                                       fit_intercept=False,
                        **self.solver_parameter)
       

    def run(self, n_iter):
        self.solver_instance.max_iter = n_iter
        self.solver_instance.fit(self.X, self.y)

    def get_result(self):
        return self.solver_instance.get_weights()
