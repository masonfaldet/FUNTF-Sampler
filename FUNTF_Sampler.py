import os
import matplotlib.pyplot as plt
import numpy as np
import polytope as polytope
from scipy.linalg import expm
from tqdm import tqdm

EPSILON = 1e-5
DEFAULT = object()


class Frame:
    """
    A class representing a frame in the complex vector space ℂ^d.

    Parameters
    ----------
    mat : array-like
        An array-like object whose entries define the frame matrix. Each column of the matrix
        represents a frame vector. The input is automatically converted into a complex matrix.

    Attributes
    ----------
    _frame : numpy.matrix
        The internal complex matrix representing the frame (with columns as frame vectors).
    dimension : int
        The ambient space dimension (number of rows in the frame matrix).
    num_vectors : int
        The number of frame vectors (number of columns in the frame matrix).
    EPSILON : float
        A small numerical tolerance used for floating-point comparisons (assumed to be defined
        globally).

    Methods
    -------
    from_vectors(*args)
        Class method to create a Frame instance from individual vectors provided as lists or 1-D arrays.
    upper_bound()
        Returns the upper frame bound, computed as the square of the largest singular value of the frame matrix.
    lower_bound()
        Returns the lower frame bound, computed as the square of the smallest singular value of the frame matrix.
    is_tight()
        Checks if the frame is tight, meaning that its upper and lower bounds are nearly equal.
    is_parseval()
        Checks if the frame is a Parseval frame (i.e., a tight frame with frame bound equal to 1).
    is_equal_norm()
        Verifies whether all frame vectors have the same norm.
    is_unit_norm()
        Verifies whether all frame vectors are of unit norm.
    is_equiangular()
        Checks if the frame is equiangular; that is, whether the absolute values of the inner products
        between distinct frame vectors are all equal.
    get_coefficients(x)
        Computes and returns the coefficients (inner products) of an input vector x with the frame vectors.
    get_frame_operator()
        Returns the frame operator, defined as the product of the frame matrix with its Hermitian transpose.
    get_analysis_operator()
        Returns the analysis operator (the Hermitian transpose of the frame matrix).
    get_synthesis_operator()
        Returns the synthesis operator (the frame matrix itself).
    coherence()
        Computes the coherence of the frame, which is the maximum absolute off-diagonal entry in the Gram matrix.
    get_partial_frame_operator(k)
        Returns the frame operator corresponding to the first k frame vectors.
    get_eigensteps()
        Computes the eigensteps: for each k from 1 to the total number of vectors, returns the eigenvalues
        and eigenvectors of the partial frame operator formed by the first k vectors.

    Notes
    -----
    - Upon initialization, a singular value decomposition (SVD) is performed on the frame matrix to ensure
      that the set of vectors is full rank (i.e., it spans ℂ^d).
    - The class assumes that a global constant EPSILON is defined to set the numerical tolerance for comparisons.

    """


    def __init__(self, mat):
        self._frame = np.mat(mat + 1j * (0 * mat))
        self.EPSILON = EPSILON
        self.dimension = self._frame.shape[0]
        self.num_vectors = self._frame.shape[1]

        # singular value decomposition used in several methods
        self._U, self._s, self._V = np.linalg.svd(self._frame)

        # frame if and only if spanning set, that is, full rank
        assert not np.any(np.isclose(self._s, 0.0, self.EPSILON / 2,
                                     self.EPSILON / 2))

    @classmethod
    def from_vectors(cls, *args):
        # each vector (as lists or 1-D arrays) passed in must have same length
        lengths = [len(v) for v in args]
        assert all(l == lengths[0] for l in lengths)
        return cls(np.mat(args).T)

    def upper_bound(self):
        return self._s[0] ** 2

    def lower_bound(self):
        return self._s[-1] ** 2

    def is_tight(self):
        return np.isclose(self.upper_bound(), self.lower_bound(),
                          self.EPSILON / 2, self.EPSILON / 2)

    def is_parseval(self):
        return self.is_tight() and np.isclose(self.upper_bound(), 1.0,
                                              self.EPSILON / 2, self.EPSILON / 2)

    def is_equal_norm(self):
        norms = np.sqrt(np.diag(self._frame.H * self._frame))
        return np.allclose(norms, norms[0], self.EPSILON / 2, self.EPSILON / 2)

    def is_unit_norm(self):
        norms = np.sqrt(np.diag(self._frame.H * self._frame))
        return np.allclose(norms, 1.0, self.EPSILON / 2, self.EPSILON / 2)

    def is_equiangular(self):
        ips = self._frame.H * self._frame
        abs_ips = []
        for i in range(self.num_vectors):
            for j in range(i + 1, self.num_vectors):
                abs_ips.append(abs(ips[i, j]))
        return np.allclose(abs_ips, abs_ips[0], self.EPSILON / 2, self.EPSILON / 2)

    def get_coefficients(self, x):
        vec = np.array(x).reshape((len(x),))
        return (vec * self._frame).A.reshape((self.num_vectors,))

    def get_frame_operator(self):
        return self._frame * self._frame.H

    def get_analysis_operator(self):
        return self._frame.H

    def get_synthesis_operator(self):
        return self._frame

    def coherence(self):
        G = self._frame.H * self._frame
        I = np.mat(np.eye(self.num_vectors, dtype=int))
        return np.max(np.abs(G) - I)

    def get_partial_frame_operator(self, k):
        assert k < self.num_vectors + 1 and k > 0
        return self._frame[:, :k] * self._frame[:, :k].H

    def get_eigensteps(self):
        e_list = np.zeros((self.num_vectors, self.dimension))
        e_vec_list = []
        for k in range(1, self.num_vectors + 1):
            eigenvalues, eigenvectors = np.linalg.eigh(self.get_partial_frame_operator(k))
            e_list[k - 1, :] = eigenvalues[::-1]
            e_vec_list.append(eigenvectors[:, ::-1])
        return e_list, e_vec_list


class Eigensteps:
    '''
    An eigenstep sequence is a doubly-indexed sequence of sequences obeying
    certain conditions
    '''

    def __init__(self, sequence_matrix):
        '''
        Takes a doubly-indexed eigenstep sequence as a matrix of the form
        [ [ l_(1,1)  0        0        0        ...  0       ]
          [ l_(2,1)  l_(2,2)  0        0        ...  0       ]
          [ l_(3,1)  l_(3,2)  l_(3,3)  0        ...  0       ]
              ...      ...      ...      ...           ...
          [ l_(N,1)  l_(N,2)  l_(N,3)  l_(N,4)  ...  l_(N,M) ] ]
        '''

        self._eigensteps = np.mat(sequence_matrix)
        self.depth = self._eigensteps.shape[0]
        self.width = self._eigensteps.shape[1]
        self.EPSILON = EPSILON

        # verify interlacing property
        assert self._eigensteps[0, 0] >= 0  # first entry is nonnegative
        assert np.all(self._eigensteps[0, 1:] == 0)  # rest of first row is zero
        # all rows interlace
        for n in range(1, self.depth):
            for m in range(self.width):
                assert self._eigensteps[n, m] >= self._eigensteps[n - 1, m]
                if (m + 1) < self.width:
                    assert self._eigensteps[n - 1, m] >= self._eigensteps[n, m + 1]

        # verify nonincreasing row sums
        assert np.all(np.diff(self.get_mu_sequence()) < self.EPSILON)

    @classmethod
    def from_sequences(cls, *args):
        # must pass in sequences as lists of nondecreasing length
        lengths = [len(args[n]) for n in range(len(args))]
        assert np.all(np.diff(lengths) >= 0)
        try:
            # list of differences should be [1, ..., 1, 0, ..., 0]
            first_zero = list(np.diff(lengths)).index(0)
            assert np.all(np.diff(lengths)[:first_zero] == 1)
            assert np.all(np.diff(lengths)[first_zero:] == 0)
        except ValueError:
            pass  # lengths are strictly increasing, this is okay

        depth = len(args)
        width = len(args[-1])
        mat = []
        for n in range(depth):
            # put into matrix, padding out with zeroes
            mat.append(args[n] + ([0] * (width - len(args[n]))))

        return cls(np.mat(mat))

    def get_mu_sequence(self):
        '''
        Returns the (nonincreasing) sequence of mu values
        '''
        return np.diff([0] + [np.sum(self._eigensteps[n, :])
                              for n in range(self.depth)])

    def get_row(self, row_index):
        return self._eigensteps[row_index, :]

    def get_column(self, col_index):
        return self._eigensteps[:, col_index]

    def get_element(self, row_index, col_index):
        return self._eigensteps[row_index, col_index]


class EigenstepSystem:
    """
     Represents the geometric structure of eigensteps associated with a frame.

     In frame theory, eigensteps are defined as the eigenvalues of the partial frame operators of a frame.
     This class encapsulates the geometric structure underlying these eigensteps by constructing an
     archetypal eigenstep matrix and formulating the associated system of linear inequalities that the
     eigensteps must satisfy (e.g., ensuring non-increasing diagonals, which reflects the interlacing
     properties of eigenvalues in partial frame operators).

     The system is parameterized by:
       - N: the number of frame vectors,
       - d: the dimension of each frame vector, and
       - mu: the normalized squared norm of the frame vectors.

     Internally, the class uses a StandardFormProgram (assumed to be defined elsewhere) to generate:
       * `table`: an archetypal eigenstep matrix whose entries are given in a symbolic form,
       * `ineqs`: a list of inequalities the eigensteps must satisfy, and
       * `A` and `b`: the half-space (H-) representation of these linear inequalities (i.e., A * x <= b).

     Parameters
     ----------
     N : int
         The number of frame vectors.
     d : int
         The dimension of each frame vector.
     mu : float
         The normalized squared norm of the frame vectors.

     Attributes
     ----------
     N : int
         Number of frame vectors.
     d : int
         Dimension of frame vectors.
     mu : float
         Normalized squared norm of frame vectors.
     table : array-like
         Archetype of a generic eigenstep matrix, generated by a StandardFormProgram.
     ineqs : list
         List of inequalities that the eigensteps must satisfy (e.g., enforcing non-increasing diagonals).
     A : array-like
         Matrix in the H-representation of the linear inequalities (i.e., defining the half-spaces A * x <= b).
     b : array-like
         Right-hand side vector in the H-representation of the linear inequalities.

     Methods
     -------
     make_eigenstep_mat(vec)
         Constructs an eigenstep matrix by substituting numerical values from a parameter vector into
         the archetypal table. The input vector must have length (d - 1) * (N - d - 1). In the table,
         symbolic tokens (e.g., 'x1', 'x2', ...) are replaced by corresponding entries from the vector,
         and the resulting expressions are evaluated to produce a concrete eigenstep matrix.
    """

    def __init__(self, N, d, mu):
        self.N = N
        self.d = d
        self.mu = mu
        program = StandardFormProgram(N, d, mu)
        table = program.make_table()
        self.table = table
        ineqs = program.get_inequalities(table)
        self.ineqs = ineqs
        [A, b] = program.get_A_and_B(self.ineqs)
        self.A = A
        self.b = b

    def make_eigenstep_mat(self, vec):
        assert (len(vec) == (self.d - 1) * (self.N - self.d - 1))
        mat = np.zeros((self.N, self.d))
        for i in range(self.N):
            for j in range(self.d):
                s = str(self.table[i, j])
                tokens = s.split(" ")
                for token in tokens:
                    if "x" in token:
                        s = s.replace(token[token.index("x"):len(token) + 1],
                                      str(vec[int(token[token.index("x") + 1:len(token) + 1])]))
                sum = 0
                tokens = s.split(" ")
                for token in tokens:
                    if "-" in token:
                        sum = sum - float(token[1:len(token) + 1])
                    else:
                        sum = sum + float(token)
                mat[i, j] = sum
        return mat


class StandardFormProgram:
    """
    Constructs a standard form representation of the eigenstep system for a frame.

    This class generates a symbolic table representing the eigenstep relationships and extracts a list of
    defining inequalities. These inequalities can then be parsed into a standard form representation
    A*x <= b, where the variable vector x has dimension (d - 1) * (N - d - 1), ensuring that the
    corresponding polytope is full-dimensional.

    The eigenstep table is an N x d array of strings. Each entry in the table is either a constant (a dependent
    eigenstep derived from the parameters N, d, and mu) or a symbolic variable representing independent eigensteps,
    arranged to capture the relationships among the eigensteps.

    Parameters
    ----------
    N : int
        The number of frame vectors.
    d : int
        The dimension of each frame vector.
    mu : float
        The normalized squared norm of the frame vectors.

    Attributes
    ----------
    N : int
        Number of frame vectors.
    d : int
        Dimension of the frame vectors.
    mu : float
        Normalized squared norm of the frame vectors.

    Methods
    -------
    make_table():
        Constructs and returns the eigenstep table as an N x d array of strings. The table encodes both
        constant values and symbolic variables that define the relationships among the eigensteps.
    get_inequalities(table):
        Processes the eigenstep table to extract and return a list of inequality strings that the eigensteps
        must satisfy. These inequalities enforce the necessary nonincreasing structure (e.g., along diagonals
        and rows) inherent to the eigenstep system.
    get_A_and_B(ineqs):
        Transforms a list of inequality strings into the half space representation of the polytope whose defining
        inequalities are provided in ineqs.
    """

    def __init__(self, N, d, mu):
        self.N = N
        self.d = d
        self.mu = mu

    def make_table(self):

        table = np.empty([self.N, self.d], dtype=object)

        for i in range(self.d):
            for j in range(self.d - i):
                table[i][j] = str((self.N * self.mu) / self.d)

        ind = 0
        for i in range(self.d, self.N - 1):
            for j in range(self.d - 1):
                table[i - j][j] = "x%d" % ind
                ind += 1

        for i in range(self.d):
            relation = str(i + 1)
            for j in range(i):
                entry = table[self.N - i - 1][j]
                if "x" not in entry:
                    relation = str(float(relation) - float(entry))
                else:
                    relation = relation + " -" + entry
            table[self.N - i - 1][i] = relation

        for i in range(1, self.N - self.d):
            relation = str(self.N - i)
            for j in range(self.d - 1):
                entry = table[i][j]
                if "x" not in entry:
                    relation = str(float(relation) - float(entry))
                else:
                    relation = relation + " -" + entry
            table[i][self.d - 1] = relation

        for i in range(self.d):
            for j in range(i + 1, self.d):
                table[self.N - i - 1][j] = str(0)
        return table

    def get_inequalities(self, table):
        list = []
        for i in range(1, self.N):
            for j in range(self.d):
                if (table[i, j] != table[i - 1, j]):
                    if (all(table[i, j] + "<=" + table[i - 1, j] != li for li in list)):
                        list.append(table[i, j] + "<=" + table[i - 1, j])
                if (j != self.d - 1 and table[i, j] != table[i - 1, j + 1]):
                    if (all(table[i - 1, j + 1] + "<=" + table[i, j] != li for li in list)):
                        list.append(table[i - 1, j + 1] + "<=" + table[i, j])
            for i in range(2, self.N - self.d + 1):
                if (all("0" + "<=" + table[i, self.d - 1] != li for li in list)):
                    list.append("0" + "<=" + table[i, self.d - 1])
        return list

    def get_A_and_B(self, list):
        A = np.zeros([len(list), (self.d - 1) * (self.N - self.d - 1)])
        B = np.zeros([len(list)])
        row = 0
        for li in list:
            if ("x" in li):
                [lhs, rhs] = li.split("<=")
                lhs_tokens = lhs.split(" ")
                rhs_tokens = rhs.split(" ")
                for token in lhs_tokens:
                    if ("x" not in token):
                        if ("-" in token):
                            rhs_tokens.append(token[1:len(token) + 1])
                        else:
                            rhs_tokens.append("-" + token)
                lhs_tokens[:] = [token for token in lhs_tokens if "x" in token]

                for token in rhs_tokens:
                    if ("x" in token):
                        if ("-" in token):
                            lhs_tokens.append(token[1:len(token) + 1])
                        else:
                            lhs_tokens.append("-" + token)
                rhs_tokens[:] = [token for token in rhs_tokens if "x" not in token]

                for token in lhs_tokens:
                    [coeff, var] = token.split("x")
                    if (coeff == ''):
                        coeff = "1"
                    if (coeff == '-'):
                        coeff = "-1"
                    A[row, int(var)] = float(coeff)
                sum = 0
                for token in rhs_tokens:
                    sum = sum + float(token)
                B[row] = sum
                row = row + 1

        A = A[~np.all(A == 0, axis=1)]
        B = B[0:len(A)]
        return [A, B]


def nu_i(row_index, d, N):
    assert (row_index > 0) and (row_index < N - 2)
    return (row_index + 1) - ((float(N) / d) * max(0.0, d - N + row_index + 1))


def random_FUNTF_eigensteps_bb(d, N, n_samples=1):

    """
    Generate random eigensteps for a finite unit norm tight frame (FUNTF) via rejection sampling.

    This function constructs an eigenstep system for a frame with N vectors in ℂ^d (with unit
    norm squared, mu=1), and then defines the associated polytope of independent eigensteps using
    its H-representation (A*x ≤ b). A bounding box for the polytope is computed, and points are
    sampled uniformly from this box. Rejection sampling is employed to ensure that the sampled
    point lies within the polytope. The independent eigenstep parameters from the valid sample are
    then substituted into the eigenstep table to construct a full eigenstep matrix. Finally, an
    Eigensteps object is created from this matrix.

    Parameters
    ----------
    d : int
        The dimension of the frame vectors.
    N : int
        The number of frame vectors.
    n_samples : int, optional
        The number of random eigenstep samples to generate (default is 1).

    Returns
    -------
    Eigensteps or list of Eigensteps
        If n_samples is 1, returns a single Eigensteps object corresponding to a random sample.
        If n_samples is greater than 1, returns a list of Eigensteps objects.
    """
    sys = EigenstepSystem(N, d, 1)
    A = sys.A
    b = sys.b
    P = polytope.Polytope(A=A, b=b, fulldim=True)
    l, u = polytope.bounding_box(P)
    eigensteps = []

    for n in tqdm(range(n_samples), desc="Eigenstep Sample", unit="step"):
        # To do: Figure out how to get bounding box of poly Ax<=b and rejection sample.
        in_poly = False
        while not in_poly:
            sample = []
            for i in range(len(l)):
                sample.append(np.random.uniform(l[i], u[i]))
            sample = np.array(sample)
            sample.reshape(len(sample), 1)
            in_poly = P.contains(sample)

        esteps = sys.make_eigenstep_mat(sample.flatten())
        esteps = esteps[::-1, :]
        try:
            eigensteps.append(Eigensteps(esteps))
        except:
            n = n - 1

    if n_samples == 1:
        return Eigensteps(esteps)
    else:
        return eigensteps


def get_permutation_matrix(permutation):
    '''
    Takes in a permutation like [0, 2, 1, 3] and return corresponding matrix
    '''
    length = len(permutation)
    pmat = np.mat(np.zeros((length, length)))
    for i in range(length):
        pmat[i, permutation[i]] = 1
    return pmat


def random_FUNTF(d, N, eigensteps=DEFAULT):
    '''
    Algorithm is based on the one presented in "Constructing Finite Frames of
    a Given Spectrum and Set of Lengths" by Cahill, Fickus, Mixon, Poteet, and
    Strawn.

    Parameters
    ----------
    d : int
        The dimension of the frame vectors.
    N : int
        The number of frame vectors.
    eigensteps : Eigensteps, optional
        The eigensteps the generated frame will have. If no eigensteps are given,
        we construct a random Eigensteps object via random_FUNTF_eigensteps_bb.

    Returns
    -------
    Frame
        The generated frame. If eigensteps are given, this frame will have those eigensteps.
    '''

    frame_vectors = []

    # A. If no specific eigensteps are provied get random ones via Hit n' Run
    if eigensteps is DEFAULT:
        eigensteps = random_FUNTF_eigensteps_bb(d, N)

    # B. Let U_1 be any unitary matrix
    U_n = np.mat(np.identity(d))

    # f_1 = first row of U_1
    frame_vectors.append(U_n[:, 0])

    # For each n = 1, ..., N-1:
    for n in range(N - 1):

        # B.1 Create V_n, the d by d block-diagonal unitary matrix whose blocks
        # correspond to the distinct values in the nth row of eigensteps
        # sequence, where the size of each block is the multiplicity of each
        # value
        # There are many such matrices in general, we will take the identity
        V_n = np.mat(np.identity(d))

        # B.2 Build I_n and J_n sets
        I_n = set()
        J_n = set()
        l_n = list(eigensteps.get_row(n).A.reshape(d, ))
        l_np1 = list(eigensteps.get_row(n + 1).A.reshape(d, ))

        for m in range(d):
            # TODO is this going to have numerical problems?
            if ((l_n.index(l_n[m]) == m) and
                    (l_n.count(l_n[m]) > l_np1.count(l_n[m]))):
                I_n.add(m)
            if ((l_np1.index(l_np1[m]) == m) and
                    (l_np1.count(l_np1[m]) > l_n.count(l_np1[m]))):
                J_n.add(m)
        assert len(I_n) == len(J_n)
        R_n = len(I_n)

        # now find the unique permutation matrix such that is increasing on both
        # I_n and I_n^C, and takes I_n to the first R_n elements
        I_nC = set(range(d)) - I_n
        J_nC = set(range(d)) - J_n
        I_n_permutation = sorted(I_n) + sorted(I_nC)
        J_n_permutation = sorted(J_n) + sorted(J_nC)
        pi_I_n = get_permutation_matrix(I_n_permutation)
        pi_J_n = get_permutation_matrix(J_n_permutation)

        # B.3 Calculate v_n and w_n vectors
        v_n_squared = np.mat(np.zeros((R_n, 1)))
        for m in I_n:
            ix = I_n_permutation.index(m)  # permuted index
            num = np.prod([l_n[m] - l_np1[m_pr] for m_pr in J_n])
            denom = np.prod([l_n[m] - l_n[m_pr] for m_pr in I_n if m_pr != m])
            v_n_squared[ix] = -num / denom
        v_n = np.sqrt(v_n_squared)

        w_n_squared = np.mat(np.zeros((R_n, 1)))
        for m in J_n:
            ix = J_n_permutation.index(m)  # permuted index
            num = np.prod([l_np1[m] - l_n[m_pr] for m_pr in I_n])
            denom = np.prod([l_np1[m] - l_np1[m_pr] for m_pr in J_n
                             if m_pr != m])
            w_n_squared[ix] = num / denom
        w_n = np.sqrt(w_n_squared)

        # B.4 Calculate f_np1
        v_n_padded = np.mat(np.zeros((d, 1)))
        v_n_padded[:R_n, 0] = v_n
        f_np1 = U_n * V_n * pi_I_n.T * v_n_padded
        frame_vectors.append(f_np1)

        # B.5 Update U_n
        W_n = v_n * w_n.T
        for m in I_n:
            ix = I_n_permutation.index(m)  # permuted index
            for m_pr in J_n:
                jx = J_n_permutation.index(m_pr)  # permuted index
                W_n[ix, jx] /= l_np1[m_pr] - l_n[m]
        W_n_padded = np.mat(np.identity(d))
        W_n_padded[:R_n, :R_n] = W_n
        U_np1 = U_n * V_n * pi_I_n.T * W_n_padded * pi_J_n
        U_n = U_np1

    return Frame(np.concatenate(frame_vectors, axis=1))


def torus_action(frame, theta=DEFAULT):
    '''
    Algorithm implements the Hamiltonian torus action defined in Shonkwiler and Faldet's paper.

    Parameters
    ----------
    frame : Frame
        The FUNTF to be acted on.
    theta : ndarray of dimension (N - d - 1)(d -1)
        The point on the torus which will act on the frame.

    Returns
    -------
    Frame
        The resulting frame after excuting the torus action.
    '''

    N = frame.num_vectors
    d = frame.dimension
    F = frame._frame

    # If point on torus isn't provided choose a random one, if a point
    # is provided assert that the dimension is correct.
    if theta is DEFAULT:
        theta = np.random.uniform(0, 2 * np.pi, (N - d - 1) * (d - 1))

    assert len(theta) == (N - d - 1) * (d - 1)

    # collect coordinates of free variables
    free_var_pos = []

    for i in range(1, N - 2):
        for j in range(max(0, d + i - N + 1), min(i, d)):
            if j != (d - 1):
                free_var_pos.append((i, j))

    n_free_variables = len(free_var_pos)
    assert n_free_variables == (d - 1) * (N - d - 1)

    # enumerate each free variable
    pos_to_idx = {}
    idx_to_pos = {}
    for idx, pos in enumerate(free_var_pos):
        pos_to_idx[pos] = idx
        idx_to_pos[idx] = pos

    # e_vals <- The eigenstep table associated to F
    # e_vects <- A list of lists, list element k is a list of the eigenvectors of S_k
    e_vals, e_vects = frame.get_eigensteps()

    for i in range(n_free_variables):
        k, j = idx_to_pos[i]
        angle = theta[i]
        mu = e_vals[k, j]
        v = np.mat(e_vects[k][:, j])

        A = expm(((1j * angle) * v) * v.H)

        for c in range(k + 1):
            F[:, c] = A @ F[:, c]

    return Frame(F)


def has_same_esteps(frame1, frame2):
    estep1, evec1 = frame1.get_eigensteps()
    estep2, evec2 = frame2.get_eigensteps()
    return np.all(np.isclose(estep1, estep2, EPSILON, EPSILON))


def coherence_distribution(d, N, num_samples=10000, on_fiber=False):
    '''
    Generate a specified number of random frames using random_FUNTF, compute the coherence of each frame
    and log results to a txt file in addition to plotting a histogram.

    Parameters
    ----------
    d : int
        The dimension of the frame vectors.
    N : int
        The number of frame vectors.
    num_samples: int
        The number of samples to generate.
    on_fiber: bool (Optional)
        Set to true if you want all samples to lie on the same fiber i.e. all sameples will have
        the same eigensteps.
    '''

    coherences = []

    if on_fiber:
        F = random_FUNTF(d, N)
        for i in tqdm(range(num_samples), desc="Random Frame Sample", unit="step"):
            F = torus_action(F)
            assert F.is_tight()
            assert F.is_unit_norm()
            coherences.append(F.coherence())
    else:
        esteps = random_FUNTF_eigensteps_bb(d, N, n_samples=num_samples)
        for i in tqdm(range(len(esteps)), desc="Random Frame Sample", unit="step"):
            F = torus_action(random_FUNTF(d, N, esteps[i]))
            assert F.is_tight()
            assert F.is_unit_norm()
            print(F.coherence())
            coherences.append(F.coherence())

    results_dir = 'Results'
    os.makedirs(results_dir, exist_ok=True)

    # Plot the histogram of coherence values
    plt.figure()
    plt.hist(coherences, bins=20, edgecolor='black', density=True)
    plt.title(f'Coherence Distribution: d = {d}, N = {N}, # of samples {num_samples}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Save the histogram in the 'Results' folder
    histogram_path = os.path.join(results_dir, f'histogram_N_{N}_d_{d}.png')
    plt.savefig(histogram_path)
    plt.close()

    # Create a text file in the 'Results' folder and write each coherence value on a separate line
    coherences_file = os.path.join(results_dir, f'coherences_N_{N}_d_{d}.txt')
    with open(coherences_file, 'w') as f:
        f.write(f"Max coherence: {np.max(np.array(coherences))}")
        f.write(f"Mean coherence: {np.mean(np.array(coherences))}")
        for value in coherences:
            f.write(f"{value}\n")

if __name__ == '__main__':

    coherence_distribution(d=2, N=4, num_samples = 20)
    # coherence_distribution(d=2, N=5, num_samples = 1000000)
    # coherence_distribution(d=2, N=6, num_samples = 1000000)
    # coherence_distribution(d=2, N=7, num_samples = 1000000)
    # coherence_distribution(d=2, N=8, num_samples = 1000000)
    # coherence_distribution(d=3, N=5, num_samples = 1000000)
    # coherence_distribution(d=3, N=6, num_samples = 1000000)
    # coherence_distribution(d=3, N=7, num_samples = 1000000)
    # coherence_distribution(d=4, N=7, num_samples = 1000000)
