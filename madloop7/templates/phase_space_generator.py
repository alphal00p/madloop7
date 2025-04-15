import math
import logging
import numpy as np
import copy

logger = logging.getLogger('madloop7_sa.phase_space_generator')


class InvalidOperation(Exception):
    pass


def almost_equal(x, y, rel_tol=0, abs_tol=0):
    """Check if two objects are equal within certain relative and absolute tolerances.
    The operations abs(x + y) and abs(x - y) need to be well-defined
    for this function to work.

    :param x: first object to be compared
    :param y: second object to be compared
    :param rel_tol: relative tolerance for the comparison
    :param abs_tol: absolute tolerance for the comparison
    :return: True if the elements are equal within tolerance, False otherwise
    :rtype: bool
    """

    diffxy = abs(x - y)
    if diffxy <= abs_tol:
        return True
    sumxy = abs(x + y)
    # Rough check that the ratio is smaller than 1 to avoid division by zero
    if sumxy < diffxy:
        return False
    return diffxy / sumxy <= rel_tol

# =========================================================================================
# Vector
# =========================================================================================


class Vector(np.ndarray):

    def __new__(cls, *args, **opts):

        if args and isinstance(args[0], Vector):
            foo = args[0].get_copy()
        else:
            foo = np.asanyarray(*args, **opts).view(cls)
        return foo

    def __array_finalize__(self, obj):

        try:
            self.eps = np.finfo(self.dtype).eps ** 0.5
        except:
            self.eps = 0
        return

    def huge(self):

        if np.issubdtype(self.dtype, np.inexact):
            return np.finfo(self.dtype).max
        elif np.issubdtype(self.dtype, np.integer):
            return np.iinfo(self.dtype).max
        else:
            raise ValueError

    def almost_zero(self, x):

        return x < self.eps

    def almost_equal(self, x, y=None):
        """Check if two numbers are equal within the square root
        of the typical numerical accuracy of the underlying array data type.
        """

        if y is None:
            y = self
        return almost_equal(x, y, rel_tol=self.eps)

    def square(self):

        return self.dot(self)

    def __abs__(self):

        foo = self.view(np.ndarray)
        return np.dot(foo, foo) ** 0.5

    def __eq__(self, other):

        return almost_equal(self, other, rel_tol=self.eps+other.eps)

    def __hash__(self):

        return tuple(x for x in self).__hash__()

    def get_copy(self):

        # The vector instantiated by get_copy() should be modified
        # without changing the previous instance, irrespectively of the
        # (presumably few) layers that compose entries of the vector
        # return copy.deepcopy(self)
        return copy.copy(self)

    def normalize(self):

        self.__idiv__(abs(self))
        return self

    def project_onto(self, v):

        return (self.dot(v) / v.square()) * v

    def component_orthogonal_to(self, v):

        return self - self.project_onto(v)

    @classmethod
    def cos(cls, v, w):
        """Cosine of the angle between two vectors."""

        assert v.square() > 0
        assert w.square() > 0
        return v.dot(w)/(abs(v)*abs(w))

    # Specific to 3D vectors
    def cross(self, v):

        assert len(self) == 3
        assert len(v) == 3
        return Vector([
            self[1] * v[2] - self[2] * v[1],
            self[2] * v[0] - self[0] * v[2],
            self[0] * v[1] - self[1] * v[0]
        ])

# =========================================================================================
# LorentzVector
# =========================================================================================


class LorentzVector(Vector):

    def __new__(cls, *args, **opts):

        if len(args) == 0:
            return super(LorentzVector, cls).__new__(cls, [0., 0., 0., 0.], **opts)
        return super(LorentzVector, cls).__new__(cls, *args, **opts)

    def space(self):
        """Return the spatial part of this LorentzVector."""

        return self[1:].view(type=Vector)

    def dot(self, v, out=None):
        """Compute the Lorentz scalar product."""
        # The implementation below allows for a check but it should be done upstream and
        # significantly slows down the code here.
        # pos = self[0]*v[0]
        # neg = self.space().dot(v.space())
        # if pos+neg != 0 and abs(2*(pos-neg)/(pos+neg)) < 100.*self.eps(): return 0
        # return pos - neg
        return self[0]*v[0] - self[1]*v[1] - self[2]*v[2] - self[3]*v[3]

    def square_almost_zero(self):
        """Check if the square of this LorentzVector is zero within numerical accuracy."""

        return self.almost_zero(self.square() / np.dot(self, self))

    def rho2(self):
        """Compute the radius squared."""

        return self.space().square()

    def rho(self):
        """Compute the radius."""

        return abs(self.space())

    def space_direction(self):
        """Compute the corresponding unit vector in ordinary space."""

        return self.space()/self.rho()

    def set_square(self, square, negative=False):
        """Change the time component of this LorentzVector
        in such a way that self.square() = square.
        If negative is True, set the time component to be negative,
        else assume it is positive.
        """

        # Note: square = self[0]**2 - self.rho2(),
        # so if (self.rho2() + square) is negative, self[0] is imaginary.
        # Letting math.sqrt fail if data is not complex on purpose in this case.
        self[0] = (self.rho2() + square) ** 0.5
        if negative:
            self[0] *= -1
        return self

    def rotoboost(self, p, q):
        """Apply the Lorentz transformation that sends p in q to this vector."""

        # NOTE: when applying the same Lorentz transformation to many vectors,
        #       this function goes many times through the same checks.

        # Compute squares
        p2 = p.square()
        q2 = q.square()
        # Check if both Lorentz squares are small compared to the euclidean squares,
        # in which case the alternative formula should be used
        if p.square_almost_zero() and q.square_almost_zero():
            # Use alternative formula
            if self.almost_equal(p):
                for i in range(len(self)):
                    self[i] = q[i]
            else:
                logger.critical("Error in vectors.rotoboost: missing formula")
                logger.critical("Boosting %s (%.9e)" %
                                (str(self), self.square()))
                logger.critical("p = %s (%.9e)" % (str(p), p2))
                logger.critical("q = %s (%.9e)" % (str(q), q2))
                logger.critical(
                    "Eq. (4.14) of arXiv:0706.0017v2, p. 26 not implemented")
                raise NotImplementedError
            return self
        else:
            # Check that the two invariants are close,
            # else the transformation is invalid
            if not almost_equal(p2, q2, rel_tol=p.eps+q.eps):
                logger.critical(
                    "Error in vectors.rotoboost: nonzero, unequal squares")
                logger.critical("p = %s (%.9e)" % (str(p), p2))
                logger.critical("q = %s (%.9e)" % (str(q), q2))
                print("Error in vectors.rotoboost: nonzero, unequal squares")
                print("p = %s (%.9e)" % (str(p), p2))
                print("q = %s (%.9e)" % (str(q), q2))
                raise InvalidOperation
            # Compute scalar products
            pq = p + q
            pq2 = pq.square()
            p_s = self.dot(p)
            pq_s = self.dot(pq)
            # Assemble vector
            self.__iadd__(2 * ((p_s/q2) * q - (pq_s/pq2) * pq))
            return self

    def pt(self, axis=3):
        """Compute transverse momentum."""

        return math.sqrt(
            sum(self[i]**2 for i in range(1, len(self)) if i != axis))

    def pseudoRap(self):
        """Compute pseudorapidity."""

        pt = self.pt()
        if pt < self.eps and abs(self[3]) < self.eps:
            return self.huge()*(self[3]/abs(self[3]))
        th = math.atan2(pt, self[3])
        return -math.log(math.tan(th/2.))

    def rap(self):
        """Compute rapidity in the lab frame. (needs checking)"""

        if self.pt() < self.eps and abs(self[3]) < self.eps:
            return self.huge()*(self[3]/abs(self[3]))

        return .5*math.log((self[0]+self[3])/(self[0]-self[3]))

    def getdelphi(self, p2):
        """Compute the phi-angle separation with p2."""

        pt1 = self.pt()
        pt2 = p2.pt()
        if pt1 == 0. or pt2 == 0.:
            return self.huge()
        tmp = self[1]*p2[1] + self[2]*p2[2]
        tmp /= (pt1*pt2)
        if abs(tmp) > (1.0+self.eps):
            logger.critical("Cosine larger than 1. in phase-space cuts.")
            raise ValueError
        if abs(tmp) > 1.0:
            return math.acos(tmp/abs(tmp))
        return math.acos(tmp)

    def deltaR(self, p2):
        """Compute the deltaR separation with momentum p2."""

        delta_eta = self.pseudoRap() - p2.pseudoRap()
        delta_phi = self.getdelphi(p2)
        return math.sqrt(delta_eta**2 + delta_phi**2)

    def boostVector(self):

        if self == LorentzVector():
            return Vector([0.] * 3)
        if self[0] <= 0. or self.square() < 0.:
            logger.critical("Attempting to compute a boost vector from")
            logger.critical("%s (%.9e)" % (str(self), self.square()))
            raise InvalidOperation
        return self.space()/self[0]

    def cosTheta(self):

        ptot = self.rho()
        assert (ptot > 0.)
        return self[3] / ptot

    @classmethod
    def cos(cls, v, w):
        """Cosine of the angle between the space part of two vectors."""

        return Vector.cos(v.space(), w.space())

    def phi(self):

        return math.atan2(self[2], self[1])

    def boost(self, boost_vector, gamma=-1.):
        """Transport self into the rest frame of the boost_vector in argument.
        This means that the following command, for any vector p=(E, px, py, pz)
            p.boost(-p.boostVector())
        transforms p to (M,0,0,0).
        """

        b2 = boost_vector.square()
        if gamma < 0.:
            gamma = 1.0 / math.sqrt(1.0 - b2)

        bp = self.space().dot(boost_vector)
        gamma2 = (gamma-1.0) / b2 if b2 > 0 else 0.
        factor = gamma2*bp + gamma*self[0]
        self_space = self.space()
        self_space += factor*boost_vector
        self[0] = gamma*(self[0] + bp)
        return self

    @classmethod
    def boost_vector_from_to(cls, p, q):
        """Determine the boost vector for a pure boost that sends p into q.
        For details, see appendix A.2.2 of Simone Lionetti's PhD thesis.

        :param LorentzVector p: Starting Lorentz vector to define the boost.
        :param LorentzVector q: Target Lorentz vector to define the boost.
        :return: Velocity vector for a boost that sends p into q.
        :rtype: Vector
        """

        eps = p.eps+q.eps
        p_abs = abs(p)
        q_abs = abs(q)
        assert almost_equal(p.square(), q.square(), rel_tol=eps) or \
            (p.square_almost_zero() and q.square_almost_zero())
        p_vec = p.space()
        q_vec = q.space()
        if almost_equal(p_vec, q_vec, rel_tol=eps):
            return Vector([0 for _ in p_vec])
        n_vec = (q_vec - p_vec).normalize()
        na = LorentzVector([1, ] + list(+n_vec))
        nb = LorentzVector([1, ] + list(-n_vec))
        assert na.square_almost_zero()
        assert nb.square_almost_zero()
        assert almost_equal(na.dot(nb), 2, rel_tol=eps)
        p_plus = p.dot(nb)
        p_minus = p.dot(na)
        q_plus = q.dot(nb)
        q_minus = q.dot(na)
        if p_minus/p_abs < eps and q_minus/q_abs < eps:
            if p_plus/p_abs < eps and q_plus/q_abs < eps:
                exppy = 1
            else:
                exppy = q_plus / p_plus
        else:
            if p_plus/p_abs < eps and q_plus/q_abs < eps:
                exppy = p_minus / q_minus
            else:
                exppy = ((q_plus*p_minus) / (q_minus*p_plus)) ** 0.5
        expmy = 1. / exppy
        return abs((exppy - expmy) / (exppy + expmy)) * n_vec

    def boost_from_to(self, p, q):
        """Apply a pure boost that sends p into q to this LorentzVector.
        For details, see appendix A.2.2 of Simone Lionetti's PhD thesis.

        :param LorentzVector p: Starting Lorentz vector to define the boost.
        :param LorentzVector q: Target Lorentz vector to define the boost.
        """

        eps = p.eps+q.eps
        p_abs = abs(p)
        q_abs = abs(q)
        assert almost_equal(p.square(), q.square(), rel_tol=eps) or \
            (p.square_almost_zero() and q.square_almost_zero())
        p_vec = p.space()
        q_vec = q.space()
        if almost_equal(p_vec, q_vec, rel_tol=eps):
            return Vector([0 for _ in p_vec])
        n_vec = (q_vec - p_vec).normalize()
        na = LorentzVector([1, ] + list(+n_vec))
        nb = LorentzVector([1, ] + list(-n_vec))
        assert na.square_almost_zero()
        assert nb.square_almost_zero()
        assert almost_equal(na.dot(nb), 2, rel_tol=eps)
        p_plus = p.dot(nb)
        p_minus = p.dot(na)
        q_plus = q.dot(nb)
        q_minus = q.dot(na)
        if p_minus/p_abs < eps and q_minus/q_abs < eps:
            if p_plus/p_abs < eps and q_plus/q_abs < eps:
                ratioa = 1
                ratiob = 1
            else:
                ratiob = q_plus / p_plus
                ratioa = 1. / ratiob
        else:
            if p_plus/p_abs < eps and q_plus/q_abs < eps:
                ratioa = q_minus / p_minus
                ratiob = 1. / ratioa
            else:
                ratioa = q_minus / p_minus
                ratiob = q_plus / p_plus
        plus = self.dot(nb)
        minus = self.dot(na)
        self.__iadd__(((ratiob - 1) * 0.5 * plus) * na)
        self.__iadd__(((ratioa - 1) * 0.5 * minus) * nb)
        return self

# =========================================================================================
# LorentzVectorDict
# =========================================================================================


class LorentzVectorDict(dict):
    """A simple class wrapping dictionaries that store Lorentz vectors."""

    def to_list(self, ordered_keys=None):
        """Return list copy of self. Notice that the actual values of the keys
        are lost in this process. The user can specify in which order (and potentially which ones)
        the keys must be placed in the list returned."""

        if ordered_keys is None:
            return LorentzVectorList(self[k] for k in sorted(self.keys()))
        else:
            return LorentzVectorList(self[k] for k in ordered_keys)

    def to_dict(self):
        """Return a copy of this LorentzVectorDict """

        return LorentzVectorDict(self)

    def to_tuple(self):
        """Return a copy of this LorentzVectorDict as an immutable tuple.
        Notice that the actual values of the keys are lost in this process.
        """

        return tuple(tuple(self[k]) for k in sorted(self.keys()))

    def __str__(self, n_initial=2):
        """Nice printout of the momenta."""

        # Use padding for minus signs
        def special_float_format(fl):
            return '%s%.16e' % ('' if fl < 0.0 else ' ', fl)

        cols_widths = [4, 25, 25, 25, 25, 25]
        template = ' '.join(
            '%%-%ds' % col_width for col_width in cols_widths
        )
        line = '-' * (sum(cols_widths) + len(cols_widths) - 1)

        out_lines = [template % ('#', ' E', ' p_x', ' p_y', ' p_z', ' M',)]
        out_lines.append(line)
        running_sum = LorentzVector()
        for i in sorted(self.keys()):
            mom = LorentzVector(self[i])
            if i <= n_initial:
                running_sum += mom
            else:
                running_sum -= mom
            out_lines.append(template % tuple(
                ['%d' % i] + [
                    special_float_format(el) for el in (list(mom) + [math.sqrt(abs(mom.square()))])
                ]
            ))
        out_lines.append(line)
        out_lines.append(template % tuple(
            ['Sum'] + [special_float_format(el) for el in running_sum] + ['']
        ))

        return '\n'.join(out_lines)

    def boost_to_com(self, initial_leg_numbers):
        """ Boost this kinematic configuration back to its c.o.m. frame given the
        initial leg numbers. This is not meant to be generic and here we *want* to crash
        if we encounter a configuration that is not supposed to ever need boosting in the
        MadNkLO construction.
        """

        if len(initial_leg_numbers) == 2:
            if __debug__:
                sqrts = math.sqrt(
                    (self[initial_leg_numbers[0]]+self[initial_leg_numbers[1]]).square())
                # Assert initial states along the z axis
                assert (abs(self[initial_leg_numbers[0]][1]/sqrts) < 1.0e-9)
                assert (abs(self[initial_leg_numbers[1]][1]/sqrts) < 1.0e-9)
                assert (abs(self[initial_leg_numbers[0]][2]/sqrts) < 1.0e-9)
                assert (abs(self[initial_leg_numbers[1]][2]/sqrts) < 1.0e-9)
            # Now send the self back into its c.o.m frame, if necessary
            initial_momenta_summed = self[initial_leg_numbers[0]
                                          ]+self[initial_leg_numbers[1]]
            sqrts = math.sqrt((initial_momenta_summed).square())
            if abs(initial_momenta_summed[3]/sqrts) > 1.0e-9:
                boost_vector = (initial_momenta_summed).boostVector()
                for vec in self.values():
                    vec.boost(-boost_vector)
            if __debug__:
                assert (abs((self[initial_leg_numbers[0]] +
                        self[initial_leg_numbers[1]])[3]/sqrts) <= 1.0e-9)
        elif len(initial_leg_numbers) == 1:
            if __debug__:
                sqrts = math.sqrt(self[initial_leg_numbers[0]].square())
                assert (abs(self[initial_leg_numbers[0]][1]/sqrts) < 1.0e-9)
                assert (abs(self[initial_leg_numbers[0]][2]/sqrts) < 1.0e-9)
                assert (abs(self[initial_leg_numbers[0]][3]/sqrts) < 1.0e-9)
        else:
            raise InvalidOperation(
                'MadNkLO only supports processes with one or two initial states.')

    def get_copy(self):
        """Return a copy that can be freely modified
        without changing the current instance.
        """

        return type(self)((i, LorentzVector(k)) for i, k in self.items())

# =========================================================================================
# LorentzVectorList
# =========================================================================================


class LorentzVectorList(list):
    """A simple class wrapping lists that store Lorentz vectors."""

    def __str__(self, n_initial=2):
        """Nice printout of the momenta."""

        return LorentzVectorDict(
            (i + 1, v) for i, v in enumerate(self)
        ).__str__(n_initial=n_initial)

    def to_list(self):
        """Return list copy of self."""

        return LorentzVectorList(self)

    def to_tuple(self):
        """Return a copy of this LorentzVectorList as an immutable tuple."""

        return tuple(tuple(v) for v in self)

    def to_dict(self):
        """Return a copy of this LorentzVectorList as a LorentzVectorDict."""

        return LorentzVectorDict((i+1, v) for i, v in enumerate(self))

    def boost_from_com_to_lab_frame(self, x1, x2, ebeam1, ebeam2):
        """ Boost this kinematic configuration from the center of mass frame to the lab frame
        given specified Bjorken x's x1 and x2.
        This function needs to be cleaned up and built in a smarter way as the boost vector can be written
        down explicitly as a function of x1, x2 and the beam energies.
        """

        if x1 is None:
            x1 = 1.
        if x2 is None:
            x2 = 1.

        target_initial_momenta = []
        for i, (x, ebeam) in enumerate(zip([x1, x2], [ebeam1, ebeam2])):
            target_initial_momenta.append(LorentzVector(
                [x*ebeam, 0., 0., math.copysign(x*ebeam, self[i][3])]))
        target_summed = sum(target_initial_momenta)
        source_summed = LorentzVector(
            [2.*math.sqrt(x1*x2*ebeam1*ebeam2), 0., 0., 0.])

        # We want to send the source to the target
        for vec in self:
            vec.boost_from_to(source_summed, target_summed)
            # boost_vec = LorentzVector.boost_vector_from_to(source_summed, target_summed)
            # import madgraph.various.misc as misc
            # misc.sprint(boost_vec)
            # vec.boost(boost_vec)

    def boost_to_com(self, initial_leg_numbers):
        """ Boost this kinematic configuration back to its c.o.m. frame given the
        initial leg numbers. This is not meant to be generic and here we *want* to crash
        if we encounter a configuration that is not supposed to ever need boosting in the
        MadNkLO construction.
        """
        # Given that this is a list, we must subtract one to the indices given
        initial_leg_numbers = tuple(n-1 for n in initial_leg_numbers)
        if len(initial_leg_numbers) == 2:
            if __debug__:
                sqrts = math.sqrt(
                    (self[initial_leg_numbers[0]]+self[initial_leg_numbers[1]]).square())
                # Assert initial states along the z axis
                assert (abs(self[initial_leg_numbers[0]][1]/sqrts) < 1.0e-9)
                assert (abs(self[initial_leg_numbers[1]][1]/sqrts) < 1.0e-9)
                assert (abs(self[initial_leg_numbers[0]][2]/sqrts) < 1.0e-9)
                assert (abs(self[initial_leg_numbers[1]][2]/sqrts) < 1.0e-9)
            # Now send the self back into its c.o.m frame, if necessary
            initial_momenta_summed = self[initial_leg_numbers[0]
                                          ]+self[initial_leg_numbers[1]]
            sqrts = math.sqrt((initial_momenta_summed).square())
            if abs(initial_momenta_summed[3]/sqrts) > 1.0e-9:
                boost_vector = (initial_momenta_summed).boostVector()
                for vec in self:
                    vec.boost(-boost_vector)
            if __debug__:
                assert (abs((self[initial_leg_numbers[0]] +
                        self[initial_leg_numbers[1]])[3]/sqrts) <= 1.0e-9)
        elif len(initial_leg_numbers) == 1:
            if __debug__:
                sqrts = math.sqrt(self[initial_leg_numbers[0]].square())
                assert (abs(self[initial_leg_numbers[0]][1]/sqrts) < 1.0e-9)
                assert (abs(self[initial_leg_numbers[0]][2]/sqrts) < 1.0e-9)
                assert (abs(self[initial_leg_numbers[0]][3]/sqrts) < 1.0e-9)
        else:
            raise InvalidOperation(
                'MadNkLO only supports processes with one or two initial states.')

    def get_copy(self):
        """Return a copy that can be freely modified
        without changing the current instance.
        """

        return type(self)([LorentzVector(p) for p in self])


class Dimension(object):
    """ A dimension object specifying a specific integration dimension."""

    def __init__(self, name, folded=False):
        self.name = name
        self.folded = folded

    def length(self):
        raise NotImplemented

    def random_sample(self):
        raise NotImplemented


class DiscreteDimension(Dimension):
    """ A dimension object specifying a specific discrete integration dimension."""

    def __init__(self, name, values, **opts):
        try:
            self.normalized = opts.pop('normalized')
        except:
            self.normalized = False
        super(DiscreteDimension, self).__init__(name, **opts)
        assert (isinstance(values, list))
        self.values = values

    def length(self):
        if normalized:
            return 1.0/float(len(values))
        else:
            return 1.0

    def random_sample(self):
        return np.int64(random.choice(values))


class ContinuousDimension(Dimension):
    """ A dimension object specifying a specific discrete integration dimension."""

    def __init__(self, name, lower_bound=0.0, upper_bound=1.0, **opts):
        super(ContinuousDimension, self).__init__(name, **opts)
        assert (upper_bound > lower_bound)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def length(self):
        return (self.upper_bound-self.lower_bound)

    def random_sample(self):
        return np.float64(self.lower_bound+random.random()*(self.upper_bound-self.lower_bound))


class DimensionList(list):
    """A DimensionList."""

    def __init__(self, *args, **opts):
        super(DimensionList, self).__init__(*args, **opts)

    def volume(self):
        """ Returns the volue of the complete list of dimensions."""
        vol = 1.0
        for d in self:
            vol *= d.length()
        return vol

    def append(self, arg, **opts):
        """ Type-checking. """
        assert (isinstance(arg, Dimension))
        super(DimensionList, self).append(arg, **opts)

    def get_discrete_dimensions(self):
        """ Access all discrete dimensions. """
        return DimensionList(d for d in self if isinstance(d, DiscreteDimension))

    def get_continuous_dimensions(self):
        """ Access all discrete dimensions. """
        return DimensionList(d for d in self if isinstance(d, ContinuousDimension))

    def random_sample(self):
        return np.array([d.random_sample() for d in self])


# =========================================================================================
# Phase space generation
# =========================================================================================

class VirtualPhaseSpaceGenerator(object):

    def __init__(self, initial_masses, final_masses,
                 beam_Es,
                 beam_types=(1, 1),
                 is_beam_factorization_active=(False, False),
                 correlated_beam_convolution=False
                 ):

        self.initial_masses = initial_masses
        self.masses = final_masses
        self.n_initial = len(initial_masses)
        self.n_final = len(final_masses)
        self.beam_Es = beam_Es
        self.collider_energy = sum(beam_Es)
        self.beam_types = beam_types
        self.is_beam_factorization_active = is_beam_factorization_active
        self.correlated_beam_convolution = correlated_beam_convolution
        # Sanity check
        if self.correlated_beam_convolution and self.is_beam_factorization_active != (True, True):
            raise PhaseSpaceGeneratorError(
                'The beam convolution cannot be set to be correlated if it is one-sided only')
        self.dimensions = self.get_dimensions()
        self.dim_ordered_names = [d.name for d in self.dimensions]

        self.dim_name_to_position = dict(
            (d.name, i) for i, d in enumerate(self.dimensions))
        self.position_to_dim_name = dict((v, k) for (
            k, v) in self.dim_name_to_position.items())

    def generateKinematics(self, E_cm, random_variables):
        """Generate a phase-space point with fixed center of mass energy."""

        raise NotImplementedError

    def get_PS_point(self, random_variables):
        """Generate a complete PS point, including Bjorken x's,
        dictating a specific choice of incoming particle's momenta."""

        raise NotImplementedError

    def boost_to_lab_frame(self, PS_point, xb_1, xb_2):
        """Boost a phase-space point from the COM-frame to the lab frame, given Bjorken x's."""

        if self.n_initial == 2 and (xb_1 != 1. or xb_2 != 1.):
            ref_lab = (PS_point[0]*xb_1 + PS_point[1]*xb_2)
            if ref_lab.rho2() != 0.:
                lab_boost = ref_lab.boostVector()
                for p in PS_point:
                    p.boost(-lab_boost)

    def boost_to_COM_frame(self, PS_point):
        """Boost a phase-space point from the lab frame to the COM frame"""

        if self.n_initial == 2:
            ref_com = (PS_point[0] + PS_point[1])
            if ref_com.rho2() != 0.:
                com_boost = ref_com.boostVector()
                for p in PS_point:
                    p.boost(-com_boost)

    def nDimPhaseSpace(self):
        """Return the number of random numbers required to produce
        a given multiplicity final state."""

        if self.n_final == 1:
            return 0
        return 3*self.n_final - 4

    def get_dimensions(self):
        """Generate a list of dimensions for this integrand."""

        dims = DimensionList()

        # Add the PDF dimensions if necessary
        if self.beam_types[0] == self.beam_types[1] == 1:
            dims.append(ContinuousDimension(
                'ycms', lower_bound=0.0, upper_bound=1.0))
            # The 2>1 topology requires a special treatment
            if not (self.n_initial == 2 and self.n_final == 1):
                dims.append(ContinuousDimension(
                    'tau', lower_bound=0.0, upper_bound=1.0))

        # Add xi beam factorization convolution factors if necessary
        if self.correlated_beam_convolution:
            # A single convolution factor xi that applies to both beams is needed in this case
            dims.append(ContinuousDimension(
                'xi', lower_bound=0.0, upper_bound=1.0))
        else:
            if self.is_beam_factorization_active[0]:
                dims.append(ContinuousDimension(
                    'xi1', lower_bound=0.0, upper_bound=1.0))
            if self.is_beam_factorization_active[1]:
                dims.append(ContinuousDimension(
                    'xi2', lower_bound=0.0, upper_bound=1.0))

        # Add the phase-space dimensions
        dims.extend([ContinuousDimension('x_%d' % i, lower_bound=0.0, upper_bound=1.0)
                     for i in range(1, self.nDimPhaseSpace()+1)])

        return dims


class FlatInvertiblePhasespace(VirtualPhaseSpaceGenerator):
    """Implementation following S. Platzer, arxiv:1308.2922"""

    # This parameter defines a thin layer around the boundary of the unit hypercube
    # of the random variables generating the phase-space,
    # so as to avoid extrema which are an issue in most PS generators.
    epsilon_border = 1e-10

    # The lowest value that the center of mass energy can take.
    # We take here 1 GeV, as anyway below this non-perturbative effects dominate
    # and factorization does not make sense anymore
    absolute_Ecm_min = 1.

    # For reference here we put the flat weights that Simon uses in his
    # Herwig implementation. I will remove them once I will have understood
    # why they don't match the physical PS volume.
    # So these are not used for now, and get_flatWeights() is used instead.
    flatWeights = {2:  0.039788735772973833942,
                   3:  0.00012598255637968550463,
                   4:  1.3296564302788840628e-7,
                   5:  7.0167897579949011130e-11,
                   6:  2.2217170114046130768e-14
                   }

    def __init__(self, *args, **opts):
        super(FlatInvertiblePhasespace, self).__init__(*args, **opts)
        if self.n_initial == 1:
            raise InvalidCmd(
                "This basic generator does not support decay topologies.")

    def get_dimensions(self):
        """ Make sure the collider setup is supported."""

        # Check if the beam configuration is supported
        if (not abs(self.beam_types[0]) == abs(self.beam_types[1]) == 1) and \
           (not self.beam_types[0] == self.beam_types[1] == 0):
            raise InvalidCmd(
                "This basic generator does not support the collider configuration: (lpp1=%d, lpp2=%d)" %
                (self.run_card['lpp1'], self.run_card['lpp2']))

        if self.beam_Es[0] != self.beam_Es[1]:
            raise InvalidCmd(
                "This basic generator only supports colliders with incoming beams equally energetic.")

        return super(FlatInvertiblePhasespace, self).get_dimensions()

    @staticmethod
    def get_flatWeights(E_cm, n, mass=None):
        """ Return the phase-space volume for a n massless final states.
        Vol(E_cm, n) = (pi/2)^(n-1) *  (E_cm^2)^(n-2) / ((n-1)!*(n-2)!)
        """
        if n == 1:
            # The jacobian from \delta(s_hat - m_final**2) present in 2->1 convolution
            # must typically be accounted for in the MC integration framework since we
            # don't have access to that here, so we just return 1.
            return 1.

        return math.pow((math.pi/2.0), n-1) *\
            (math.pow((E_cm**2), n-2)/(math.factorial(n-1)*math.factorial(n-2)))

    @staticmethod
    def bisect(v, n, target=1.e-16, maxLevel=80):
        """Solve v = (n+2) * u^(n+1) - (n+1) * u^(n+2) for u."""

        if (v == 0. or v == 1.):
            return v

        level = 0
        left = 0.
        right = 1.

        checkV = -1.
        u = -1.

        while (level < maxLevel):
            u = (left + right) * (0.5**(level + 1))
            checkV = (u**(n+1)) * (n+2.-(n+1.)*u)
            error = abs(1. - checkV / v)
            if (error == 0. or error <= target):
                break
            left *= 2.
            right *= 2.
            if (v <= checkV):
                right -= 1.
            else:
                left += 1.
            level += 1

        return u

    @staticmethod
    def rho(M, N, m):
        """Returns sqrt((sqr(M)-sqr(N+m))*(sqr(M)-sqr(N-m)))/(8.*sqr(M))"""

        Msqr = M**2
        return ((Msqr-(N+m)**2) * (Msqr-(N-m)**2))**0.5 / (8.*Msqr)

    def setInitialStateMomenta(self, output_momenta, E_cm):
        """Generate the initial state momenta."""

        if self.n_initial not in [1, 2]:
            raise InvalidCmd(
                "This PS generator only supports 1 or 2 initial states")

        if self.n_initial == 1:
            if self.initial_masses[0] == 0.:
                raise PhaseSpaceGeneratorError(
                    "Cannot generate the decay phase-space of a massless particle.")
            if self.E_cm != self.initial_masses[0]:
                raise PhaseSpaceGeneratorError(
                    "Can only generate the decay phase-space of a particle at rest.")

        if self.n_initial == 1:
            output_momenta[0] = LorentzVector(
                [self.initial_masses[0], 0., 0., 0.])
            return

        elif self.n_initial == 2:
            if self.initial_masses[0] == 0. or self.initial_masses[1] == 0.:
                output_momenta[0] = LorentzVector(
                    [E_cm/2.0, 0., 0., +E_cm/2.0])
                output_momenta[1] = LorentzVector(
                    [E_cm/2.0, 0., 0., -E_cm/2.0])
            else:
                M1sq = self.initial_masses[0]**2
                M2sq = self.initial_masses[1]**2
                E1 = (E_cm**2+M1sq-M2sq) / E_cm
                E2 = (E_cm**2-M1sq+M2sq) / E_cm
                Z = math.sqrt(E_cm**4 - 2*E_cm**2*M1sq - 2*E_cm **
                              2*M2sq + M1sq**2 - 2*M1sq*M2sq + M2sq**2) / E_cm
                output_momenta[0] = LorentzVector([E1/2.0, 0., 0., +Z/2.0])
                output_momenta[1] = LorentzVector([E2/2.0, 0., 0., -Z/2.0])
        return

    def get_PS_point(self, random_variables):
        """Generate a complete PS point, including Bjorken x's,
        dictating a specific choice of incoming particle's momenta.
        """

        # if random_variables are not defined, than just throw a completely random point
        if random_variables is None:
            random_variables = self.dimensions.random_sample()

        # Check the sensitivity of te inputs from the integrator
        if any(math.isnan(r) for r in random_variables):
            logger.warning('Some input variables from the integrator are malformed: %s' %
                           (', '.join('%s=%s' % (name, random_variables[pos]) for name, pos in
                                      self.dim_name_to_position.items())))
            logger.warning(
                'The PS generator will yield None, triggering the point to be skipped.')
            return None, 0.0, (0., 0.), (0., 0.)

        # Phase-space point weight to return
        wgt = 1.0

        # if any(math.isnan(r) for r in random_variables):
        #    misc.sprint(random_variables)

        # Avoid extrema since the phase-space generation algorithm doesn't like it
        random_variables = [min(max(rv, self.epsilon_border),
                                1.-self.epsilon_border) for rv in random_variables]

        # Assign variables to their meaning.
        if 'ycms' in self.dim_name_to_position:
            PDF_ycm = random_variables[self.dim_name_to_position['ycms']]
        else:
            PDF_ycm = None
        if 'tau' in self.dim_name_to_position:
            PDF_tau = random_variables[self.dim_name_to_position['tau']]
        else:
            PDF_tau = None
        PS_random_variables = [rv for i, rv in enumerate(
            random_variables) if self.position_to_dim_name[i].startswith('x_')]

        # Also generate the ISR collinear factorization convolutoin variables xi<i> if
        # necessary. In order for the + distributions of the PDF counterterms and integrated
        # collinear ISR counterterms to hit the PDF only (and not the matrix elements or
        # observables functions), a change of variable is necessary: xb_1' = xb_1 * xi1
        if self.correlated_beam_convolution:
            # Both xi1 and xi2 must be set equal then
            xi1 = random_variables[self.dim_name_to_position['xi']]
            xi2 = random_variables[self.dim_name_to_position['xi']]
        else:
            if self.is_beam_factorization_active[0]:
                xi1 = random_variables[self.dim_name_to_position['xi1']]
            else:
                xi1 = None
            if self.is_beam_factorization_active[1]:
                xi2 = random_variables[self.dim_name_to_position['xi2']]
            else:
                xi2 = None

        # Now take care of the Phase-space generation:
        # Set some defaults for the variables to be set further
        xb_1 = 1.
        xb_2 = 1.
        E_cm = self.collider_energy

        # We generate the PDF from two variables \tau = x1*x2 and ycm = 1/2 * log(x1/x2), so that:
        #  x_1 = sqrt(tau) * exp(+ycm)
        #  x_2 = sqrt(tau) * exp(-ycm)
        # The jacobian of this transformation is 1.
        if abs(self.beam_types[0]) == abs(self.beam_types[1]) == 1:

            tot_final_state_masses = sum(self.masses)
            if tot_final_state_masses > self.collider_energy:
                raise PhaseSpaceGeneratorError(
                    "Collider energy is not large enough, there is no phase-space left.")

            # Keep a hard cut at 1 GeV, which is the default for absolute_Ecm_min
            tau_min = (max(tot_final_state_masses,
                       self.absolute_Ecm_min)/self.collider_energy)**2
            tau_max = 1.0

            if self.n_initial == 2 and self.n_final == 1:
                # Here tau is fixed by the \delta(xb_1*xb_2*s - m_h**2) which sets tau to
                PDF_tau = tau_min
                # Account for the \delta(xb_1*xb_2*s - m_h**2) and corresponding y_cm matching to unit volume
                wgt *= (1./self.collider_energy**2)
            else:
                # Rescale tau appropriately
                PDF_tau = tau_min+(tau_max-tau_min)*PDF_tau
                # Including the corresponding Jacobian
                wgt *= (tau_max-tau_min)

            # And we can now rescale ycm appropriately
            ycm_min = 0.5 * math.log(PDF_tau)
            ycm_max = -ycm_min
            PDF_ycm = ycm_min + (ycm_max - ycm_min)*PDF_ycm
            # and account for the corresponding Jacobian
            wgt *= (ycm_max - ycm_min)

            xb_1 = math.sqrt(PDF_tau) * math.exp(PDF_ycm)
            xb_2 = math.sqrt(PDF_tau) * math.exp(-PDF_ycm)
            # /!\ The mass of initial state momenta is neglected here.
            E_cm = math.sqrt(xb_1*xb_2)*self.collider_energy

        elif self.beam_types[0] == self.beam_types[1] == 0:
            xb_1 = 1.
            xb_2 = 1.
            E_cm = self.collider_energy
        else:
            raise InvalidCmd(
                "This basic PS generator does not yet support collider mode (%d,%d)." % self.beam_types)

        # Now generate a PS point
        PS_point, PS_weight = self.generateKinematics(
            E_cm, PS_random_variables)

        # Apply the phase-space weight
        wgt *= PS_weight

        return LorentzVectorList(PS_point), wgt, (xb_1, xi1), (xb_2, xi2)

    def generateKinematics(self, E_cm, random_variables):
        """Generate a self.n_initial -> self.n_final phase-space point
        using the random variables passed in argument.
        """

        # Make sure the right number of random variables are passed
        assert (len(random_variables) == self.nDimPhaseSpace())

        # Make sure that none of the random_variables is NaN.
        if any(math.isnan(rv) for rv in random_variables):
            raise PhaseSpaceGeneratorError("Some of the random variables passed " +
                                           "to the phase-space generator are NaN: %s" % str(random_variables))

        # The distribution weight of the generate PS point
        weight = 1.

        output_momenta = []

        mass = self.masses[0]
        if self.n_final == 1:
            if self.n_initial == 1:
                raise InvalidCmd("1 > 1 phase-space generation not supported.")
            if mass/E_cm < 1.e-7 or ((E_cm-mass)/mass) > 1.e-7:
                raise PhaseSpaceGeneratorError(
                    "1 > 2 phase-space generation needs a final state mass equal to E_c.o.m.")
            output_momenta.append(LorentzVector([mass/2., 0., 0., +mass/2.]))
            output_momenta.append(LorentzVector([mass/2., 0., 0., -mass/2.]))
            output_momenta.append(LorentzVector([mass, 0., 0.,       0.]))
            weight = self.get_flatWeights(E_cm, 1)
            return output_momenta, weight

        M = [0.]*(self.n_final-1)
        M[0] = E_cm

        weight *= self.generateIntermediatesMassive(M, E_cm, random_variables)
        M.append(self.masses[-1])

        Q = LorentzVector([M[0], 0., 0., 0.])
        nextQ = LorentzVector()

        for i in range(self.n_initial+self.n_final-1):

            if i < self.n_initial:
                output_momenta.append(LorentzVector())
                continue

            q = 4.*M[i-self.n_initial]*self.rho(
                M[i-self.n_initial], M[i-self.n_initial+1], self.masses[i-self.n_initial])
            cos_theta = 2. * \
                random_variables[self.n_final-2+2*(i-self.n_initial)]-1.
            sin_theta = math.sqrt(1.-cos_theta**2)
            phi = 2.*math.pi * \
                random_variables[self.n_final-1+2*(i-self.n_initial)]
            cos_phi = math.cos(phi)
            sin_phi = math.sqrt(1.-cos_phi**2)

            if (phi > math.pi):
                sin_phi = -sin_phi

            p = LorentzVector([0., q*sin_theta*cos_phi, q *
                              sin_theta*sin_phi, q*cos_theta])
            p.set_square(self.masses[i-self.n_initial]**2)
            p.boost(Q.boostVector())
            p.set_square(self.masses[i-self.n_initial]**2)
            output_momenta.append(p)

            nextQ = Q - p
            nextQ.set_square(M[i-self.n_initial+1]**2)
            Q = nextQ

        output_momenta.append(Q)

        self.setInitialStateMomenta(output_momenta, E_cm)

        return LorentzVectorList(output_momenta), weight

    def generateIntermediatesMassless(self, M, E_cm, random_variables):
        """Generate intermediate masses for a massless final state."""

        for i in range(2, self.n_final):
            u = self.bisect(random_variables[i-2], self.n_final-1-i)
            M[i-1] = math.sqrt(u*(M[i-2]**2))

        return self.get_flatWeights(E_cm, self.n_final)

    def generateIntermediatesMassive(self, M, E_cm, random_variables):
        """Generate intermediate masses for a massive final state."""

        K = list(M)
        K[0] -= sum(self.masses)

        weight = self.generateIntermediatesMassless(K, E_cm, random_variables)
        del M[:]
        M.extend(K)

        for i in range(1, self.n_final):
            for k in range(i, self.n_final+1):
                M[i-1] += self.masses[k-1]

        weight *= 8.*self.rho(
            M[self.n_final-2],
            self.masses[self.n_final-1],
            self.masses[self.n_final-2])

        for i in range(2, self.n_final):
            weight *= (self.rho(M[i-2], M[i-1], self.masses[i-2]) /
                       self.rho(K[i-2], K[i-1], 0.)) * (M[i-1]/K[i-1])

        weight *= math.pow(K[0]/M[0], 2*self.n_final-4)

        return weight

    def invertKinematics(self, E_cm, momenta):
        """ Returns the random variables that yields the specified momenta configuration."""

        # Make sure the right number of momenta are passed
        assert (len(momenta) == (self.n_initial + self.n_final))
        moms = momenta.get_copy()

        # The weight of the corresponding PS point
        weight = 1.

        if self.n_final == 1:
            if self.n_initial == 1:
                raise PhaseSpaceGeneratorError(
                    "1 > 1 phase-space generation not supported.")
            return [], self.get_flatWeights(E_cm, 1)

        # The random variables that would yield this PS point.
        random_variables = [-1.0]*self.nDimPhaseSpace()

        M = [0., ]*(self.n_final-1)
        M[0] = E_cm

        Q = [LorentzVector(), ]*(self.n_final-1)
        Q[0] = LorentzVector([M[0], 0., 0., 0.])

        for i in range(2, self.n_final):
            for k in range(i, self.n_final+1):
                Q[i-1] = Q[i-1] + moms[k+self.n_initial-1]
            M[i-1] = abs(Q[i-1].square()) ** 0.5

        weight = self.invertIntermediatesMassive(M, E_cm, random_variables)

        for i in range(self.n_initial, self.n_final+1):
            # BALDY another copy? moms not used afterwards
            p = LorentzVector(moms[i])
            # Take the opposite boost vector
            boost_vec = -Q[i-self.n_initial].boostVector()
            p.boost(boost_vec)
            random_variables[self.n_final-2+2 *
                             (i-self.n_initial)] = (p.cosTheta()+1.)/2.
            phi = p.phi()
            if (phi < 0.):
                phi += 2.*math.pi
            random_variables[self.n_final-1+2 *
                             (i-self.n_initial)] = phi / (2.*math.pi)

        return random_variables, weight

    def invertIntermediatesMassive(self, M, E_cm, random_variables):
        """ Invert intermediate masses for a massive final state."""

        K = list(M)
        for i in range(1, self.n_final):
            K[i-1] -= sum(self.masses[i-1:])

        weight = self.invertIntermediatesMassless(K, E_cm, random_variables)
        weight *= 8.*self.rho(M[self.n_final-2],
                              self.masses[self.n_final-1],
                              self.masses[self.n_final-2])
        for i in range(2, self.n_final):
            weight *= (self.rho(M[i-2], M[i-1], self.masses[i-2])/self.rho(K[i-2], K[i-1], 0.)) \
                * (M[i-1]/K[i-1])

        weight *= math.pow(K[0]/M[0], 2*self.n_final-4)

        return weight

    def invertIntermediatesMassless(self, K, E_cm, random_variables):
        """ Invert intermediate masses for a massless final state."""

        for i in range(2, self.n_final):
            u = (K[i-1]/K[i-2])**2
            random_variables[i-2] = \
                (self.n_final+1-i)*math.pow(u, self.n_final-i) - \
                (self.n_final-i)*math.pow(u, self.n_final+1-i)

        return self.get_flatWeights(E_cm, self.n_final)
