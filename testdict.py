import timeit
from statistics import mean


def dict_search():
    SETUP_CODE = '''
from dataclasses import dataclass

LOCATIONS = {
    "EuclidianDistance": False,
    "CosineDistance": True,
}  # TODO add similarities


class BaseDS:  # base distance and similarity class
    """Base Class for Distance and Similarity types."""

    def needs_locations(self):
        return LOCATIONS[self.__class__.__name__]


@dataclass
class EuclidianDistance(BaseDS):
    """Euclidian Distance. L2Norm."""
    '''

    TEST_CODE = """
ds = EuclidianDistance()
ds.needs_locations()
    """

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=10, number=10**6)

    # printing minimum exec. time
    print("Dict search time: {}".format(mean(times)))


def attr_search():
    SETUP_CODE = '''
from dataclasses import dataclass

class BaseDS:  # base distance and similarity class
    """Base Class for Distance and Similarity types."""


@dataclass
class EuclidianDistance(BaseDS):
    """Euclidian Distance. L2Norm."""
    @property
    def needs_locations(self):
        return False

    '''

    TEST_CODE = """
ds = EuclidianDistance()
ds.needs_locations
    """

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=10, number=10**6)

    # printing minimum exec. time
    print("Attr search time: {}".format(mean(times)))


def attr_search2():
    SETUP_CODE = '''
from dataclasses import dataclass

class BaseDS:  # base distance and similarity class
    """Base Class for Distance and Similarity types."""


@dataclass
class EuclidianDistance(BaseDS):
    """Euclidian Distance. L2Norm."""

    def needs_locations(self):
        return False

    '''

    TEST_CODE = """
ds = EuclidianDistance()
ds.needs_locations()
    """

    # timeit.repeat statement
    times = timeit.repeat(setup=SETUP_CODE, stmt=TEST_CODE, repeat=10, number=10**6)

    # printing minimum exec. time
    print("Attr search2 time: {}".format(mean(times)))


if __name__ == "__main__":
    dict_search()
    attr_search()
    attr_search2()
