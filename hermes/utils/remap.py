"""Remap data."""


class ListNotSorted(Exception):
    """Raised when input list is not sorted."""


class OutOfRange(Exception):
    """Raised when range of input lists differ."""


def _raise_not_sorted(name: str):
    raise ListNotSorted("input list %s is not sorted" % (name))


def rescale_2d_data_linear(x1, y1, x2):
    # Gilad: include "independent variable"
    # Austin: rebinning
    # linear interopolation
    y2 = list()
    if min(x2) < min(x1):
        raise OutOfRange("Out of range min(x2) < min(x1)")
    if max(x2) > max(x1):
        raise OutOfRange("Out of range max(x2) > max(x1)")
    if x1 != sorted(x1):
        _raise_not_sorted("x1")
    if x2 != sorted(x2):
        _raise_not_sorted("x2")

    id_x = 0
    for x in x2:
        y = None
        x_p0 = x1[id_x]
        if x == x_p0:
            # perfect match
            y = y1[id_x]
        else:
            x_p1 = x1[id_x + 1]
            while x_p1 < x:
                id_x = id_x + 1
                x_p0 = x1[id_x]
                x_p1 = x1[id_x + 1]
            if x == x_p0:
                # perfect match
                y = y1[id_x]
            elif x == x_p1:
                # perfect match
                y = y1[id_x + 1]
            elif x_p0 < x and x < x_p1:
                # interpolate
                y_p0 = y1[id_x]
                y_p1 = y1[id_x + 1]
                y = (x - x_p0) * (y_p1 - y_p0) / (x_p1 - x_p0) + y_p0
            else:
                raise Exception("Something went wrong during interpolation")
        if y:
            y2.append(y)
        else:
            raise Exception("Something went wrong during rescale at location:", x)

    if len(x2) != len(y2):
        raise Exception("Something went wrong lists not equal")

    return y2
