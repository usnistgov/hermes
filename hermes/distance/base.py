"""Shared methods for distance and similarity classes."""

LOCATIONS = {
    "EuclidianDistance": False,
    "CosineDistance": True,
}  # TODO add similarities


class BaseDS:  # base distance and similarity class
    def needs_locations(self):
        return LOCATIONS[self.__class__.__name__]
