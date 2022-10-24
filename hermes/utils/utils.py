from numpy import ndarray
def compute_similarity(tp: str, locations: ndarray, measurements: ndarray):
    match tp:
        case "normal":
            similarity = locations/measurements
        case "reverese":
            similarity = measurements/locations
    return similarity

def compute_distance(tp: str, locations: ndarray, measurements: ndarray):
    match tp:
        case "normal":
            distance = locations/measurements
        case "reverese":
            distance = measurements/locations
    return distance