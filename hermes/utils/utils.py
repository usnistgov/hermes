from numpy import ndarray
from sklearn.metrics import pairwise_distances
from orix.quaternion.orientation import Misorientation
from orix.quaternion import symmetry





def compute_distance(tp: str, x: ndarray):
    '''Compute the pairwise distances for each elment of x to every elment of x.
    x should be a (n x m) matrix, where n is the number of entries, and m is the dimensions.
    
    From sklearn valid tp's are: [‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’], 
    From scipy valid tp's are: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘correlation’, ‘dice’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
    '''
    distance_matrix = pairwise_distances(x, metric = 'tp') 
    return distance_matrix

def compute_orientation_distance(symmetry: str, x: ndarray):
    '''Compute all the pairwise orientation distance for each entry in x'''
    distance_matrix = 
    



def compute_similarity(tp: str, distance_matrix: ndarray):
    match tp:
        case "RBF": #Radial Basis Function
            similarity_matrix = np.exp(-(distance_matrix/lengthscale)**2)
        case: "Inverse": 
            similarity_matrix = 1/(distance_matrix + delta)

    return similarity_matrix


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