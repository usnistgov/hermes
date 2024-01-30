import hermes

import numpy as np


# Set up the domain (locations where we have measurements)
indexes = np.arange(0,100)

locations_x = np.linspace(-10,10,10).reshape(-1,1)
locations_y = np.linspace(-10,10,10).reshape(-1,1)

grid_locations_x, grid_locations_y = np.meshgrid(locations_x,locations_y)

domain = np.concatenate((grid_locations_x.reshape(-1,1),
                            grid_locations_y.reshape(-1,1)), axis = 1)


# Set up the labels
#Assuming 3 classes

class_0 = np.ones(100)*0.5
class_0[0:25] = 0
class_0[70:] = 0.25

class_1 = np.ones(100)*0.25
class_1[25:70] = 0.25
class_1[70:] = 0.5

class_2 = 1 - (class_0 + class_1)

all_probabilities = np.concatenate((class_0.reshape(-1,1),
                                class_1.reshape(-1,1),
                                class_2.reshape(-1,1)), axis = 1)

all_labels = np.argmax(all_probabilities, axis = 1).reshape(-1,1)


#Pick points that are measured:
num_measurements = 50
measured_indexes = np.random.permutation(indexes)[0:num_measurements]

locations = domain[measured_indexes]
probabilites = all_probabilities[measured_indexes]
labels = all_labels[measured_indexes]

# Hermes Heteroscedastic GPC
classification_method = hermes.classification.HeteroscedasticGPC(
        indexes = indexes,
        measured_indexes = measured_indexes,
        locations = locations,
        labels = labels,
        domain = domain,
        probabilities = probabilities
    )

#Train on the measured locations
classification_method.train()
#Predict the lables everywhere in the Domain
classification_method.predict()
#Predict the lables at just the unmeasured locations in the Domain
classification_method.predict_unmeasured()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#### Asperational ML as a Service Workflow #####

# #Train on the measured locations
# model_path = "/some/path/to/save/the/model/"
# classification_method.ML_service_train(model_path)

# #Predict the lables everywhere in the Domain
# domain_path = "/some/path/to/save/the/data/of/the/domain/locations/"
# classification_method.ML_as_service_predict(domain_path)

# #Predict the lables at just the unmeasured locations in the Domain
# unmeasured_locations_path = "/some/path/to/save/data/of/unmeasured/locations/"
# classification_method.ML_as_service_predict_unmeasured(unmeasured_locations_path)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~