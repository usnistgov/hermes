import hermes
import numpy as np
import pandas as pd
#Set up the instrument 

#sim_load_dir = "/nfs/chess/id4baux/2023-2/sarker-3729-a/hermes_061623/coordinate/"
#wafer_coords_file = "XY_Coordinates_177.txt"
#wafer_composition_file = "CombiView_Format_GeSbTe_Composition.txt"
#wafer_xrd_file = "GeSbTe_XRD_MetaStable_Background subtracted and with normalization.txt"



QM2_instrument = hermes.instruments.CHESSQM2Beamline(
                                    simulation = False,
                                    #sample_name = "CoCaAl020222p",
                                    wafer_directory = "/nfs/chess/id4baux/2023-2/sarker-3729-a/hermes_061623/coordinate/",
                                    wafer_coords_file = "XY_Coordinates_binary_fine3.csv",
                                    #wafer_composition_file = "CombiView_Format_GeSbTe_Composition.txt",
                                    #wafer_xrd_file = "GeSbTe_XRD_MetaStable_Background subtracted and with normalization.txt",
                                    diffraction_space_bins = 10000,
)

#Define the domains
domain = QM2_instrument.xy_locations.to_numpy()
#domain_2d = QM2_instrument.composition_domain_2d
#domain_3d = QM2_instrument.composition_domain[1]

#Get the indexes in the domain:
indexes = np.arange(0, domain.shape[0])



#Initialize containers for locations and measurements:
#measured_indexes = np.array([])
#locations = np.array([]).reshape(-1,domain.shape[1])
#measurements = np.array([]).reshape(-1, QM2_instrument.diffraction_space_bins)
# picking up from run that stopped 
measured_indexes = np.array([14, 19, 43, 12, 70, 67, 42, 23, 21, 63, 24])
locations = domain[measured_indexes]
measurements = np.array([]).reshape(-1, QM2_instrument.diffraction_space_bins)
for idx in measured_indexes:
    file_loc = f"/nfs/chess/id4baux/2023-2/sarker-3729-a/hermes_061623/AlZr_060823_1/AlZr_060823_1_{idx}/AlZr_060823_1_{idx}_PIL10_001_000_integrated.csv"
    measurement = pd.read_table(file_loc, delimiter = ",")
    measurement = measurement.to_numpy()[:,1].reshape(1,-1)
    measurements = np.concatenate((measurements, measurement), axis=0)
print(measurements.shape)

#Choose the initial locations
start_measurements = 1
flag = True
while flag == True:
    initialization_method = hermes.loopcontrols.RandomStart(domain, start_measurements)
    next_indexes = initialization_method.initialize()
    flag = np.isin(measured_indexes, next_indexes)[0]
    print("next_indexes =", next_indexes)
    next_locations = domain[next_indexes]

#Initialize the archiver
archiver = hermes.archive.CombiMappingModels(save_directory = "/nfs/chess/id4baux/2023-2/sarker-3729-a/hermes_061623/models/",
                                             instrument = QM2_instrument)
archiver.next_indexes = next_indexes
archiver.write_metadata_file()

AL_loops = 76 - start_measurements + 1
print("Starting Loop")

for n in range(AL_loops):

    print("Moving and Measureing")

    next_measurements = QM2_instrument.move_and_measure(next_indexes)
    measured_indexes = np.append(measured_indexes, next_indexes)
    locations = np.append(locations, next_locations, axis =0)
    measurements = np.append(measurements, next_measurements, axis = 0)

    print(locations.shape)
    print(measurements.shape)

    cluster_method = hermes.clustering.RBPots(measurements=measurements, 
                                            measurements_distance_type= hermes.distance.CosineDistance(),
                                            measurements_similarity_type= hermes.similarity.SquaredExponential(lengthscale=0.01),
                                            locations = locations,
                                            resolution = 0.2,
                                            )
    cluster_method.form_graph()
    cluster_method.cluster()
    cluster_method.get_local_membership_prob()

    classification_method = hermes.classification.HeteroscedasticGPC(
        indexes = indexes,
        measured_indexes = measured_indexes,
        locations = locations,
        labels = cluster_method.labels,
        domain = domain,
        probabilities = cluster_method.probabilities
    )

    classification_method.train()
    classification_method.predict()
    classification_method.predict_unmeasured()

    acquisition_method = hermes.acquire.PureExplore(classification_method.unmeasured_locations,
                                    classification_method.mean_unmeasured,
                                    classification_method.var_unmeasured)

    next_locations = acquisition_method.calculate()
    next_indexes = classification_method.return_index(next_locations)

    print("Next location = ", next_locations)

   

    archiver.cluster_method = cluster_method
    archiver.classification_method = classification_method
    archiver.measured_indexes = measured_indexes
    archiver.next_indexes = next_indexes

    archiver.write_loopdata_file(Loop_index = n)
    
