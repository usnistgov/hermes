import hermes
import numpy as np

#Set up the instrument 

#sim_load_dir = "/nfs/chess/id4baux/2023-2/sarker-3729-a/hermes_061623/coordinate/"
#wafer_coords_file = "XY_Coordinates_177.txt"
#wafer_composition_file = "CombiView_Format_GeSbTe_Composition.txt"
#wafer_xrd_file = "GeSbTe_XRD_MetaStable_Background subtracted and with normalization.txt"



QM2_instrument = hermes.instruments.CHESSQM2Beamline(simulation = False,
                                    wafer_directory = "/nfs/chess/id4baux/2023-2/sarker-3729-a/hermes_061623/coordinate/",
                                    wafer_coords_file = "XY_Coordinates_177.txt",
                                    #wafer_composition_file = "CombiView_Format_GeSbTe_Composition.txt",
                                    #wafer_xrd_file = "GeSbTe_XRD_MetaStable_Background subtracted and with normalization.txt",
                                    diffraction_space_bins = 10000,
)

#Define the domains
domain = QM2_instrument.xy_locations.to_numpy()
#domain_2d = QM2_instrument.composition_domain_2d
#domain_3d = QM2_instrument.composition_domain[1]

#Choose the initial locations
start_measurements = 4
initialization_method = hermes.loopcontrols.RandomStart(domain, start_measurements)
next_indexes = initialization_method.initialize()
print("next_indexes =", next_indexes)
next_locations = domain[next_indexes]

#Initialize containers for locations and measurements:
locations = np.array([]).reshape(-1,domain.shape[1])
measurements = np.array([]).reshape(-1, QM2_instrument.diffraction_space_bins)

#Initialize the archiver
archiver = hermes.archive.CombiMappingModels(save_directory = "/nfs/chess/id4baux/2023-2/sarker-3729-a/hermes_061623/models/",
                                             instrument = QM2_instrument)
archiver.next_indexes = next_indexes
archiver.write_metadata_file()

AL_loops = 173
print("Starting Loop")

for n in range(AL_loops):

    print("Moving and Measureing")

    next_measurements = QM2_instrument.move_and_measure(next_indexes)

    locations = np.append(locations, next_locations, axis =0)
    measurements = np.append(measurements, next_measurements, axis = 0)

    print(locations.shape)
    print(measurements.shape)

    cluster_method = hermes.clustering.RBPots(measurements=measurements, 
                                            measurements_distance_type= hermes.distance.CosineDistance(),
                                            measurements_similarity_type= hermes.similarity.SquaredExponential(lengthscale=0.1),
                                            locations = locations,
                                            resolution = 0.2,
                                            )
    cluster_method.form_graph()
    cluster_method.cluster()
    cluster_method.get_local_membership_prob()

    classification_method = hermes.classification.HeteroscedasticGPC(
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
    archiver.next_indexes = next_indexes
    archiver.locations = locations
    archiver.measurements = measurements

    archiver.write_loopdata_file(Loop_index = n)
    
