import hermes


#Set up the instrument 

sim_load_dir = "C:/Users/asm6/hermes_CHESS/"
wafer_coords_file = "XY_Coordinates_177.txt"
wafer_composition_file = "CombiView_Format_GeSbTe_Composition.txt"
wafer_xrd_file = "GeSbTe_XRD_MetaStable_Background subtracted and with normalization.txt"



QM2_instrument = hermes.instruments.CHESSQM2Beamline(simulation = True,
                                    wafer_directory = "C:/Users/asm6/hermes_CHESS/",
                                    wafer_coords_file = "XY_Coordinates_177.txt",
                                    wafer_composition_file = "CombiView_Format_GeSbTe_Composition.txt",
                                    wafer_xrd_file = "GeSbTe_XRD_MetaStable_Background subtracted and with normalization.txt")

#Define the domains
domain_2d = QM2_instrument.composition_domain_2d
domain_3d = QM2_instrument.composition_domain[1]

#Choose the initial locations
start_measurements = 11
initialization_method = hermes.loopcontrols.RandomStart(domain_2d, start_measurements)
next_indexes = initialization_method.initialize() 
next_locations = domain_2d[next_indexes]

#Initialize containers for locations and measurements:
locations = np.array([]).reshape(-1,domain_2d.shape[1])
measurements = np.array([]).reshape(-1, QM2_instrument.two_theta_space.shape[0])

#Initialize the archiver
archiver = hermes.archive.CombiMappingModels(save_directory = "/some/directory/",
                                             instrument = QM2_instrument)
archiver.write_metadata_file()

AL_loops = 2

for n in range(AL_loops):

    next_measurements = QM2_instrument.move_and_measure(domain_3d[next_indexes])

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
        domain = domain_2d,
        probabilities = cluster_method.probabilities
    )

    classification_method.train()
    classification_method.predict_unmeasured()

    acquisition_method = hermes.acquire.PureExplore(classification_method.unmeasured_locations,
                                    classification_method.mean_unmeasured,
                                    classification_method.var_unmeasured)

    next_locations = acquisition_method.calculate()
    next_indexes = classification_method.return_index(next_locations)

    archiver.cluster_method = cluster_method
    archiver.classification_method = classification_method
    archiver.next_indexes = next_indexes
    archiver.locations = locations
    archiver.measurements = measurements

    archiver.write_loopdata_file(Loop_index = n)
    
