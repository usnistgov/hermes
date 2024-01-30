
import hermes

hermes.instruments.CHESSQM2Beamline()

instrument = hermes.instruments.CHESSQM2Beamline(
                                        simulation = False,
                                        #sample_name = "CoCaAl020222p",
                                        wafer_directory = "/nfs/chess/id4baux/2023-2/sarker-3729-a/hermes_061623/coordinate/",
                                        wafer_coords_file = "XY_Coordinates_binary_fine3.csv",
                                        #wafer_composition_file = "CombiView_Format_GeSbTe_Composition.txt",
                                        #wafer_xrd_file = "GeSbTe_XRD_MetaStable_Background subtracted and with normalization.txt",
                                        diffraction_space_bins = 10000,
    )

Pipeline = hermes.Pipeline(instrument=instrument,
    #Define the domains
    domain = instrument.xy_locations.to_numpy(),

    initialization_method = hermes.loopcontrols.RandomStart(domain, 0),

    #Data Analysis steps
    data_analysis = hermes.Pipeline(
        cluster_method = hermes.clustering.RBPots(measurements=measurements, 
                                                    measurements_distance_type= hermes.distance.CosineDistance(),
                                                    measurements_similarity_type= hermes.similarity.SquaredExponential(lengthscale=0.01),
                                                    locations = locations,
                                                    resolution = 0.2),
        classification_method = hermes.classification.HeteroscedasticGPC(
            indexes = indexes,
            measured_indexes = measured_indexes,
            locations = locations,
            labels = cluster_method.labels,
            domain = domain,
            probabilities = cluster_method.probabilities
        ),
        acquisition_method = hermes.acquire.PureExplore(classification_method.unmeasured_locations,
                                            classification_method.mean_unmeasured,
                                            classification_method.var_unmeasured),
        archiver = hermes.archive.CombiMappingModels(save_directory = "/nfs/chess/id4baux/2023-2/sarker-3729-a/hermes_061623/models/",
                                                instrument = QM2_instrument)

    )
    #Stoping Crideria
        AL_loops = 1,
        convergance = hermes.loopcontrols.convergance(classification_outputs),

)
# this method good:
Pipeline = hermes.Pipeline()
Pipeline.instrument = hermes.instruments.CHESSQM2Beamline()
Pipeline.domain = Pipeline.instrument.xy_locations.to_numpy()

# hermes.Pipeline(instrument=instrument, domain="xy_locations")
