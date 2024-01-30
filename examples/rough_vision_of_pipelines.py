import hermes
import numpy as np

#############################
"""Rough example of building a custom pipeline"""
#############################
Pipeline = hermes.Pipeline(
    Instrument = hermes.instruments.CHESSQM2Beamline(
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

    initialization_method = hermes.loopcontrols.RandomStart(domain, start_measurements)



    #Initialize the archiver

    #Data Analysis steps
    Data_analysis = hermes.Pipeline(
        cluster_method = hermes.clustering.RBPots(measurements=measurements, 
                                                    measurements_distance_type= hermes.distance.CosineDistance(),
                                                    measurements_similarity_type= hermes.similarity.SquaredExponential(lengthscale=0.01),
                                                    locations = locations,
                                                    resolution = 0.2,)
        classification_method = hermes.classification.HeteroscedasticGPC(
            indexes = indexes,
            measured_indexes = measured_indexes,
            locations = locations,
            labels = cluster_method.labels,
            domain = domain,
            probabilities = cluster_method.probabilities
        )
        acquisition_method = hermes.acquire.PureExplore(classification_method.unmeasured_locations,
                                            classification_method.mean_unmeasured,
                                            classification_method.var_unmeasured)
        archiver = hermes.archive.CombiMappingModels(save_directory = "/nfs/chess/id4baux/2023-2/sarker-3729-a/hermes_061623/models/",
                                                instrument = QM2_instrument)

    )
    #Stoping Crideria
        AL_loops = 1
        convergance = hermes.loopcontrols.convergance(classification_outputs)

)

#############################
"""Pre-built example"""
#############################
from Hermes.pipelines import PhaseMappingPipeline

pipeline = PhaseMappingPipeline(instrument = hermes.instruments.CHESSQM2Beamline())
