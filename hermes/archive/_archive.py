from dataclasses import dataclass
from datetime import datetime

import numpy as np
import json

@dataclass
class Archiver:
    """Base class for archiving"""


@dataclass
class JSONizer(Archiver):
    """Class for writing JSON's"""

@dataclass
class CombiMappingModels(JSONizer):
    """For archiving the models analyze combi wafers with:
    Instrument
    Clustering Model
    Classification Model"""
    save_directory = "/some/directory/"


    instrument: Any
    cluster_method: Any
    classification_method: Any 
    
    next_indexes: Any   
    locations: Any 
    measurements: Any 


    def write_metadata_file(self):
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%Y-%m-%d")

        results_dictionary = {}

        results_dictionary["Timestamp"] = timestampStr
    
        results_dictionary["Domain"] = self.instrument.composition_domain
        results_dictionary["Two_Theta_Space"] = self.instrument.two_theta_space

        results_dictionary["Next_indexes"] = self.next_indexes

        filename = "Loop_start_"+timestampStr+".json"
        fullfilename = self.save_directory+filename
        with open(fullfilename, "w") as outfile:
            json.dump(results_dictionary, outfile)

    def write_loopdata_file(self, Loop_index:int):
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%Y-%m-%d")

        results_dictionary = {}

        results_dictionary["Timestamp"] = timestampStr

        results_dictionary["locations"] = self.locations
        results_dictionary["measurements"] = self.measurements

        results_dictionary["labels"] = self.cluster_method.labels
        results_dictionary["probabilities"] = self.cluster_method.probabilities

        results_dictionary["kernel"] = self.classification_method.model.kernel.name
        results_dictionary["k_lengthscale"] = self.classification_method.model.kernel.lengthscales.numpy()
        results_dictionary["k_variance"] = self.classification_method.model.kernel.variance.numpy()

        results_dictionary["mean"] = self.classification_method.mean
        results_dictionary["var"] = self.classification_method.var

        results_dictionary["Next_indexes"] = self.next_indexes

        filename = f"Loop_{Loop_index}"+timestampStr+".json"
        fullfilename = self.save_directory+filename
        with open(fullfilename, "w") as outfile:
            json.dump(results_dictionary, outfile)
