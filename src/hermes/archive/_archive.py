from dataclasses import dataclass
from datetime import datetime

from typing import Any, Optional

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
    save_directory: any #"/some/directory/"

    instrument: any
    
    cluster_method: Optional[Any] = None
    classification_method: Optional[Any] = None 
    
    measured_indexes: Optional[Any] = None
    next_indexes: Optional[Any] = None
    # locations: Optional[Any] = None
    # measurements: Optional[Any] = None


    def write_metadata_file(self):
        dateTimeObj = datetime.now()
        # timestampStr = dateTimeObj.strftime("%Y-%m-%d")
        timestampStr = str(dateTimeObj.timestamp())

        results_dictionary = {}

        results_dictionary["Timestamp"] = timestampStr

        results_dictionary["Sample_name"] = self.instrument.sample_name

        if self.instrument.composition_domain is not None:
            results_dictionary["Domain"] = self.instrument.composition_domain
        # if self.instrument.diffraction_space is not None:
          #  results_dictionary["Diffraction_Space"] = self.instrument.diffraction_space

        results_dictionary["Next_indexes"] = str(self.next_indexes.tolist())

        filename = "Loop_start_"+timestampStr+".json"
        fullfilename = self.save_directory+filename
        with open(fullfilename, "w") as outfile:
            json.dump(results_dictionary, outfile)

    def write_loopdata_file(self, Loop_index:int):
        dateTimeObj = datetime.now()
        # timestampStr = dateTimeObj.strftime("%Y-%m-%d")
        timestampStr = str(dateTimeObj.timestamp())

        results_dictionary = {}

        results_dictionary["Timestamp"] = timestampStr

        results_dictionary["Measured_indexes"] = str(self.measured_indexes.tolist())

        results_dictionary["labels"] = str(self.cluster_method.labels.tolist())
        results_dictionary["probabilities"] = str(self.cluster_method.probabilities.tolist())

        results_dictionary["kernel"] = self.classification_method.model.kernel.name
        results_dictionary["k_lengthscale"] = str(self.classification_method.model.kernel.lengthscales.numpy())
        results_dictionary["k_variance"] = str(self.classification_method.model.kernel.variance.numpy())

        results_dictionary["mean"] = str(self.classification_method.mean.numpy().tolist())
        results_dictionary["var"] = str(self.classification_method.var.numpy().tolist())

        results_dictionary["Next_indexes"] = str(self.next_indexes)

        filename = f"Loop_{Loop_index}"+timestampStr+".json"
        fullfilename = self.save_directory+filename
        with open(fullfilename, "w") as outfile:
            json.dump(results_dictionary, outfile)
