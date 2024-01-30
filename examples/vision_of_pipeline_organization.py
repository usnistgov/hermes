class Pipeline():
	instrument: hermes.instrument = 
	domain = 
	

class ClusteringClassification(Pipeline):
	clustering_method = 
	classification_method = 
	archive_method = 
	
class AL_Pipeline(Pipeline)
	initialization_method = 
	stopping_criteria = 
	acquistion_method = 
	
class PhaseMappingPipeline(AL_Pipeline):
	clustering_method: contigous_clustering = RB_potts
	classification_method = 

	archive_method = 

class JointPhaseMappingPipeline(PhaseMappingPipeline):
	regression_method = 
	
#Using default data analysis methods
mypipeline = PhaseMappingPipeline()
mypipeline.instrument = hermes.instrument.QM2?????
mypipeline.domain = mypipeline.instrument.xy_locations

#Re-setting the clustering step 
mypipeline.clustering_method = hermes.clustering.SpectralClustering()