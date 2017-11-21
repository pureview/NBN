from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianModel

def generate_gen_dataset(bif_path,num):
    reader=BIFReader(bif_path)
    model=reader.get_model()
    sampler=BayesianModelSampling(model)
    return sampler.forward_sample(num)

