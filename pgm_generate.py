'''
Generate samples for NBN training.
'''

from pgmpy.sampling.Sampling import BayesianModelSampling
from pgmpy.readwrite.BIF import BIFReader
import pickle

num_samples=100000
bif='dataset/insurance.bif'
out='dataset/insurance.dat'

def generate():
    reader=BIFReader(bif)
    model=reader.get_model()
    infer=BayesianModelSampling(model)
    data=infer.forward_sample(num_samples)
    pickle.dump(data.to_dict(orient='records'),open(out,'wb'))

if __name__ == '__main__':
    generate()
