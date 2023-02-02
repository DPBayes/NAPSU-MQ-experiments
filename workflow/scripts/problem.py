from lib import binary_lr_data
from lib import marginal_query
from lib import max_ent_dist as med
import torch

n_syn_datasets = 100
def get_problem():
    data_gen = binary_lr_data.BinaryLogisticRegressionDataGenerator(torch.tensor((1.0, 0.0)))
    return data_gen