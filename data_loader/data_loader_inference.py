import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
from os.path import join, exists
from scipy import sparse
from torch.utils.data import Dataset
from base import BaseDataLoader

class DDIGraphInferenceDataLoader():
    def __init__(self, logger, data_dir):
        pass