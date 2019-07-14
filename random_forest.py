

import numpy as np
from decision_tree import Decision_tree
from mpi4py import MPI
import numpy as np
import os
from functools import reduce
import time
import json
from cache import Cache
from read_file import read_length
from read_file import read_csv
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
mode = MPI.MODE_WRONLY|MPI.MODE_CREATE|MPI.MODE_APPEND

class Random_forest:
    
    def __init__(self, n_trees=10, n_processes=1, num_max_features=5, max_depth=5, min_samples_split=5):
        self.n_trees = n_trees
        self.n_processes = n_processes
        self.num_max_features = num_max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = [{} for x in range(world_size)]
    
    def fit(self, files_name_x, files_name_y):
        file_name_x = files_name_x[rank%len(files_name_x)]
        file_name_y = files_name_y[rank%len(files_name_y)]
        self.length = read_length(file_name_y)
        _range = world_size // self.n_processes
        if rank==0: # ps
            send_reqs = [comm.Isend([length/(world_size-1)*(x-1),length/(world_size-1)*(x)],dest = x) for x in range(1,world_size)]
            [req.wait() for req in send_reqs]
            [comm.Recv(self.trees[x],source=x) for x in range(1,world_size)]
            for x in range(len(self.trees)):
                self.trees[x] = json.loads(self.trees[x])
        else:
            data_range = []
            comm.Recv(data_range,source=0)
            if self.cache == None:
                self.cache = Cache(file_name_x, file_name_y, length , _range)
            tree = Decision_tree(self.max_features_num, 
                                             self.max_depth, 
                                             self.min_samples_split) 
            tree.build(cache.read_next())
            req = comm.Isend(tree.get_param(),dest = 0)
            req.wait()
    
    def predict(self, files_name_x):
        X = []
        y = []
        for file in files_name_x:
            X = X + read_csv(file)
        
        for tree in self.trees:
            y += tree.predict(X)
        
        y /= self.n_trees
        
        return y