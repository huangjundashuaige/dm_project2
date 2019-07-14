import os
import pickle
from read_file import read_csv

class Cache:
    def __init__(self,file_name_x, file_name_y, length, range):
        self.cache = []  # read in in a order
        self.file_name_x = file_name_x
        self.file_name_y = file_name_y
        self.length = length
        self.range = []
        self.flag = 0
        self.result = []
    def read_next(self):
        self.result = self.cache
        self.cache = [read_csv(file_name_x,[self.flag,(self.flag+self.range)%self.length]),
                        read_csv(file_name_y,[self.flag,(self.flag+self.range)%self.length])]
        return self.result
    