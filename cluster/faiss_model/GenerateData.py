import numpy as np
from src.service.SingletonType import SingletonType


class DataGenerate(metaclass=SingletonType):
    def __init__(self):
        self.data, self.query = generateData(256, 2000000,10)
    def getData(self):
        return self.data, self.query

def generateData(dim, n_data, n_query):
    np.random.seed(0)
    data = []
    mu = 3
    sigma = 10
    for i in range(n_data):
        data.append(np.random.normal(mu, sigma, dim))
    data = np.array(data).astype('float32')
    query = []

    np.random.seed(12)
    for i in range(n_query):
        query.append(np.random.normal(mu, sigma, dim))
    query = np.array(query).astype('float32')
    return data, query
