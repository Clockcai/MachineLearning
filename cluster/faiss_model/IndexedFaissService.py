import faiss

# 线程安全的单例模式
import threading
class SingletonType(type):
    _instance_lock = threading.Lock()
    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType,cls).__call__(*args, **kwargs)
        return cls._instance

class IndexedFaiss(metaclass=SingletonType):
    def __init__(self, data, query, nlist):
        self.data = data
        self.query = query
        self.dim = self.data.shape[1]
        self.quantizer = faiss.IndexFlatL2(self.dim)  # 量化器
        self.index = faiss.IndexIVFFlat(self.quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        # METRIC_L2计算L2距离, 或faiss.METRIC_INNER_PRODUCT计算内积
        self.index.train(self.data)  # 训练数据集应该与数据库数据集同分布
        self.index.add(self.data)

    def getSimIndex(self, nprobe, k = 10):
        self.index.nprobe = nprobe  # 选择n个维诺空间进行索引,
        dis, ind = self.index.search(self.query, k)
        return ind

class NormalFaiss(metaclass=SingletonType):
    def __init__(self, data, query):
        self.data = data
        self.query = query
        self.dim = self.data.shape[1]

    def getSimIndex(self, k):
        index = faiss.IndexFlatL2(self.dim)  # 构建index
        print(index.is_trained)  # False时需要train
        index.add(self.data)  # 添加数据
        dis, ind = index.search(self.query, k)
        return ind





