import os
import lmdb
from torch.utils.data import Dataset
import pickle


class DatasetForLMBD(Dataset):
    def __init__(self, db_path_list):
        super().__init__()
        
        self.dbs = db_path_list
        self.envs = [self._init_db(db_path) for db_path in self.dbs]
        
    def _init_db(self, db_path):
        env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
            readonly=True, lock=False,
            readahead=False, meminit=False)
        return env

    def read_lmdb(self, txn, key):
        data = txn.get(key.encode("utf-8"))
        data = pickle.loads(data)
        return data
    
    def _get_env_and_index(self, index):
        # determine which dataset the index corresponds to
        for env in self.envs:
            if index < env.stat()["entries"]:
                return env, index
            else:
                index -= env.stat()["entries"]
        raise IndexError("Index out of range")
    
    def __len__(self):
        # the total length is the sum of the lengths of all datasets
        return sum(env.stat()["entries"] for env in self.envs)
    
    def __getitem__(self, index):
        env, index = self._get_env_and_index(index)
        txn = env.begin()
        data = self.read_lmdb(txn, str(index))
        
        return {
            "id": data["id"],
            "text": data["text"],
            "gloss": data["gloss"],
            "feature": data["feature"]
        }


def custom_collate(batch):
    id_list, text_list, gloss_list, feature_list = [], [], [], []
    for elem in batch:
        id_list.append(elem["id"])
        text_list.append(elem["text"])
        gloss_list.append(elem["gloss"])
        feature_list.append(elem["feature"].cpu())

    return {
        "id": id_list,
        "text": text_list,
        "gloss": gloss_list,
        "feature": feature_list,
    }