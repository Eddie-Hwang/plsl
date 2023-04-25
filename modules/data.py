from torch.utils.data import Dataset


# Create your own dataset
class DummyData(Dataset):
    def __init__(self, file_path):
        super().__init__()

        self.file_path = file_path

    def __getitem__(self, index):
        return
    
    def __len__(self):
        return
    

def custom_collate(batch):
    return {}