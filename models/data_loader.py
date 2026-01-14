# data_loader.py
import torch
import pandas as pd
import json
import os

class AtravelDataset:
    def __init__(self, base_path):
        self.base_path = base_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open(os.path.join(base_path, 'mappings', 'individual_to_id.json'), 'r') as f:
            self.entity_to_id = json.load(f)
        with open(os.path.join(base_path, 'mappings', 'object_property_to_id.json'), 'r') as f:
            self.rel_to_id = json.load(f)
            
        self.num_nodes = len(self.entity_to_id)
        self.num_relations = len(self.rel_to_id)

    def _load_tsv(self, filename):
        path = os.path.join(self.base_path, 'abox', 'splits', filename)
        df = pd.read_csv(path, sep='\t', names=['s', 'r', 'o'])
        
        s = torch.tensor([self.entity_to_id[val] for val in df['s']], dtype=torch.long)
        r = torch.tensor([self.rel_to_id[val] for val in df['r']], dtype=torch.long)
        o = torch.tensor([self.entity_to_id[val] for val in df['o']], dtype=torch.long)
        
        edge_index = torch.stack([s, o], dim=0)
        return edge_index.to(self.device), r.to(self.device)

    def get_splits(self):
        train_idx, train_rel = self._load_tsv("train.tsv")
        val_idx, val_rel = self._load_tsv("valid.tsv")
        test_idx, test_rel = self._load_tsv("test.tsv")
        
        return {
            'train': (train_idx, train_rel),
            'valid': (val_idx, val_rel),
            'test': (test_idx, test_rel)
        }
    
    def get_statistics(self):
            train_edges, train_types = self._load_tsv("train.tsv")
            val_edges, val_types = self._load_tsv("valid.tsv")
            test_edges, test_types = self._load_tsv("test.tsv")

            stats = {
                "Entita'": [
                    torch.unique(train_edges).size(0),
                    torch.unique(val_edges).size(0),
                    torch.unique(test_edges).size(0)
                ],
                "Relazioni": [
                    torch.unique(train_types).size(0),
                    torch.unique(val_types).size(0),
                    torch.unique(test_types).size(0)
                ],
                "Triple (Edges)": [
                    train_edges.size(1),
                    val_edges.size(1),
                    test_edges.size(1)
                ]
            }
            return stats