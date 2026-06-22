from abc import ABC, abstractmethod
import pickle

class BaseProcessor(ABC):
    def __init__(self, base_path, train):
        self.base_path = base_path
        self.train = train
        self.all_dfs = []
        self.all_netobj2pro = {}  # object UUID → attributestring
        self.all_subject2pro = {}  # process UUID → attributestring
        self.all_file2pro = {}  # file UUID → attributestring
        self.all_labels = []
        self.total_loaded_bytes = 0
        ###########
        self.begin  = []
        self.malicious = []
        self.snapshots = []
        self.benign_idx_start = 0
        self.benign_idx_end = 0
        self.malicious_idx_start = 0
        self.malicious_idx_end = 0


    @abstractmethod
    def load(self):
        """loadoriginaldata, returnprocessgood's  DataFrame andattributemappingdict"""
        pass

    @abstractmethod
    def create_snapshots_from_graph(self, df, is_malicious):
        pass


    def build_graph(self, gid=None):
        self.snapshots = []
        print("===============buildbenign graphdetect=============")
        self.benign_idx_start = len(self.snapshots)
        benign_snaps = self.create_snapshots_from_graph(self.begin, is_malicious=False)
        self.snapshots.extend(benign_snaps)
        self.benign_idx_end = len(self.snapshots) - 1 if benign_snaps else -1

        print("===============buildmalicious graphdetect=============")
        self.malicious_idx_start = len(self.snapshots)
        mal_snaps = self.create_snapshots_from_graph(self.malicious, is_malicious=True)
        self.snapshots.extend(mal_snaps)
        self.malicious_idx_end = len(self.snapshots) - 1 if mal_snaps else -1

        print(f"totalgenerate {len(self.snapshots)} snapshot")
        print(f"benignsnapshotindexrange: {self.benign_idx_start} to {self.benign_idx_end}")
        print(f"malicioussnapshotindexrange: {self.malicious_idx_start} to {self.malicious_idx_end}")

        # outputfile: ifidentity gid, thenconcatenatetofile
        report_file = f"all_snapshots_{gid}.txt" if gid else "all_snapshots.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            for i, g in enumerate(self.snapshots):
                f.write(f"Community {i}:\n")
                for v in g. vs :
                    attrs = v.attributes()
                    attr_str = ", ".join([f"{k}={v[k]}" for k in attrs])
                    f.write(f"  Vertex {v.index}: {attr_str}\n")
                f.write("\n")
            print(f"{report_file} write completed ")

        snapshot_data = {
            'all_snapshots': self.snapshots,
            'benign_idx_start': self.benign_idx_start,
            'benign_idx_end': self.benign_idx_end,
            'malicious_idx_start': self.malicious_idx_start,
            'malicious_idx_end': self.malicious_idx_end,
        }
        snapshot_file = f"snapshot_data_{gid}.pkl" if gid else "snapshot_data.pkl"
        with open(snapshot_file, 'wb') as f:
            pickle.dump(snapshot_data, f)
        print(f"snapshotdatasavedto: {snapshot_file}")
