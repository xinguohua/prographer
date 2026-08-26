from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    def __init__(self, base_path, train):
        self.base_path = base_path
        self.train = train
        self.all_dfs = []
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
        """Load and normalize the source audit events."""
        pass

    @abstractmethod
    def create_snapshots_from_graph(self, df, is_malicious):
        pass


    def build_graph(self, gid=None):
        self.snapshots = []
        print("[snapshot] building benign windows")
        self.benign_idx_start = len(self.snapshots)
        benign_snaps = self.create_snapshots_from_graph(self.begin, is_malicious=False)
        self.snapshots.extend(benign_snaps)
        self.benign_idx_end = len(self.snapshots) - 1 if benign_snaps else -1

        print("[snapshot] building attack windows")
        self.malicious_idx_start = len(self.snapshots)
        mal_snaps = self.create_snapshots_from_graph(self.malicious, is_malicious=True)
        self.snapshots.extend(mal_snaps)
        self.malicious_idx_end = len(self.snapshots) - 1 if mal_snaps else -1

        print(f"[snapshot] total={len(self.snapshots)}")
        print(f"[snapshot] benign_index_range={self.benign_idx_start}..{self.benign_idx_end}")
        print(f"[snapshot] attack_index_range={self.malicious_idx_start}..{self.malicious_idx_end}")
