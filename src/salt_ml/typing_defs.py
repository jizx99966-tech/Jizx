from dataclasses import dataclass


@dataclass
class SplitResult:
    train_rows: int
    test_rows: int
    feature_count: int
