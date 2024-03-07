from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DataTrainingArguments:
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data directory of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(default=512)