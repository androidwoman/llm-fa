from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_path: str = field()