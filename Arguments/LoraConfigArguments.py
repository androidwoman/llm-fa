from typing import Optional
from dataclasses import dataclass, field
from peft import TaskType

@dataclass
class LoraArguments:
    r: int = field()
    lora_alpha: int = field()
    lora_dropout: float = field()
    task_type: str = field( default=TaskType.CAUSAL_LM)
    
