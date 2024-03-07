# Don't change this file
from transformers import TrainerCallback

class TestCallBack(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        pass