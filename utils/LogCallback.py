from transformers import TrainerCallback
import logging


class LogCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, logs=None, **kwargs):
        logging.info("Training Start: "+ str(state.global_step)) #Don't change this line

    
    def on_train_end(self, args, state, control, logs=None, **kwargs):
        logging.info("Training Totla flos: "+ str(state.total_flos)) #Don't change this line

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step > 50:
            logging.info("Trainer is running at step: "+ str(state.global_step)) #Don't change this line