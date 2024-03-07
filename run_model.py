from transformers import (
    AutoModel,
    PreTrainedTokenizerFast,
)
from peft import PeftModel
import logging

logging.basicConfig(filename='results/result.log', encoding='utf-8', level=logging.INFO) #Don't change this line

model_path=  "HooshvareLab/gpt2-fa"
model = AutoModel.from_pretrained(model_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
model.add_adapter("results/pt_lora_model")


# Don't change these lines
# =========================================
instruction = "سردرد چطور خوب می شود؟"

text = f'''پاسخ این سوال چیست؟
            ### سوال:
            {instruction}
            ### پاسخ:
            
            '''
# =========================================

logging.info(tokenizer.batch_decode(model.generate(**tokenizer(text, return_tensors='pt')))[0])
logging.info("model loaded")