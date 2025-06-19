from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

def load_model():
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    return model, tokenizer
