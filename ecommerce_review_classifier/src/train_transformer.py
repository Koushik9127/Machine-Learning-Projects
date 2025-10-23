
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

model_name = "bert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

ds = load_dataset("amazon_polarity", split="train[:4%]").train_test_split(test_size=0.2)

def tok(batch):
    return tokenizer(batch["content"], truncation=True, padding="max_length", max_length=128)

ds = ds.map(tok, batched=True).rename_column("label", "labels")
ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

args = TrainingArguments(
    output_dir="../models/bert",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"])
trainer.train()
trainer.save_model("../models/bert_finetuned")

