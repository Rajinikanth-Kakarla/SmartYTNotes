from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/bart-large-cnn"  # or another summarization model
AutoModelForSeq2SeqLM.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)
