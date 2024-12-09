from transformers import pipeline
mps_device = "mps"
pipe = pipeline('zero-shot-classification', device = mps_device)
seq = "i love watching the office show"
labels = ['negative', 'positive']
pipe(seq, labels)