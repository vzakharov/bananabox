#@title Define the app's init and inference functions

import os
import sys
import torch as t

from transformers import JukeboxModel , JukeboxTokenizer
from transformers.models.jukebox import convert_jukebox

model_id = 'openai/jukebox-1b-lyrics' #@param ['openai/jukebox-1b-lyrics', 'openai/jukebox-5b-lyrics']
sample_rate = 44100
total_duration_in_seconds = 200
raw_to_tokens = 128
chunk_size = 32
max_batch_size = 16

if 'google.colab' in sys.modules:

  cache_path = '/content/drive/My Drive/jukebox-webui/_data/' #@param {type:"string"}
  # Connect to your Google Drive
  from google.colab import drive
  drive.mount('/content/drive')

else:

  cache_path = '~/.cache/'

def tokens_to_seconds(tokens, level = 2):

  global sample_rate, raw_to_tokens
  return tokens * raw_to_tokens / sample_rate / 4 ** (2 - level)

def seconds_to_tokens(sec, level = 2):

  global sample_rate, raw_to_tokens, chunk_size

  tokens = sec * sample_rate // raw_to_tokens
  tokens = ( (tokens // chunk_size) + 1 ) * chunk_size

  # For levels 1 and 0, multiply by 4 and 16 respectively
  tokens *= 4 ** (2 - level)

  return int(tokens)

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
  global model

  print(f"Loading model from/to {cache_path}...")
  model = JukeboxModel.from_pretrained(
    model_id,
    device_map = "auto",
    torch_dtype = t.float16,
    cache_dir = f"{cache_path}/jukebox/models",
    resume_download = True,
    min_duration = 0
  ).eval()
  print("Model loaded: ", model)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
  global model

  print(f"Received inputs: {model_inputs}")
  
  n_samples = 4
  generation_length = seconds_to_tokens(1)
  offset = 0
  level = 0

  model.total_length = seconds_to_tokens(total_duration_in_seconds)

  sampling_kwargs = dict(
    temp = 0.98,
    chunk_size = chunk_size,
  )

  metas = model_inputs

  labels = JukeboxTokenizer.from_pretrained(model_id)(**metas)['input_ids'][level].cuda().repeat(n_samples, 1)

  zs = [ t.zeros(n_samples, 0, dtype=t.long, device='cuda') for _ in range(3) ]

  zs = model.sample_partial_window(
    zs, labels, offset, sampling_kwargs, level = level, tokens_to_sample = generation_length, max_batch_size = max_batch_size
  )

  print(f"Generated {len(zs)} samples: {zs}")

  return zs