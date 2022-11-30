#@title Define the app's init and inference functions

import os
import sys
import torch as t

from transformers import JukeboxModel , JukeboxTokenizer
from transformers.models.jukebox import convert_jukebox

if not 'google.colab' in sys.modules:

  import download as dl
  cache_path = dl.cache_path

else:

  # Monkey patch convert_jukebox.convert_openai_checkpoint:
  # def convert_openai_checkpoint(model_name=None, pytorch_dump_folder_path=None)
  # to use the cache_path instead of the default download path (pytorch_dump_folder_path)
  # (Avoid patching twice)
  try:
    patched_convert_openai_checkpoint
    # (If the above line doesn't throw an error, then the function has already been patched)
    print("convert_openai_checkpoint already patched")
  except NameError:
    original_convert_openai_checkpoint = convert_jukebox.convert_openai_checkpoint

    def patched_convert_openai_checkpoint(model_name=None, pytorch_dump_folder_path=None):
      print(f"Using cache path: {cache_path}")
      return original_convert_openai_checkpoint(model_name, cache_path)

    convert_jukebox.convert_openai_checkpoint = patched_convert_openai_checkpoint
    print("convert_openai_checkpoint patched")

def tokens_to_seconds(tokens, level = 2):

  global hps, raw_to_tokens
  return tokens * raw_to_tokens / hps.sr / 4 ** (2 - level)

def seconds_to_tokens(sec, level = 2):

  global hps, raw_to_tokens, chunk_size

  tokens = sec * hps.sr // raw_to_tokens
  tokens = ( (tokens // chunk_size) + 1 ) * chunk_size

  # For levels 1 and 0, multiply by 4 and 16 respectively
  tokens *= 4 ** (2 - level)

  return int(tokens)

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
  global model

  print("Loading model...")
  model = JukeboxModel.from_pretrained('openai/jukebox-5b-lyrics', device_map="auto", torch_dtype=t.float16).eval()
  print("Model loaded: ", model)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
  global model

  print(f"Received inputs: {model_inputs}")
  
  sampling_kwargs = dict(
    n_samples = 4,
    temp = 0.98,
    sample_length_in_seconds = 200,
    sample_tokens = seconds_to_tokens(1),
    chunk_size = 32,
    max_batch_size = 16,
    fp16 = True,
  )

  metas = { 
    **model_inputs,
    **dict(
      total_length =  hps.sample_length,
      offset = 0,
    )
  }

  labels = JukeboxTokenizer.from_pretrained('openai/jukebox-5b-lyrics')(**metas)['input_ids']

  zs =  model.ancestral_sample(labels, **sampling_kwargs)

  print(f"Generated {len(zs)} samples: {zs}")

  return zs