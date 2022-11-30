#@title Define the app's init and inference functions

import os
import sys
import torch as t

import jukebox
import jukebox.utils.dist_adapter as dist

from jukebox.make_models import make_vqvae, make_prior, MODELS
from jukebox.hparams import Hyperparams, setup_hparams, REMOTE_PREFIX
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.remote_utils import download
from jukebox.utils.sample_utils import get_starts
from jukebox.utils.torch_utils import empty_cache
from jukebox.sample import sample_partial_window, load_prompts, upsample, sample_single_window

if not 'google.colab' in sys.modules:
  import download as dl
  cache_path = dl.cache_path

def monkey_patched_load_checkpoint(path):
  global cache_path
  restore = path
  if restore.startswith(REMOTE_PREFIX):
      remote_path = restore
      local_path = os.path.join(cache_path, remote_path[len(REMOTE_PREFIX):])
      # Assert that the file exists
      assert os.path.exists(local_path), f"File {local_path} does not exist. If running in Colab, first run the code from the download.py file to download the model."
      restore = local_path
  dist.barrier()
  checkpoint = t.load(restore, map_location=t.device('cpu'))
  print("Restored from {}".format(restore))
  return checkpoint

jukebox.make_models.load_checkpoint = monkey_patched_load_checkpoint
# (We monkey-patch the load_checkpoint function to use the local model files instead of having to download them from the internet)

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
  global model, priors, vqvae, hps, raw_to_tokens, chunk_size, lower_batch_size, lower_level_chunk_size, rank, local_rank, device
  # (We use `model` instead of top_prior because Banana will be looking for a global variable named `model` to optimize its cold start time)
  # TODO: Think of how to combine this with the upsampling priors (create another Banana model for upsampling?)

  print("Loading model...")

  total_duration = 200

  raw_to_tokens = 128
  chunk_size = 16
  lower_batch_size = 16
  lower_level_chunk_size = 32

  hps = Hyperparams()
  hps.sr = 44100
  hps.levels = 3
  hps.hop_fraction = [ 0.5, 0.5, 0.125 ]
  hps.sample_length = seconds_to_tokens(total_duration)

  try:
    rank, local_rank, device
  except NameError:
    rank, local_rank, device = setup_dist_from_mpi()
  # (So we don't have to run the code twice if re-running the cell)

  print(f"Rank: {rank}, Local Rank: {local_rank}, Device: {device}")

  try:
    vqvae, priors, model
  except NameError:
    vqvae, *priors = MODELS['5b_lyrics']
    
    print("Loading VQVAE...")
    vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)
    print("VQVAE loaded")

    print("Loading top prior...")
    model = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)   
    print("Top prior loaded")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
  global model, priors, vqvae, hps, raw_to_tokens, chunk_size, lower_batch_size, lower_level_chunk_size, rank, local_rank, device, zs, wavs

  print(f"Received inputs: {model_inputs}")
  
  n_samples = 4
  temperature = 0.98
  seconds_to_sample = 1

  metas = [{ 
    **model_inputs,
    **dict(
      total_length =  hps.sample_length,
      offset = 0,
    )
  }] * n_samples

  hps.n_samples = n_samples

  labels = model.labeller.get_batch_labels(metas, device)

  sampling_kwargs = dict(
    temp=temperature, fp16=True, max_batch_size=lower_batch_size,
    chunk_size=lower_level_chunk_size
  )

  tokens_to_sample = seconds_to_tokens(seconds_to_sample)

  zs = [ t.zeros(n_samples, 0, dtype=t.long, device='cuda') for _ in range(3) ]
  zs = sample_partial_window(zs, labels, sampling_kwargs, 2, model, tokens_to_sample, hps)

  print(f"Generated {len(zs)} samples: {zs}")

  wavs = vqvae.decode(zs[2:], start_level=2).cpu().numpy()

  return wavs