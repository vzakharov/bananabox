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

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global top_prior, priors, vqvae, hps, raw_to_tokens, chunk_size, lower_batch_size, lower_level_chunk_size, rank, local_rank, device
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
    hps.sample_length = int(total_duration * hps.sr // raw_to_tokens) * raw_to_tokens

    try:
      rank, local_rank, device
    except NameError:
      rank, local_rank, device = setup_dist_from_mpi()
    # (So we don't have to run the code twice if re-running the cell)

    print(f"Rank: {rank}, Local Rank: {local_rank}, Device: {device}")

    try:
      vqvae, priors, top_prior
    except NameError:
      vqvae, *priors = MODELS['5b_lyrics']
      
      print("Loading VQVAE...")
      vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)
      print("VQVAE loaded")

      print("Loading top prior...")
      top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)   
      print("Top prior loaded")


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    print(f"Received inputs: {model_inputs}")
    # Parse out your arguments
    metas = model_inputs
    tokens = tokenizer(**metas)['input_ids']
    conditioning = [ i.cuda() for i in tokens ]
    print("Conditioning loaded; generating...")
    
    # Run the model
    zs = sample_partial_window(top_prior, priors, vqvae, hps, conditioning, chunk_size, lower_batch_size, lower_level_chunk_size)
    # TODO: This is not working code, just a placeholder
    print(f"Generated {len(zs)} samples: {zs}")

    return zs
  
# If testing with Colab, define a user_src object which has attributes for init and inference
# Check if imported modules include google.colab
if 'google.colab' in sys.modules:
  class UserSrc:
    def __init__(self):
      self.init = init
      self.inference = inference

  user_src = UserSrc()