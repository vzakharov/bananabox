import torch as t

from jukebox.make_models import make_vqvae, make_prior, MODELS
from jukebox.hparams import Hyperparams, setup_hparams, REMOTE_PREFIX
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.remote_utils import download
from jukebox.utils.sample_utils import get_starts
from jukebox.utils.torch_utils import empty_cache
from jukebox.sample import sample_partial_window, load_prompts, upsample, sample_single_window

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global top_prior, priors, vqvae, hps, raw_to_tokens, chunk_size, lower_batch_size, lower_level_chunk_size
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

    rank, local_rank, device = setup_dist_from_mpi(backend="gloo")
    # (Using gloo because windows doesn't support nccl)
    print(f"Rank: {rank}, Local Rank: {local_rank}, Device: {device}")


    vqvae, *priors = MODELS['5b_lyrics']

    vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = hps.sample_length)), device)
    print(f"VQVAE: {vqvae}")

    top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)   
    print(f"Top Prior: {top_prior}")


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
    zs = model.ancestral_sample(conditioning, chunk_size = 32, sample_length_in_seconds=1, sample_levels=[0])
    print(f"Generated {len(zs)} samples: {zs}")

    return zs