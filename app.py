from transformers import JukeboxModel, JukeboxTokenizer
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model, tokenizer
    
    model_id = 'openai/jukebox-1b-lyrics'
    print(f"Loading model {model_id}")
    model = JukeboxModel.from_pretrained(model_id, device_map="auto", torch_dtype = torch.float16).eval()
    print("Model loaded")
    tokenizer = JukeboxTokenizer.from_pretrained(model_id)
    print("Tokenizer loaded")

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