import torch
import importlib
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from tqdm import tqdm
from fractions import Fraction
import pandas as pd
from music21 import *

import functions as f

importlib.reload(f)
from music_transformer import MusicTransformer

seq_len = 64
batch_size = 256

def load_model(n_part):
    # Construire le chemin correctement
    file_path = os.path.join("model", f"results_opt_{n_part}p.csv")

    # Lire le CSV
    df = pd.read_csv(file_path, sep=';')

    print("opt bon")
    
    # Select the row with the highest test accuracy
    best_row = df.loc[df['test_accuracy'].idxmax()]

    checkpoint_path = os.path.join("model", f"music_transformer_{n_part}p.pth")

    # Charger le modèle
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("modèle bon")
    # Access the vocabulary
    token_to_id = checkpoint["token_to_id"]
    id_to_token = checkpoint["id_to_token"]

    # Recreate the model with the correct dimensions
    model = MusicTransformer(
        vocab_size=len(token_to_id),
        num_layers=int(best_row["num_layers"]),
        d_model=int(best_row["d_model"]),
        d_ff=int(best_row["d_ff"]),
        nhead=int(best_row["nhead"]),
        dropout=float(best_row["dropout"]),
        max_len=seq_len
    )

    # Load the trained weights into the model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(torch.device("cpu"))
    model.eval()

    return model, token_to_id, id_to_token

def extract_key_and_nhand_from_score(filepath):
    score = converter.parse(filepath)

    armures = []
    for part in score.parts:
        for ks in part.recurse().getElementsByClass(key.KeySignature):
            armures.append(ks)

    score_key = armures[0].asKey()
    nhand = len(score.parts)

    return score_key, nhand

def load_music_sheet(filpeath, npart):    
    tokens = f.extract_events_from_score(filpeath, n_hand=npart)
    tokens = f.merge_events_simple(tokens)
    tokens = f.events_to_tokens2(tokens)
    tokens.append("barline")
    return tokens

def filter_tokens(tokens):
    # Find the last two barlines and extract the relevant segment
    start = [i for i, t in enumerate(tokens) if t == "barline"][-2]
    segment = tokens[start:]
    #print(segment)

    # Last time signature (e.g., "time_4/4")
    last_time_signature = [t for t in tokens if t.startswith("time")][-1]
    
    # Extract the maximum allowed duration from the time signature
    sig = last_time_signature.split("_")[-1]  
    max_time = f.parse_time_signature(sig)

    total_time = 0
    filtered_segment = []

    for i, tok in enumerate(segment):
        if tok.startswith("duration"):
            dur_str = tok.split("_")[1]
            dur = float(Fraction(dur_str))
            total_time += dur

            # If the total time exceeds the max → remove this duration and the previous token
            if total_time > max_time:
                if filtered_segment:
                    filtered_segment.pop()  # remove the previous token
                total_time -= dur          # undo the duration
                continue  # skip adding this duration
        
        if tok.startswith("main"):
            total_time = 0

        filtered_segment.append(tok)

    return tokens[:start] + filtered_segment

def sample_top_k(logits, k):
    # Keep only the top k most probable tokens
    topk_logits, topk_indices = torch.topk(logits, k)

    # Apply softmax to convert logits into probabilities
    probs = functional.softmax(topk_logits, dim=-1)

    # Sample randomly according to the probability distribution
    choice = torch.multinomial(probs, 1).item()
    return topk_indices[choice].item()

def generation(tokens, n_barline, model, token_to_id, id_to_token, k):
    i = 0
    print("test 6 : dans la fonction génération")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("test 7 : modèle ok")
    seed = [token_to_id[tok] for tok in tokens]  # Convert seed tokens to IDs
    
    while i < n_barline:
        # Prepare input: take the last seq_len tokens
        x = torch.tensor([seed[-seq_len:]]).to(device)
        with torch.no_grad():
            logits = model(x)[0, -1]  # Predictions for the last time step
        print("test 8 : tout est prêt")
        
        # Apply top-k sampling instead of argmax
        next_token_id = sample_top_k(logits, k)
        seed.append(next_token_id)  # Add the sampled token ID
        print("print 9 : un token de plus")
        
        # If the last token is a barline, filter the sequence
        if [id_to_token[i] for i in seed][-1] == 'barline':
            filtered_tokens = filter_tokens([id_to_token[i] for i in seed])
            seed = [token_to_id[t] for t in filtered_tokens]
            i += 1
    print("test 10 : fini")
    # Convert token IDs back to readable tokens
    generated_tokens = [id_to_token[i] for i in seed]
    #print(len(generated_tokens))
    return generated_tokens

def write_music_sheet(generated_tokens, npart, key_score, filepath):
    
    importlib.reload(f)

    events_restored = f.tokens_to_events2(generated_tokens)
    score = f.decode_to_score(events_restored, npart)

    i = interval.Interval(key_score.tonic, pitch.Pitch('C'))

    score = score.transpose(i.reverse())

    k = key.KeySignature(sharps=key_score.sharps)
  
    for part in score.parts:
        part.insert(0, k)
    
    # Make sure output folder exists
    output_dir = r"uploads"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.basename(filepath)                 
    name_no_ext = os.path.splitext(base_name)[0]                
    safe_name = f"{name_no_ext}"                 
    output_path = os.path.join(output_dir, safe_name)

    score.write('mxl', fp=output_path)

    print(f"File written to: {output_path}")




