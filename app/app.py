import streamlit as st
import torch
import json
import torch.nn.functional as F
import pandas as pd

SEED = 42

@st.cache_resource
def init_count_model():
    return torch.load("count_probs.pt")

@st.cache_resource
def init_single_layer_model():
    return torch.load("single_layer.pt")

@st.cache_resource
def init_mlp():
    mlp_layers = torch.load("mlp.pt")
    return mlp_layers["emb"], mlp_layers['w1'], mlp_layers['b1'], mlp_layers['w2'], mlp_layers['b2']

@st.cache_resource
def init_char_index_mappings():
    with open("ctoi.json") as ci, open("itoc.json") as ic:
        return json.load(ci), json.load(ic)

count_p = init_count_model()
single_layer_w = init_single_layer_model()
mlp_emb, mlp_w1, mlp_b1, mlp_w2, mlp_b2 = init_mlp()
ctoi, itoc = init_char_index_mappings()

def predict_with_count(starting_char:str, num_words):
    g = torch.Generator().manual_seed(SEED)   
    output = []
    for _ in range(num_words):
        if starting_char not in ctoi:
            raise ValueError("Starting Character is not a valid alphabet. Please input a valid alphabet.")
        prev = ctoi[starting_char]
        out = []
        out.append(starting_char)
        while True:
            p = count_p[prev]
            pred = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itoc[str(pred)])
            if pred==0: 
                break           # end if '.' is predicted -> end of word
            prev = pred
        output.append(''.join(out[:-1]))            # discard '.' at the end
    return output

def predict_with_single_layer_nn(starting_char:str, num_words):
    g = torch.Generator().manual_seed(SEED)
    output = []
    for _ in range(num_words):
        out = []
        if starting_char not in ctoi:
            raise ValueError("Starting Character is not a valid alphabet. Please input a valid alphabet.")
        ix = ctoi[starting_char]
        out.append(starting_char)
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = xenc @ single_layer_w 
            counts = logits.exp() 
            probs = counts/counts.sum(1, keepdim=True)
            
            ix = torch.multinomial(probs, generator=g, replacement=True, num_samples=1).item()
            out.append(itoc[str(ix)])
            if ix==0:
                break
        output.append(''.join(out[:-1]))
    return output

def predict_with_mlp(starting_char:str, num_words):
    g = torch.Generator().manual_seed(SEED)
    output = []
    context_length = 3
    for _ in range(num_words):
        out = []
        context = [0]*(context_length-1)
        if starting_char not in ctoi:
            raise ValueError("Starting Character is not a valid alphabet. Please input a valid alphabet.")
        ix = ctoi[starting_char]
        out.append(starting_char)
        context+=[ix]
        while True:
            emb = mlp_emb[torch.tensor([context])]
            h = torch.tanh(emb.view(1,-1) @ mlp_w1 + mlp_b1)        # create batch_size 1
            logits = h @ mlp_w2 + mlp_b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(itoc[str(ix)])
            if ix == 0:
                break
        output.append(''.join(out[:-1]))
    return output

def predict(query, num_words):
    try:
        preds = [predict_with_count(query, num_words), predict_with_single_layer_nn(query, num_words), predict_with_mlp(query, num_words)]
        labels = ["Count Based LM", "Single Linear Layer LM", "Embedding Based Single Hidden Layer LM"]
        results = {labels[idx]: preds[idx] for idx in range(len(preds))}
        st.write(pd.DataFrame(results, index=range(num_words)))
    except ValueError as e:
        st.write(f"ERROR: {e.args[0]}")

# title and description
st.title("""
Make More Names.
         
This app creates the requested number of names starting with the input character below. The results will be predicted from the basic count based to advanced transformer based Character Level Language Model.""")

# search bar
query = st.text_input("Please input the starting character...", "", max_chars=1)

# number of words slider
num_words = st.slider("Number of names to generate:", min_value=1, max_value=50, value=5)

if query != "":
    predict(query.lower(), num_words)
