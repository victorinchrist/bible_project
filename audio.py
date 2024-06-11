import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import streamlit as st

# select gpu if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# torch double type
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model
model_id = "openai/whisper-large-v3"

# process model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# process processor
processor = AutoProcessor.from_pretrained(model_id)

# create pipe
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)
