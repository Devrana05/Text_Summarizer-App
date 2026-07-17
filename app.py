from fastapi import FastAPI, Request
from pydantic import BaseModel           
# to validate the input data(whether the input is in correct format or not)

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch 
import re

from fastapi.templating import Jinja2Templates   #UI part 
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize our fastapi app

app = FastAPI(title = "Text Summarizer App", description = "Text Summarization using T5", version = "1.0")

# model & tokenizer initialization

model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device 

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# Templating

templates = Jinja2Templates(directory=".")

# Input Schema (format)

class DialogueInput(BaseModel):
    dialogue: str

# Data Cleaning

def clean_data(text):
    text = re.sub(r"\r\n"," ", text)  # line replacement
    text = re.sub(r"\s+"," ", text)  # empty space replacement
    text = re.sub(r"<.*?>"," ", text)  # HTML tag replacement
    text = text.strip().lower()

    return text


# Summarization Function

def summarize_dialogue(dialogue : str) -> str:

  # Data Preprocessing

    dialogue = clean_data(dialogue)

  # generate tokens of the dialogue
    inputs = tokenizer(
        dialogue,
        padding = "max_length",
        max_length = 512,
        truncation = True,
        return_tensors = "pt"
    ).to(device)

    # generate summary -> which will be in tokens
    model.to(device)
    targets = model.generate(
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4,   # transformer will generate 4 diff. o/p and give us the best one among those 4 o/p.
        early_stopping = True
    )



    # tokens id convertion into text => decoding

    summary = tokenizer.decode(targets[0], skip_special_tokens = True)

    return summary

# API Endpoints 
# 1. Get ( / ) -> to get something from the server 
# 2. Post ( /summarize )  -> to send something to the server

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary": summary}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")