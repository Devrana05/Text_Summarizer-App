# Text Summarizer

A dialogue/text summarization web app powered by a fine-tuned **T5-small** model, served through a **FastAPI** backend with a simple HTML/JS frontend.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-teal)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![T5](https://img.shields.io/badge/Model-T5--small-orange)

## Overview

This project fine-tunes a pre-trained `t5-small` model on the **SAMSum** dialogue summarization dataset to generate concise summaries of conversational text. The fine-tuned model is served via a FastAPI backend, with a lightweight web UI for users to paste text and get an instant summary.

## Project Structure

```
.
├── text_summarizer.ipynb   # Data preprocessing, tokenization, and model fine-tuning
├── app.py                  # FastAPI backend serving the summarization model
├── index.html              # Frontend UI for interacting with the API
└── saved_summary_model/    # Fine-tuned model + tokenizer (generated after training)
```

## How It Works

### 1. Training (`text_summarizer.ipynb`)

- **Dataset:** [SAMSum](https://huggingface.co/datasets/samsum) dialogue-summary pairs (`samsum-train.csv`, `samsum-validation.csv`).
- **Sampling:** A subset of 4,000 training examples and 500 validation examples is used (random seed 42) for faster experimentation.
- **Preprocessing:** A `clean_data` function normalizes text by:
  - Replacing line breaks and extra whitespace
  - Stripping HTML tags
  - Lowercasing and trimming the text
- **Tokenization:** Uses `T5Tokenizer` (`t5-small` vocabulary, ~32k tokens). Dialogues are tokenized to a max length of 512, summaries (labels) to a max length of 150, both with padding and truncation.
- **Model:** `T5ForConditionalGeneration` (`t5-small`) is fine-tuned using HuggingFace's `Trainer` API with the following key settings:
  - 6 training epochs
  - Batch size of 8 (train & eval)
  - Weight decay: 0.01
  - Warmup steps: 500
  - Evaluation & checkpointing after every epoch
- **Output:** The fine-tuned model and tokenizer are saved to `./saved_summary_model`.

### 2. Inference API (`app.py`)

- Loads the fine-tuned model and tokenizer from `./saved_summary_model`.
- Automatically selects the best available device: **MPS** (Apple Silicon) → **CUDA** → **CPU**.
- Applies the same text-cleaning logic used during training before generating a summary.
- Uses beam search (`num_beams=4`, `early_stopping=True`) to generate summaries up to 150 tokens long.

**Endpoints:**

| Method | Endpoint      | Description                                  |
|--------|---------------|-----------------------------------------------|
| GET    | `/`           | Serves the web UI (`index.html`)              |
| POST   | `/summarize/` | Accepts `{"dialogue": "<text>"}` and returns `{"summary": "<text>"}` |

### 3. Frontend (`index.html`)

A minimal, styled single-page UI where users can:
- Paste or type text into a textarea
- Click **Summarize** to send the text to `/summarize/`
- View the generated summary rendered on the page, with basic loading/error handling

## Setup & Installation

### Prerequisites

- Python 3.9+
- A fine-tuned model available at `./saved_summary_model` (produced by running `text_summarizer.ipynb`, or your own fine-tuned T5 checkpoint)

### Install dependencies

```bash
pip install fastapi uvicorn transformers torch pandas jinja2 python-multipart
```

### Run the training notebook (optional, if you don't already have a saved model)

1. Download the SAMSum dataset as `samsum-train.csv` and `samsum-validation.csv`.
2. Open and run `text_summarizer.ipynb` end-to-end.
3. This produces the `./saved_summary_model` directory containing the fine-tuned weights and tokenizer.

### Run the app

```bash
uvicorn app:app --reload
```

Then open your browser at:

```
http://127.0.0.1:8000
```

## Usage

1. Paste any dialogue or block of text into the textarea.
2. Click **Summarize**.
3. The generated summary appears in the output panel below the form.

You can also call the API directly:

```bash
curl -X POST "http://127.0.0.1:8000/summarize/" \
  -H "Content-Type: application/json" \
  -d '{"dialogue": "Your text here..."}'
```

Response:

```json
{
  "summary": "Generated summary text."
}
```

## Tech Stack

- **Model:** T5-small (fine-tuned) via HuggingFace `transformers`
- **Training:** PyTorch + HuggingFace `Trainer`
- **Backend:** FastAPI, Jinja2 templating
- **Frontend:** Vanilla HTML/CSS/JavaScript

## Notes & Possible Improvements

- Training was done on a sampled subset (4,000/500 examples) for speed; using the full SAMSum dataset would likely improve summary quality.
- `clean_data` lowercases all text, which the model then also generates in — consider preserving case if that's not desired for final output.
- Add input length validation and rate limiting for production use.
- Add a `requirements.txt` and `Dockerfile` for easier deployment.
