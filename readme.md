# Translate text with LLM

This is a simple app that translates text using a language model through Ollama.

## Installation

```bash
pip install srt requests
```

Needs Ollama to be installed and running with a model (the one used here is `gemma4:26b`).

```bash
ollama pull gemma4:26b
```

## Example

```bash
python <PATH_TO>/translate_srt.py
    film.en.srt film.fr.srt \
    --source English --target French \
    --model "gemma4:26b" --limit 40
```

```bash
python <PATH_TO>/translate_full.py
    <PATH_TO_SUBTITLES> \
    --source English --target French \
    --model "gemma4:26b" --limit 40
```