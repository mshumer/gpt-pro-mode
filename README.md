# gpt-pro-mode

[![Twitter Follow](https://img.shields.io/twitter/follow/mattshumer_?style=social)](https://x.com/mattshumer_)

gpt-oss-pro-mode: [![Open Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XeYmOHJwACtavCjJM-eOqlPxHgTD2KNP?usp=sharing)

gpt-5-pro-mode: [![Open Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vD7T-qkfWrx8-bBUsxI09w5su0K6J4Xp?usp=sharing)

Run the attached notebooks to access Pro Mode! Star this repo and let me know what you want me to add!

You can also set up the integrated Pro Mode API endpoint.

### Run it

```bash
export OPENAI_API_KEY=sk-...   # set your key
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Example request

```bash
curl -X POST http://localhost:8000/pro-mode \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain self-play in RL with a concrete example.","num_gens":5}'
```

New: tournament mode! If `num_gens` is `> 20`, it generates and synthesizes in groups of 10, then synthesizes the them all into one; otherwise, it does the regular single-pass synth.
