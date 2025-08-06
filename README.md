# GPT-OSS Pro Mode

A collection of Jupyter notebooks that implement **Pro Mode** - an advanced AI reasoning technique that generates multiple candidate responses in parallel and then synthesizes them into a single, high-quality answer.

## 🎯 What is Pro Mode?

Pro Mode is a sophisticated approach to AI reasoning that mimics how expert humans think through complex problems:

1. **Generate Multiple Perspectives**: Creates several candidate responses to the same question
2. **Parallel Processing**: Uses multiple AI instances simultaneously for efficiency
3. **Intelligent Synthesis**: Combines the best parts of each candidate into a final, refined answer

This technique significantly improves answer quality, reduces errors, and provides more comprehensive responses compared to single-shot AI interactions.

## 🚀 Key Benefits

- **Higher Quality Answers**: Multiple perspectives lead to more thorough and accurate responses
- **Error Reduction**: Synthesis process catches and corrects individual mistakes
- **Better Reasoning**: Combines strengths from different approaches
- **Comprehensive Coverage**: Addresses aspects that single responses might miss

## 📁 Available Implementations

This repository contains three different implementations of Pro Mode:

### 1. **Groq Implementation** (`OpenAI_Open_Source_Pro_Mode_Groq.ipynb`)
- Uses Groq's fast inference API
- Requires Groq API key
- Optimized for speed and cost-effectiveness
- Best for production use cases

### 2. **Ollama Local Implementation** (`OpenAI_Open_Source_Pro_Mode_Ollama_Local.ipynb`)
- Runs locally using Ollama
- No API costs or internet required
- Uses the `gpt-oss:120b` model
- Perfect for privacy-conscious users

### 3. **Ollama Turbo Implementation** (`OpenAI_Open_Source_Pro_Mode_Ollama_Turbo.ipynb`)
- Enhanced version with additional features
- More sophisticated synthesis process
- Better error handling and retry logic

## 🔧 How It Works

### The Pro Mode Process

1. **Parallel Generation**: 
   - Takes your prompt and generates `n_runs` candidate responses simultaneously
   - Uses high temperature (0.9) for creative diversity
   - Runs in parallel threads for efficiency

2. **Synthesis Phase**:
   - An expert editor AI analyzes all candidates
   - Merges strengths, corrects errors, removes repetition
   - Uses low temperature (0.2) for focused synthesis
   - Produces a single, refined final answer

3. **Quality Output**:
   - Returns both the final synthesized answer and all candidates
   - Allows inspection of individual candidates if needed

### Example Usage

```python
# Set up your preferred implementation
# (Groq, Ollama Local, or Ollama Turbo)

# Define your question
PROMPT = "Explain self-play in reinforcement learning with a concrete example."
NUMBER_OF_CANDIDATES = 3  # Adjust based on complexity

# Run Pro Mode
result = pro_mode(client, PROMPT, NUMBER_OF_CANDIDATES)

# Get the final synthesized answer
print("=== FINAL ANSWER ===")
print(result["final"])

# Optionally inspect individual candidates
for i, candidate in enumerate(result["candidates"], 1):
    print(f"\n--- Candidate {i} ---")
    print(candidate)
```

## 🛠️ Setup Instructions

### For Groq Implementation:
1. Get a Groq API key from [groq.com](https://groq.com)
2. Set environment variable: `export GROQ_API_KEY="your-key-here"`
3. Run the notebook: `OpenAI_Open_Source_Pro_Mode_Groq.ipynb`

### For Ollama Local Implementation:
1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Pull the model: `ollama pull gpt-oss:120b`
3. Run the notebook: `OpenAI_Open_Source_Pro_Mode_Ollama_Local.ipynb`

### For Ollama Turbo Implementation:
1. Follow same setup as Ollama Local
2. Run the notebook: `OpenAI_Open_Source_Pro_Mode_Ollama_Turbo.ipynb`

## ⚙️ Configuration Options

- **`n_runs`**: Number of candidate responses (2-5 recommended)
- **`MAX_COMPLETION_TOKENS`**: Maximum response length (default: 30000)
- **Temperature**: 0.9 for candidates, 0.2 for synthesis
- **Model**: `openai/gpt-oss-120b` (Groq) or `gpt-oss:120b` (Ollama)

## 🎯 When to Use Pro Mode

**Best for:**
- Complex reasoning problems
- Technical explanations
- Creative writing tasks
- Problem-solving scenarios
- Research and analysis

**Not needed for:**
- Simple factual questions
- Basic text generation
- Real-time applications (due to parallel processing time)

## 📊 Performance Tips

- Start with 2-3 candidates for most questions
- Increase to 4-5 for very complex problems
- Monitor API costs when using Groq
- Consider local Ollama for privacy-sensitive tasks

## 🤝 Contributing

Created by Matt Shumer ([@mattshumer_](https://x.com/mattshumer_) on X)

Feel free to:
- Submit issues and feature requests
- Contribute improvements to the synthesis logic
- Add new model implementations
- Share your use cases and results

## 📄 License

This project is open source. Please check individual notebook headers for specific licensing information.

---

**Pro Mode transforms single-shot AI interactions into collaborative reasoning sessions, delivering significantly higher quality results through intelligent synthesis of multiple perspectives.**

# gpt-oss-pro-mode

[![Twitter Follow](https://img.shields.io/twitter/follow/mattshumer_?style=social)](https://x.com/mattshumer_)

[![Open Notebook In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XeYmOHJwACtavCjJM-eOqlPxHgTD2KNP?usp=sharing)