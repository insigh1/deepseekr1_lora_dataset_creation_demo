# DeepSeekR1 LoRA Dataset Config Generator

This project provides a Python tool to automatically generate comprehensive YAML configuration files for dataset creation using LLMs (Fireworks/DeepSeek). It is designed to help you quickly set up high-quality, structured configs for training, evaluation, or data generation tasks.

---

## Features
- **LLM-powered config generation**: Uses Fireworks AI to analyze your task and generate optimal config parameters.
- **Covers all aspects**: Sampling, prompt structure, validation, diversity, evaluation metrics, and more.
- **Customizable**: Edit the generated YAML or extend the Python code for your needs.
- **Easy to use**: One command to generate a ready-to-use config file.

---

## Requirements
- Python 3.8+
- [Fireworks AI API key](https://fireworks.ai/)
- Python packages: `pydantic`, `requests`, `pyyaml`

Install dependencies:
```bash
pip install pydantic requests pyyaml
```

---

## Setup
1. **Clone this repo** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd deepseekr1_lora_dataset_creation_demo
   ```
2. **Set your Fireworks API key**:
   ```bash
   export FIREWORKS_API_KEY=sk-...your_key_here...
   ```

---

## Usage
Run the config generator with your task description:

```bash
python create_config.py --task "YOUR TASK DESCRIPTION HERE"
```

**Example:**
```bash
python create_config.py --task "Generate technical documentation Q&A pairs for Python beginners"
```

**Optional:** Specify an output file:
```bash
python create_config.py --task "..." --output my_config.yaml
```
If not specified, the script auto-generates a filename.

---

## Output
- The script generates a YAML config file in the current directory.
- The config includes:
  - Sampling parameters
  - Prompt structure
  - Required criteria
  - Advanced options
  - Dataset parameters (with message format)
  - Quality controls
  - Diversity parameters
  - Evaluation metrics
  - Example inputs

---

## Troubleshooting
- **Missing API Key:**
  - If you see an error about `FIREWORKS_API_KEY`, make sure you exported it in your shell.
- **Missing dependencies:**
  - If you get `ModuleNotFoundError`, install the missing package with `pip install ...`.
- **LLM/API errors:**
  - If the LLM returns invalid JSON, the script will try to recover, but you may need to rerun.

---

## Customization
- Edit the generated YAML file to fine-tune your config.
- The code is modular: you can import functions from `create_config.py` in your own scripts.
- Advanced users can extend the Pydantic models in `config_models.py`.

---

## License
MIT License

---

## Contact
For questions or suggestions, open an issue or contact the maintainer.
