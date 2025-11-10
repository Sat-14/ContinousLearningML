# ContinousLearningML

This repository contains a small demo script, `code.py`, that compares several
continual-learning strategies on SplitCIFAR-100 using a pretrained ViT.

Quick start
1. Create a Python environment (3.8+ recommended).
2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the experiment (warning: needs GPU or will be slow):

```powershell
python code.py
```

Notes
- The repository includes three paper PDFs (`paper1.pdf`, `paper2.pdf`, `paper3.pdf`) used as references.
- `code.py` was cleaned to be more robust; it saves partial results to `cl_vit_results_partial.csv` during runs.

If you want, I can (a) summarize the PDFs, (b) add a lightweight smoke test to run on CPU, or (c) prepare an experiment config.

