# Jupyter Notebooks

Interactive notebooks for exploring and analyzing the prompt optimization system.

## Notebooks

### 1. Data Exploration (`01_data_exploration.ipynb`)
Explore and visualize the datasets.

**Topics covered:**
- Loading Dataset A (Simple QA) and Dataset B (Multi-step Reasoning)
- Category distribution analysis
- Difficulty distribution (Dataset A)
- Reasoning step counts (Dataset B)
- Sample examples from each category
- Dataset validation

**Run this first** to understand the data structure.

### 2. Prompt Techniques Demo (`02_prompt_techniques_demo.ipynb`)
Demonstrates all 7 prompt optimization techniques.

**Techniques demonstrated:**
1. Baseline (direct questioning)
2. Chain-of-Thought (CoT)
3. Chain-of-Thought++ (CoT++)
4. ReAct (Reasoning + Acting)
5. Tree-of-Thoughts (ToT)
6. Role-Based Prompting
7. Few-Shot Learning

**Use this** to see how each technique formats prompts differently.

### 3. Results Analysis (`03_results_analysis.ipynb`)
Analyzes experimental results and generates visualizations.

**Analysis includes:**
- Loading experimental results
- Accuracy and loss comparisons
- Statistical validation (confidence intervals, t-tests)
- Pairwise technique comparisons
- Overall rankings
- Summary tables

**Note:** Uses mock data for demonstration. After running actual experiments, update the results path to analyze real data.

## Getting Started

### Prerequisites
```bash
# Install Jupyter
pip install jupyter notebook

# Or use JupyterLab
pip install jupyterlab
```

### Launch Jupyter

From the project root:
```bash
# Start Jupyter Notebook
jupyter notebook notebooks/

# Or JupyterLab
jupyter lab notebooks/
```

### Running Notebooks

1. Open `01_data_exploration.ipynb` first
2. Run all cells: `Cell > Run All`
3. Explore the outputs and visualizations
4. Move to the next notebook

## Tips

- **Restart kernel** if you modify source code: `Kernel > Restart & Run All`
- **Save plots** by right-clicking on figures
- **Modify examples** to test different scenarios
- **Add new cells** to try your own analyses

## Integration with Experiments

After running a full experiment:
```bash
python main.py run-experiment --model gpt-4
```

Update the results path in `03_results_analysis.ipynb`:
```python
results_path = '../results/experiment_results.json'
```

Then re-run the notebook to analyze your actual experimental data!

## Export Options

Export notebooks to different formats:

```bash
# HTML
jupyter nbconvert --to html 01_data_exploration.ipynb

# PDF (requires LaTeX)
jupyter nbconvert --to pdf 01_data_exploration.ipynb

# Python script
jupyter nbconvert --to script 01_data_exploration.ipynb
```
