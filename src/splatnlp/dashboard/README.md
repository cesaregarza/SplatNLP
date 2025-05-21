# SAE Feature Dashboard

This dashboard provides an interactive, visual interface for exploring and interpreting features learned by a Sparse Autoencoder (SAE) trained on Splatoon ability set data. It is designed to help researchers and developers understand the internal representations of the model, debug feature behavior, and generate insights for further model development.

## Features (Current & Planned)

### 1. Feature Selector
- Dropdown or search to select a specific SAE feature by index.
- Hyperlinkable feature numbers for easy sharing/bookmarking.

### 2. Feature Summary
- Display the selected feature number, auto-interpretation score, and (optionally) a human explanation.
- Show ablation/prediction score for the feature.

### 3. Activation Histogram
- Histogram of nonzero activations for the selected feature across the dataset.
- Option to view all activations or only nonzero values.

### 4. Top Output Logits
- Bar chart or table showing the top 10 most negative and positive output logits for the feature.

### 5. Top Activating Examples
- Table or card view of the top 20 examples that maximally activate the feature.
- Each example shows context, input tokens, and output predictions.

### 6. Subsampled Intervals Grid
- Ten evenly spaced intervals spanning the full range of activation values.
- For each interval, show a few representative examples, with context and ablation coloring.
- Indicate if an example appears in multiple intervals.

### 7. Correlations
- Top 3 neurons by activation (how much the feature activates them).
- Top 3 neurons by token correlation.
- Top 3 features from a parallel run (different random seed).

### 8. Tooltips & Color Coding
- Hover over any token to see its activation value and ablation loss.
- Blue underline = lower ablation loss (better prediction), red = higher loss.
- Bold = token from training data used to select the example.

### 9. Data/Model Management
- Persistent caching of activations and records for fast reloads.
- Support for large datasets via efficient serialization (joblib, etc).

## Usage

1. Run the dashboard with:
   ```bash
   python scripts/run_dashboard.py
   ```
2. Open the provided local URL in your browser.
3. Select a feature and explore its properties interactively.

## Development Roadmap
- [x] Basic Dash app scaffold
- [x] Feature selector and activation histogram
- [x] Persistent caching of activations
- [ ] Wire up all visualizations to real data
- [ ] Add top logits, examples, intervals, and correlations
- [ ] Add tooltips, ablation coloring, and hyperlinks
- [ ] Support for custom datasets and models

---

**Contributions and feedback are welcome!** 