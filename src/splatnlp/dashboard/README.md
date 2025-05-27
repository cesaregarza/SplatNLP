# SAE Feature Dashboard

This dashboard provides an interactive, visual interface for exploring and interpreting features learned by a Sparse Autoencoder (SAE) trained on Splatoon ability set data. It is designed to help researchers and developers understand the internal representations of the model, debug feature behavior, and generate insights for further model development.

## Features

This section details the currently implemented features of the dashboard.

### 1. Feature Selector
- **Dropdown Menu**: Allows selection of a specific SAE feature by its index. The dropdown is dynamically populated based on the loaded SAE model's `hidden_dim`.
- **URL Synchronization**: The selected feature ID is reflected in the URL (e.g., `/?feature=X`), enabling direct linking and bookmarking of specific feature views. The dashboard also initializes with the feature specified in the URL, if present.

### 2. Feature Summary
- **Selected Feature Display**: Shows the ID of the currently selected feature.
- *(Planned: Display for auto-interpretation score, human explanation, and ablation/prediction score for the feature, pending data integration.)*

### 3. Activation Histogram
- **Histogram Display**: Visualizes the distribution of activations for the selected SAE feature across the dataset.
- **Filtering Options**: Provides a radio button toggle to view either "All Activations" or only "Non-Zero Activations" (values > 1e-6).

### 4. Top Output Logits
- **Logit Influence Chart**: Displays a bar chart showing the output vocabulary tokens whose logits are most positively and negatively influenced by the selected SAE feature. This is calculated by projecting the SAE feature's decoder vector through the primary model's output layer.

### 5. Top Activating Examples
- **Tabular View**: Presents a table of the top ~20 examples from the dataset that maximally activate the selected SAE feature.
- **Example Details**: Each row shows: Rank, Weapon name, Input ability tokens, the SAE feature's activation value, Top predicted ability tokens, and the Original index of the example.
- **Token Projection Tooltips**: If per-token primary model hidden state activations are provided via the `--token-activations-path` argument to the CLI, hovering over input ability tokens will display a tooltip showing the projection of that token's activation vector onto the selected SAE feature's direction vector.

### 6. Subsampled Intervals Grid
- **Activation Intervals**: Divides the full range of the selected SAE feature's activation values into ~10 evenly spaced intervals.
- **Representative Examples**: For each interval, a few representative examples are displayed.
- **Example Details**: Each example shows the weapon, input abilities, the feature's activation value, and the top model prediction.
- **Token Projection Tooltips**: Similar to "Top Activating Examples," if per-token primary model hidden state activations are provided, hovering over input ability tokens will display their activation projection onto the selected SAE feature's direction.

### 7. Correlations
- **SAE Feature-to-Feature Correlations**: Shows a list of other SAE features whose activation patterns across the dataset are most correlated (Pearson correlation) with the selected feature.
- **SAE Feature-to-Token-Logit Correlations**: Displays a list of output vocabulary tokens whose logit values (across all examples) are most correlated (Pearson correlation) with the selected SAE feature's activation values.

## Memory-Efficient Processing

The dashboard includes memory-efficient processing capabilities for handling large datasets:

1. **Precomputed Analytics**: All heavy computations are performed offline and stored in a single analytics file
2. **Chunked Processing**: Activations are processed in chunks to manage memory usage
3. **Parallel Processing**: Multi-core CPU utilization for faster processing
4. **Efficient Storage**: HDF5 format for large activation matrices
5. **Minimal Memory Footprint**: Only essential data is kept in memory

### Processing Pipeline

The dashboard provides a simple command-line interface for processing data:

```bash
# Generate activations
./dashboard.sh generate primary_model.pth sae_model.pth vocab.json weapon_vocab.json data.json

# Precompute all dashboard analytics
./dashboard.sh precompute activations.h5 metadata.pkl primary_model.pth sae_model.pth dashboard_analytics.joblib

# Run the dashboard
./dashboard.sh run
```

## Usage

1. **Prepare Data:**
   * Ensure you have trained primary model and SAE model checkpoints (`.pth` files)
   * Have your `vocab.json` and `weapon_vocab.json` files ready
   * Generate the activations and precompute analytics using the processing pipeline above

2. **Run the Dashboard:**
   ```bash
   ./dashboard.sh run
   ```
   Or directly:
   ```bash
   python -m splatnlp.dashboard.cli \
       --precomputed-analytics-path /path/to/your/dashboard_analytics.joblib \
       --vocab-path /path/to/your/vocab.json \
       --weapon-vocab-path /path/to/your/weapon_vocab.json \
       # Optional arguments:
       # --enable-dynamic-tooltips \
       # --primary-model-checkpoint /path/to/your/primary_model.pth \
       # --sae-model-checkpoint /path/to/your/sae_model.pth \
       # --host 127.0.0.1 \
       # --port 8050 \
       # --debug
   ```

3. **Interact:**
   Open the local URL provided by the script (usually `http://127.0.0.1:8050/`) in your web browser. Select an SAE feature and explore its various analytical views.

## Development Roadmap
- [x] Basic Dash app scaffold
- [x] Feature selector and activation histogram
- [x] Memory-efficient processing pipeline
- [x] Wire up all visualizations to real data
- [x] Add top logits, examples, intervals, and correlations components
- [x] Add tooltips for token activation projections
- [x] Precompute all analytics for faster loading
- [ ] Integrate auto-interpretation scores and human explanations
- [ ] Add ablation-related scores and visualizations
- [ ] Further enhancements to interactivity and visual styling
- [ ] Support for custom datasets and models

---

**Contributions and feedback are welcome!** 