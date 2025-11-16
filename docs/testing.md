# Testing

The project has a comprehensive test suite covering model components, data processing, and training utilities. Tests run automatically via GitHub Actions on every push and PR.

## Test Structure

```
tests/
├── test_training_loop.py        # Training/validation/test epochs
├── test_set_completion_model.py # Model architecture validation
├── test_generate_datasets.py    # Data loading and batching
├── test_hook.py                 # Activation hooks
├── test_ablation_component.py   # Dashboard components
├── test_preprocess.py           # Data preprocessing
├── test_embeddings.py           # Doc2Vec experiments
├── test_vocab.py                # Tokenization
└── ...
```

14+ test files covering different modules.

## Running Tests

```bash
poetry run pytest -q
```

Or with verbose output:
```bash
poetry run pytest -v
```

Run specific test file:
```bash
poetry run pytest tests/test_training_loop.py
```

## Test Examples

### Model Forward Pass

Tests that the model produces correct output shapes and handles edge cases:

```python
def test_forward_pass_shape():
    model = SetCompletionModel(
        vocab_size=100,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        output_dim=100,
    )
    abilities = torch.randint(0, 100, (4, 10))  # batch=4, seq=10
    weapons = torch.randint(0, 50, (4,))
    mask = torch.zeros(4, 10, dtype=torch.bool)

    output = model(abilities, weapons, mask)

    assert output.shape == (4, 100)  # batch × vocab_size
```

Verifies that variable-length inputs produce consistent output dimensionality.

### Training Loop

Tests the training infrastructure returns expected metrics:

```python
def test_train_epoch():
    model = create_test_model()
    optimizer = AdamW(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    metrics = train_epoch(model, dataloader, criterion, optimizer, device)

    assert "loss" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert metrics["f1"] >= 0 and metrics["f1"] <= 1
```

Ensures the training loop runs without errors and produces valid metric values.

### Data Loading

Tests that datasets load correctly and batches have proper structure:

```python
def test_generate_dataloaders():
    train_loader, val_loader, test_loader = generate_dataloaders(
        data_path="test_data.csv",
        vocab_path="test_vocab.json",
        weapon_vocab_path="test_weapon_vocab.json",
        batch_size=32,
    )

    batch = next(iter(train_loader))
    abilities, weapons, targets, metadata = batch

    assert abilities.shape[0] == 32  # batch size
    assert weapons.shape[0] == 32
    assert targets.shape[0] == 32
```

### Hook Functionality

Tests that activation hooks capture and modify activations correctly:

```python
def test_hook_captures_activations():
    model = SetCompletionModel(...)
    hook = SetCompletionHook(sae_model)
    model.register_forward_pre_hook(hook)

    output = model(input)

    assert hook.last_activation is not None
    assert hook.last_activation.shape == (batch, hidden_dim)
```

### Preprocessing

Tests data transformation pipeline:

```python
def test_ability_bucketing():
    raw_abilities = {"swim_speed_up": 12, "ink_saver_main": 6}
    tokens = bucket_abilities(raw_abilities)

    assert "swim_speed_up_12" in tokens
    assert "ink_saver_main_6" in tokens
    assert "swim_speed_up_3" in tokens  # lower thresholds included
```

## Fixtures and Test Data

Tests use synthetic data to avoid dependencies on real datasets:

```python
@pytest.fixture
def test_vocab():
    return {
        "<PAD>": 0,
        "<MASK>": 1,
        "ability_1": 2,
        "ability_2": 3,
        # ...
    }

@pytest.fixture
def test_model(test_vocab):
    return SetCompletionModel(
        vocab_size=len(test_vocab),
        embedding_dim=16,
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
    )
```

Fixtures ensure consistent test setup and avoid code duplication.

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/tests.yml`):

```yaml
name: Tests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install --with dev
      - name: Run tests
        run: poetry run pytest -q
```

**Triggers**:
- Every push to `main`
- Every pull request targeting `main`

**Environment**:
- Ubuntu latest
- Python 3.11
- All dev dependencies installed

Tests must pass before merging PRs.

## Code Quality Tools

In addition to tests, the project enforces code quality:

**Black** (formatting):
```bash
poetry run black .
```
Line length: 80 characters (configured in `pyproject.toml`).

**isort** (import sorting):
```bash
poetry run isort .
```
Consistent import ordering.

Run both before committing to ensure consistent style.

## What's Tested

**Correctness**:
- Model produces expected output shapes
- Metrics are computed correctly
- Data loading doesn't corrupt inputs
- Hooks capture activations properly

**Edge Cases**:
- Empty batches
- Variable sequence lengths
- Padding handling
- Zero-division in metrics

**Integration**:
- End-to-end training runs without errors
- Checkpoints save and load correctly
- Distributed training synchronizes properly

## What's Not Tested (Yet)

**Performance**:
- No benchmarking of training speed
- No memory usage tests

**Convergence**:
- No tests that training actually improves metrics
- No regression tests for model quality

**Production**:
- API endpoint testing is minimal
- No load testing

## Writing New Tests

Follow existing patterns:

```python
def test_new_feature():
    # Arrange
    model = create_test_model()
    input_data = generate_test_input()

    # Act
    result = model.new_method(input_data)

    # Assert
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype
    assert torch.isfinite(result).all()
```

Keep tests:
- Fast (use small models, synthetic data)
- Isolated (no external dependencies)
- Deterministic (set random seeds)
- Readable (clear arrange-act-assert structure)

## Code Location

Test suite: `tests/`
CI configuration: `.github/workflows/tests.yml`
Code quality config: `pyproject.toml`
Dev dependencies: `pyproject.toml` (under `[tool.poetry.group.dev.dependencies]`)
