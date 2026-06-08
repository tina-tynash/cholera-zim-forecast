# Contributing to Cholera Zimbabwe Forecast

Thank you for your interest in contributing! This project welcomes contributions from epidemiologists, data scientists, and public health professionals.

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/cholera-zim-forecast.git
cd cholera-zim-forecast
pip install -r requirements.txt
python data/synthetic/generate_synthetic.py
python src/data/etl.py
pytest tests/ -v
```

## Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Make changes with tests
4. Run: `pytest tests/ -v && flake8 src/ tests/ --max-line-length=120`
5. Commit: `git commit -m "feat: description"`
6. Open a Pull Request

## Code Standards

- PEP8 compliant (max line length 120)
- Google-style docstrings on all functions
- Type hints on all function signatures
- Tests for new model logic or data transformations

## Adding a New Model

1. Create `src/models/your_model.py` with a `.fit()` and `.predict()` interface
2. Add it to `EnsembleForecaster` in `src/models/ensemble.py`
3. Write tests in `tests/test_models.py`

## Reporting Issues

Use GitHub Issues. Include: Python version, OS, error traceback, and steps to reproduce.

## License

By contributing, you agree your contributions will be licensed under MIT.
