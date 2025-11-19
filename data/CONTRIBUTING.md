# Contributing

Thanks for your interest in contributing!

## Quick rules
- Work on feature branches for non-trivial changes
- Keep commits small and descriptive
- Add tests for bug fixes and new features
- Update docs when APIs change

## How to run tests locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pytest -q
