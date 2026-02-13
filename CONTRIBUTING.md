# Contributing to ROCm Profiling Playbook

Thank you for your interest in contributing!

## Development Setup

### Prerequisites
- ROCm 5.0+ installed
- Python 3.8+
- AMD GPU

```bash
git clone https://github.com/sudheerdevu/ROCm-Profiling-Playbook.git
cd ROCm-Profiling-Playbook

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Adding Guides

1. Create new guide in `guides/` or `docs/`
2. Follow existing markdown format
3. Include practical examples
4. Add code samples

## Adding Analysis Tools

1. Add to `analysis/` directory
2. Include docstrings
3. Add usage examples

## Pull Request Process

1. Fork repository
2. Create feature branch
3. Test on actual AMD GPU hardware
4. Submit PR with examples

## License

Contributions are licensed under MIT License.
