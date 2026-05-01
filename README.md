# interest-rate-derivatives

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

[![Coverage][coverage-badge]][coverage-link]

<!-- SPHINX-START -->

The go-to package for pricing interest rate derivatives with real-time treasury yield curve data visualization.

## Dashboard

### Quick Start

The package includes an interactive Streamlit dashboard for visualizing U.S. Treasury yield curves from FRED (Federal Reserve Economic Data).

#### Running the Dashboard

```bash
streamlit run src/interest_rate_derivatives/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

#### Getting a FRED API Token

1. Visit https://fred.stlouisfed.org/user/register
2. Create a free account
3. Go to your account settings and generate an API key
4. Either:
   - Enter the API key directly in the dashboard's sidebar, or
   - Save it in a `.env` file in the project root:
     ```
     FRED_API_KEY=your_api_key_here
     ```

**Note:** Without an API key, the dashboard displays placeholder data. With an API key, you get real-time treasury yields.

### Dashboard Features

- Interactive yield curve visualization with Plotly
- Real-time data from FRED API
- Curve statistics (shortest/longest maturity yields, slope)
- Detailed data table with maturity and yield information
- Date selection for historical yield curves

## Development

### Installing the Development Environment

We use `uv` for fast and reliable dependency management.

#### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

#### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AlexSamuseva/interest-rate-derivatives-pricing.git
   cd interest-rate-derivatives
   ```

2. Install dependencies with uv:
   ```bash
   uv sync
   ```

This command will create a virtual environment and install all dependencies (including dev, test, and docs dependencies).

### Running the Dashboard in Development

After setting up the environment with `uv sync`, run:

```bash
streamlit run src/interest_rate_derivatives/app.py
```

The dashboard will be available at `http://localhost:8501` with hot-reload enabled for development.

### Running Tests

```bash
uv run pytest
```

### Code Quality Checks

- **Linting:** `uv run ruff check .`
- **Type checking:** `uv run mypy src tests`
- **Formatting:** `uv run ruff format .`

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/AlexSamuseva/interest-rate-derivatives-pricing/actions/workflows/ci.yml/badge.svg
[actions-link]:             https://github.com/AlexSamuseva/interest-rate-derivatives-pricing/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/interest-rate-derivatives
[conda-link]:               https://github.com/conda-forge/interest-rate-derivatives-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/AlexSamuseva/interest-rate-derivatives-pricing/discussions
[pypi-link]:                https://pypi.org/project/interest-rate-derivatives/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/interest-rate-derivatives
[pypi-version]:             https://img.shields.io/pypi/v/interest-rate-derivatives
[rtd-badge]:                https://readthedocs.org/projects/interest-rate-derivatives/badge/?version=latest
[rtd-link]:                 https://interest-rate-derivatives.readthedocs.io/en/latest/?badge=latest
[coverage-badge]:           https://codecov.io/github/org/interest-rate-derivatives/branch/main/graph/badge.svg
[coverage-link]:            https://codecov.io/github/org/interest-rate-derivatives

<!-- prettier-ignore-end -->
