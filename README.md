
<div id="top">  <!-- HEADER STYLE: CLASSIC -->
<div align="left">

# TIME_SERIES_ANALYSIS_AMDOX  
<em>Unlock Future Insights from Time Series Data</em>

<!-- BADGES -->
<img src="https://img.shields.io/github/license/mommy15/TIME_SERIES_ANALYSIS_AMDOX?style=flat&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
<img src="https://img.shields.io/github/last-commit/mommy15/TIME_SERIES_ANALYSIS_AMDOX?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
<img src="https://img.shields.io/github/languages/top/mommy15/TIME_SERIES_ANALYSIS_AMDOX?style=flat&color=0080ff" alt="repo-top-language">
<img src="https://img.shields.io/github/languages/count/mommy15/TIME_SERIES_ANALYSIS_AMDOX?style=flat&color=0080ff" alt="repo-language-count">

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=flat&logo=TensorFlow&logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat&logo=pandas&logoColor=white" alt="pandas">

</div>
<br>

---
---

## Live Application

https://timeseriesanalysis-cryptocurrency.streamlit.app

> Note: When deployed on Streamlit Cloud, the application may run in **Demo Mode** due to temporary Yahoo Finance API restrictions on shared cloud IPs.  
> When executed locally, the application works fully with live Yahoo Finance data.
---

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Features](#features)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

TIME_SERIES_ANALYSIS_AMDOX is an advanced developer tool crafted to facilitate comprehensive cryptocurrency time series analysis, forecasting, and visualization. Built with Python and Streamlit, it empowers users to explore historical crypto market data, engineer features, and apply multiple modeling techniques through an interactive web dashboard.

The application supports **local execution with live Yahoo Finance data** and **cloud deployment with automatic demo-mode fallback** to ensure stability in environments where external APIs may be rate-limited.

**Why TIME_SERIES_ANALYSIS_AMDOX?**

This project aims to simplify complex financial data workflows while providing meaningful analytical insights. The core capabilities include:

- 🧩 **🎯 Data Collection & Feature Engineering:** Fetches and preprocesses historical cryptocurrency price data.
- 🧠 **📊 Interactive Dashboards:** Visualizes trends, volatility, and indicators using Streamlit.
- 🔍 **🧪 Multiple Modeling Techniques:** Forecasts future prices using ARIMA, LSTM, and Prophet.
- 💹 **📈 Sentiment & Volatility Metrics:** Enhances analysis with NLP-based sentiment scoring and rolling volatility.
- ⚙️ **🛠️ End-to-End Pipeline:** Integrates data ingestion, analysis, modeling, and visualization in one workflow.
- 🖥️ **🧰 Compatibility & Stability:** Designed for Python 3.10.13 with cloud-safe dependency management.

---

## Features

|      | Component        | Details |
| :--- | :--------------- | :-------------------------------------------------------------------------------------------------------------- |
| ⚙️  | **Architecture** | <ul><li>Modular pipeline covering data ingestion, preprocessing, modeling, and visualization</li><li>Supports ARIMA, LSTM, and Prophet forecasting models</li><li>Streamlit-based orchestration for user interaction</li></ul> |
| 🔩 | **Code Quality** | <ul><li>Readable, modular Python code</li><li>Clear separation of analytical stages</li><li>Reusable functions for data loading and modeling</li></ul> |
| 📄 | **Documentation** | <ul><li>Comprehensive README explaining setup, usage, and deployment behavior</li><li>Inline comments for clarity</li></ul> |
| 🔌 | **Integrations** | <ul><li>Yahoo Finance for historical price data</li><li>pandas and NumPy for data processing</li><li>statsmodels, TensorFlow, Prophet for forecasting</li><li>VADER for sentiment analysis</li></ul> |
| 🧩 | **Modularity** | <ul><li>Logical separation of analytics sections</li><li>Configurable dependencies via requirements.txt</li></ul> |
| 🧪 | **Testing** | <ul><li>Manual testing via local execution and Streamlit UI</li><li>Model outputs validated visually</li></ul> |
| ⚡️ | **Performance** | <ul><li>Optimized numerical libraries</li><li>Lightweight LSTM configuration for interactive use</li></ul> |
| 🛡️ | **Security** | <ul><li>No authentication layer (educational scope)</li><li>No sensitive data storage</li></ul> |
| 📦 | **Dependencies** | <ul><li>Managed through requirements.txt</li><li>Runtime pinned via runtime.txt</li></ul> |

---

## Project Structure

```sh
└── TIME_SERIES_ANALYSIS_AMDOX/
    ├── LICENSE
    ├── app.py
    ├── requirements.txt
    └── runtime.txt
````

---

## Getting Started

### Prerequisites

This project requires the following:

* **Programming Language:** Python 3.10+
* **Package Manager:** pip

---

### Installation

Build TIME_SERIES_ANALYSIS_AMDOX from source and install dependencies:

1. **Clone the repository:**

```sh
❯ git clone https://github.com/mommy15/TIME_SERIES_ANALYSIS_AMDOX
```

2. **Navigate to the project directory:**

```sh
❯ cd TIME_SERIES_ANALYSIS_AMDOX
```

3. **Install the dependencies:**

```sh
❯ pip install -r requirements.txt
```

---

### Usage

Run the application locally using Streamlit:

```sh
streamlit run app.py
```

Running locally enables **live Yahoo Finance data access**.

---

### Testing

This project is validated through:

* Local execution testing
* Streamlit UI-based validation of plots and forecasts
* Visual verification of ARIMA, LSTM, and Prophet outputs

Formal unit testing is not included due to the exploratory and visualization-focused nature of the application.

---

## License

Time_series_analysis_amdox is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the LICENSE file.

---

<div align="left"><a href="#top">⬆ Return</a></div>

---
