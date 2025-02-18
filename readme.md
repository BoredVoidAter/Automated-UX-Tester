# Automated UX Report Generator

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automate the process of gathering user experience (UX) feedback for websites using Google Gemini and the `browser-use` library. This project provides scripts to:

1.  Transform various UX report template formats into a standardized structure.
2.  Use an AI agent powered by Gemini to browse a target website based on the standardized report tasks.
3.  Generate a filled-in UX report based on the agent's findings.

## Problem Solved

Traditional UX testing requires significant manual effort in recruiting participants, conducting sessions, and analyzing feedback. This project aims to streamline the initial feedback gathering phase by using AI agents to perform predefined tasks on a website, providing rapid, automated insights to supplement human testing.

## Features

*   **Report Standardization:** Converts UX report templates from potentially varied formats into a consistent input format for the automation script (`transform_report_format.py`).
*   **AI-Powered Task Extraction:** Uses Gemini to understand the tasks required from the standardized report template.
*   **Automated Browser Interaction:** Leverages the `browser-use` library and Langchain to enable a Gemini agent to navigate and interact with websites (clicking, typing, extracting content).
*   **Automated Report Generation:** Employs Gemini to analyze the agent's browsing history and findings to fill out the original report template.
*   **Configurable Models:** Allows specifying different Gemini models for various stages (task extraction, agent control, report generation) via command-line arguments.
*   **Headless/Headed Mode:** Supports running the browser automation visibly or in the background.

## How It Works

The project consists of two main scripts:

1.  **`transform_report_format.py`:**
    *   Takes a UX report template file (in a text-readable format) as input.
    *   Uses a Gemini model to identify the core questions/tasks meant for website evaluation.
    *   Outputs a new file (`.txt`) containing these tasks in a standardized format:
        ```
        Question 1?
        [Agent Answer Here]

        Question 2?
        [Agent Answer Here]
        ```
2.  **`automated_ux_reporter.py`:**
    *   Takes the standardized report template file (output from the transformer), a target website URL, and an output file path as input.
    *   Uses Gemini to extract a combined task list from the standardized report.
    *   Initializes a `browser-use` agent with a specified Gemini model (via Langchain) to perform the extracted tasks on the target URL.
    *   Once the agent completes its run (or hits limits), it uses Gemini again to analyze the agent's actions, visited URLs, extracted content, and errors.
    *   Fills in the `[Agent Answer Here]` placeholders in the original standardized report template with the analysis results.
    *   Saves the completed report to the specified output file.

## Prerequisites

*   **Python:** Version 3.11 or higher.
*   **Google Gemini API Key:** You need an active API key from Google AI Studio.
*   **Environment Management (Recommended):** `uv` (as recommended by `browser-use`) or `pip` with `venv`.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Set Up Python Environment (using `uv`):**
    ```bash
    # Install uv if you haven't already: pip install uv
    uv venv --python 3.11 # Or your python version >= 3.11
    source .venv/bin/activate # Linux/macOS
    # .venv\Scripts\activate # Windows
    ```
    *(Alternatively, use `python -m venv .venv` and standard pip)*

3.  **Install Dependencies:**
    ```bash
    uv pip install browser-use google-generativeai langchain-google-genai python-dotenv pydantic playwright
    ```
    *(If not using `uv`, replace `uv pip` with `pip`)*

4.  **Install Playwright Browsers:**
    ```bash
    playwright install
    ```

5.  **Create `.env` File:**
    Create a file named `.env` in the project root directory and add your Gemini API key:
    ```dotenv
    GEMINI_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY
    ```

## Usage

**Step 1: Standardize the Report Format (If Necessary)**

If your input report is not already in the required simple format, use the transformer script first:

```bash
python transform_report_format.py path/to/your/input_report.txt path/to/standardized_report.txt --gemini-model gemini-1.5-pro-latest
```

*   Replace `path/to/your/input_report.txt` with the path to your original report template.
*   Replace `path/to/standardized_report.txt` with the desired output path for the standardized file.
*   You can change the `--gemini-model` if needed.

**Step 2: Run the Automated Reporter**

Use the standardized report file (either your original one if already formatted, or the output from Step 1) to run the main automation:

```bash
python automated_ux_reporter.py path/to/standardized_report.txt path/to/completed_report.txt "https://target-website.com" --gemini-model-agent gemini-1.5-pro-latest --headless```

*   Replace `path/to/standardized_report.txt` with the path to the correctly formatted input template.
*   Replace `path/to/completed_report.txt` with the desired path for the final, filled-in report.
*   Replace `"https://target-website.com"` with the actual URL to evaluate.
*   Adjust `--gemini-model-agent` (and other model args) as needed (using stronger models like `gemini-1.5-pro-latest` for the agent is recommended).
*   Use `--headless` to run the browser without a visible UI (optional).

## Configuration

*   **API Key:** The Gemini API key *must* be set in the `.env` file.
*   **Command-Line Arguments (`automated_ux_reporter.py`):**
    *   `input_file`: Path to the standardized input report template.
    *   `output_file`: Path to save the completed report.
    *   `target_url`: The URL of the website to test.
    *   `--gemini-model-task-extraction`: Model for extracting tasks initially (default: `gemini-1.5-flash-latest`).
    *   `--gemini-model-agent`: Model used by the `browser-use` agent (default: `gemini-1.5-pro-latest`). **Crucial for performance.**
    *   `--gemini-model-report-generation`: Model for filling the final report (default: `gemini-1.5-pro-latest`).
    *   `--headless`: Run browser headlessly.
*   **Command-Line Arguments (`transform_report_format.py`):**
    *   `input_file`: Path to the original report template.
    *   `output_file`: Path to save the standardized report.
    *   `--gemini-model`: Model for the transformation task (default: `gemini-1.5-pro-latest`).

## Standardized Input Format (`automated_ux_reporter.py`)

The main script requires the input report template (`input_file`) to be a plain text file with the following structure:

```
Question/Task Description 1
[Agent Answer Here]

Another Question/Task Description 2
[Agent Answer Here]

A third item to evaluate
[Agent Answer Here]```

*   Each question or task description should be on its own line.
*   The exact placeholder string `[Agent Answer Here]` must follow *immediately* on the next line.
*   A blank line should separate each question/placeholder pair.

Use the `transform_report_format.py` script to convert other formats into this structure.

## License

This project is licensed under the MIT License - see the LICENSE file for details (or assume MIT if no file is present).

## Acknowledgements

*   [Google Gemini](https://ai.google.dev/)
*   [browser-use](https://docs.browser-use.com/) library
*   [Langchain](https://www.langchain.com/)
*   [Playwright](https://playwright.dev/)