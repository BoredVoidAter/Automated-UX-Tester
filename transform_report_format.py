import asyncio
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Gemini API Client
from google.generativeai import GenerativeModel, configure as configure_google_genai
from google.generativeai.types import GenerationConfig  # Optional: for generation config

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# --- Helper Functions ---


async def read_file(filepath: Path) -> str:
    """Reads content from a file."""
    # Note: This simple reader assumes text-based input (txt, md, potentially basic docx/pdf if saved as text).
    # For complex formats (binary docx, pdf), you'd need libraries like python-docx or PyPDF2/PyMuPDF
    # to extract text first before passing to Gemini.
    try:
        return filepath.read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.error(f"Error: Input report file not found at {filepath}")
        raise
    except UnicodeDecodeError:
        logging.warning(f"Could not decode {filepath} as UTF-8. Trying 'latin-1'.")
        try:
            return filepath.read_text(encoding="latin-1")
        except Exception as e:
            logging.error(f"Error reading file {filepath} even with latin-1: {e}")
            raise
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        raise


async def write_file(filepath: Path, content: str):
    """Writes content to a file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        logging.info(f"Successfully wrote transformed report to {filepath}")
    except Exception as e:
        logging.error(f"Error writing file {filepath}: {e}")
        raise


async def transform_report_gemini(input_report_content: str, model_name: str, api_key: str) -> str:
    """
    Uses Gemini API to transform the input report content into the target format.
    """
    logging.info(f"Transforming report format using Gemini model: {model_name}")

    # Define the target format clearly in the prompt
    target_format_description = """
    The target format is plain text. Each question or task for the user/agent should be on its own line,
    followed immediately by a newline and the exact placeholder string "[Agent Answer Here]".
    There should be a blank line separating each question/placeholder pair.

    Example of the TARGET format:
    Was it easy to find the contact information?
    [Agent Answer Here]

    Describe the checkout process.
    [Agent Answer Here]

    How does the site look on mobile?
    [Agent Answer Here]
    """

    prompt = f"""
    You are an AI assistant specializing in standardizing document formats.
    Your task is to transform the provided UX report template (which could be in various formats)
    into a specific, simple text format.

    Identify the core questions or tasks within the input report that are intended to be answered
    by a user or an AI agent evaluating a website's user experience.

    Discard any introductory text, instructions *to the human analyst*, metadata, complex formatting,
    or sections that are not direct questions/tasks for the website evaluation.

    Format the output *exactly* as described below:
    {target_format_description}

    Input Report Content:
    ---
    {input_report_content}
    ---

    Transformed Report Content (in the specified target format):
    """

    try:
        # Configure the client library
        configure_google_genai(api_key=api_key)
        model = GenerativeModel(model_name)
        response = await model.generate_content_async(
            contents=prompt
            # Optional: generation_config=GenerationConfig(temperature=0.1) # Lower temp might help consistency
        )

        if response and hasattr(response, "text"):
            transformed_content = response.text.strip()
            logging.info("Successfully received transformed content from Gemini.")
            # Basic validation: check if the placeholder exists
            if "[Agent Answer Here]" not in transformed_content:
                logging.warning(
                    "Transformed content might not be in the expected format - placeholder '[Agent Answer Here]' missing."
                )
            return transformed_content
        else:
            logging.error(
                f"Failed to get valid text response from Gemini model {model_name} during transformation. Response: {response}"
            )
            raise ValueError(f"Invalid response from Gemini model {model_name} during transformation")

    except Exception as e:
        logging.error(f"Error during Gemini report transformation with model {model_name}: {e}")
        raise


# --- Main Execution Logic ---


async def main(args):
    """Main function to orchestrate the report transformation."""
    try:
        # 1. Get API Key securely
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or .env file.")

        # 2. Read the input report file
        logging.info(f"Reading input report from: {args.input_file}")
        input_content = await read_file(args.input_file)

        # 3. Transform the report using Gemini
        transformed_content = await transform_report_gemini(
            input_report_content=input_content, model_name=args.gemini_model, api_key=gemini_api_key
        )

        # 4. Write the transformed report
        await write_file(args.output_file, transformed_content)

        logging.info("Report transformation process completed successfully.")

    except ValueError as ve:
        logging.error(f"Configuration or Value Error: {ve}")
    except FileNotFoundError as fnfe:
        logging.error(f"File Error: {fnfe}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during transformation: {e}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform a UX report into a standardized format using Gemini.")

    parser.add_argument("input_file", type=Path, help="Path to the input UX report file (any text-readable format).")
    parser.add_argument(
        "output_file", type=Path, help="Path to save the transformed UX report file (in the standardized .txt format)."
    )

    # Model Configuration
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-1.5-pro-latest",
        help="Gemini model to use for the transformation task.",
    )

    args = parser.parse_args()

    # Run the asynchronous main function
    asyncio.run(main(args))
