import asyncio
import os
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv
from pydantic import SecretStr  # Good practice for API keys

# Gemini API Client (Correct usage)
from google.generativeai import GenerativeModel, configure as configure_google_genai
from google.generativeai.types import GenerationConfig  # Optional: for generation config

# Langchain integration for Browser-Use
from langchain_google_genai import ChatGoogleGenerativeAI

# Browser-Use components
from browser_use import Agent as BrowserAgent
from browser_use import Browser, BrowserConfig, AgentHistoryList

# Removed: from browser_use.browser.context import BrowserContextConfig (if not directly used)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# --- Helper Functions ---


async def read_file(filepath: Path) -> str:
    """Reads content from a file."""
    try:
        return filepath.read_text(encoding="utf-8")
    except FileNotFoundError:
        logging.error(f"Error: Input report file not found at {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error reading file {filepath}: {e}")
        raise


async def write_file(filepath: Path, content: str):
    """Writes content to a file."""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        logging.info(f"Successfully wrote completed report to {filepath}")
    except Exception as e:
        logging.error(f"Error writing file {filepath}: {e}")
        raise


# CORRECTED Gemini API Usage
async def generate_content_gemini(prompt: str, model_name: str, api_key: str) -> str:
    """Helper to generate content using google-genai asynchronously."""
    try:
        # Configure the client library (consider doing this once globally if needed)
        configure_google_genai(api_key=api_key)
        model = GenerativeModel(model_name)
        response = await model.generate_content_async(
            contents=prompt
            # Optional: add generation_config=GenerationConfig(...)
        )
        if response and hasattr(response, "text"):
            content = response.text.strip()
            if not content:
                logging.warning(f"Gemini model {model_name} returned empty content for prompt.")
            return content
        else:
            # Log the full response if possible for debugging, without sensitive parts
            logging.error(f"Failed to get valid text response from Gemini model {model_name}. Response: {response}")
            raise ValueError(f"Invalid response from Gemini model {model_name}")
    except Exception as e:
        logging.error(f"Error during Gemini content generation with model {model_name}: {e}")
        raise


async def extract_tasks_from_report(report_content: str, model_name: str, api_key: str) -> str:
    """
    Uses Gemini API to extract actionable tasks from the UX report template.
    Returns a single string describing the combined tasks.
    """
    logging.info(f"Extracting tasks from report using Gemini model: {model_name}")
    prompt = f"""
    Analyze the following UX report template. Identify the core questions or tasks
    that require navigating and interacting with a website to gather information or test functionality.
    Synthesize these into a single, coherent set of instructions for an AI agent that will browse the website.
    The agent needs to understand what specific information to look for or what actions to perform.

    Example questions it might need to answer:
    - How easy is it to find the contact information?
    - Can a user successfully add a specific product (e.g., 'Blue T-shirt, size M') to the cart?
    - Is the checkout process intuitive?
    - What is the process for signing up for the newsletter?

    Combine these implied actions into a single task description for the agent. Focus on the actions needed.

    Report Template:
    ---
    {report_content}
    ---

    Combined Task Instructions for Web Agent:
    """
    task_description = await generate_content_gemini(prompt, model_name, api_key)
    logging.info(f"Extracted task description:\n{task_description}")
    if not task_description:
        return "Explore the website and gather general usability information based on the provided report structure."  # Fallback
    return task_description


# CORRECTED Browser-Use Agent Initialization
async def run_browser_agent(
    task: str, target_url: str, model_name: str, api_key: SecretStr, headless: bool
) -> AgentHistoryList:
    """
    Initializes and runs the Browser-Use agent.
    """
    logging.info(f"Initializing Browser-Use agent for target URL: {target_url} with model: {model_name}")

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key, temperature=0.0, convert_system_message_to_human=True
        )

        browser_config = BrowserConfig(headless=headless)
        # We pass a browser instance created with the config
        browser = Browser(config=browser_config)

        # Define initial actions to open the target URL
        initial_actions = [{"open_tab": {"url": target_url}}]

        # Pass initial_actions during Agent initialization
        agent = BrowserAgent(
            task=task,
            llm=llm,
            browser=browser,  # Provide the browser instance
            initial_actions=initial_actions,  # Pass initial actions here
            # use_vision=True, # Keep default or set based on needs/model capability
            save_conversation_path="logs/agent_conversation",  # Optional: for debugging
        )

        logging.info("Running Browser-Use agent...")
        # Run the agent WITHOUT passing initial_actions here
        history = await agent.run(max_steps=50)  # Limit steps
        logging.info("Browser-Use agent finished.")

        # Remember to close the browser manually if you provide it to the agent
        await browser.close()
        logging.info("Browser closed.")

        return history

    except Exception as e:
        # Ensure browser is closed even if an error occurs during run
        if "browser" in locals() and browser and not browser.is_closed:
            await browser.close()
            logging.warning("Browser closed due to error during agent execution.")
        logging.error(f"Error during Browser-Use agent execution: {e}")
        raise


async def generate_filled_report(
    report_template: str, agent_history: AgentHistoryList, target_url: str, model_name: str, api_key: str
) -> str:
    """
    Uses Gemini API to fill the report template based on agent's actions and findings.
    """
    logging.info(f"Generating filled report using Gemini model: {model_name}")

    # --- Prepare a summary of the agent's findings ---
    visited_urls = "\n - ".join(agent_history.urls())
    final_outcome = agent_history.final_result() or "Agent did not explicitly extract a final result."
    action_summary = "\n - ".join(agent_history.action_names())

    # Check for errors
    errors = agent_history.errors()
    error_summary = "\n - ".join(map(str, errors)) if errors else "None"

    findings_summary = f"""
    Agent Exploration Summary for Target URL: {target_url}

    Visited URLs:
     - {visited_urls if visited_urls else 'None recorded'}

    Key Actions Performed (Types):
     - {action_summary if action_summary else 'None recorded'}

    Final Extracted Content/Outcome by Agent:
    {final_outcome}

    Errors Encountered During Run:
     - {error_summary}
    """
    # --- Prompt Gemini to fill the report ---
    prompt = f"""
    You are an expert UX Analyst AI. Your task is to fill out the provided UX report template
    based *only* on the summary of actions, findings, and errors from an automated web agent that explored
    the website: {target_url}.

    Use the information in the 'Agent Exploration Summary' to answer the questions in the 'Original Report Template'.
    Be objective and stick to what the agent's summary indicates. If the summary doesn't provide enough information
    to answer a specific question, state that the agent did not gather that specific information or encountered errors preventing it.
    Mention relevant errors if they likely impacted the ability to answer a question.
    Maintain the original structure and questions of the template, providing answers below each question.

    Original Report Template:
    ---
    {report_template}
    ---

    Agent Exploration Summary:
    ---
    {findings_summary}
    ---

    Completed UX Report:
    """
    filled_report = await generate_content_gemini(prompt, model_name, api_key)
    logging.info("Successfully generated filled report content.")
    return filled_report


# --- Main Execution Logic ---


async def main(args):
    """Main function to orchestrate the UX report automation."""
    try:
        # 1. Get API Keys securely
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables or .env file.")

        # Pydantic SecretStr for Langchain/Browser-Use
        gemini_api_key_secret = SecretStr(gemini_api_key)

        # 2. Read the input report template
        logging.info(f"Reading report template from: {args.input_file}")
        report_template = await read_file(args.input_file)

        # 3. Extract tasks using Gemini API
        tasks = await extract_tasks_from_report(report_template, args.gemini_model_task_extraction, gemini_api_key)

        # 4. Run the Browser-Use agent
        agent_history = await run_browser_agent(
            task=tasks,
            target_url=args.target_url,
            model_name=args.gemini_model_agent,
            api_key=gemini_api_key_secret,
            headless=args.headless,
        )

        # 5. Generate the filled report using Gemini API
        completed_report = await generate_filled_report(
            report_template=report_template,
            agent_history=agent_history,
            target_url=args.target_url,
            model_name=args.gemini_model_report_generation,
            api_key=gemini_api_key,
        )

        # 6. Write the completed report
        await write_file(args.output_file, completed_report)

        logging.info("Automated UX reporting process completed successfully.")

    except ValueError as ve:
        logging.error(f"Configuration or Value Error: {ve}")
    except FileNotFoundError as fnfe:
        logging.error(f"File Error: {fnfe}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)  # Log traceback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate UX report filling using Gemini and Browser-Use.")

    parser.add_argument("input_file", type=Path, help="Path to the input UX report template file.")
    parser.add_argument("output_file", type=Path, help="Path to save the completed UX report file.")
    parser.add_argument("target_url", type=str, help="The target website URL for the UX evaluation.")

    # Model Configuration
    parser.add_argument(
        "--gemini-model-task-extraction",
        type=str,
        default="gemini-1.5-flash-latest",
        help="Gemini model for extracting tasks from the report.",
    )
    parser.add_argument(
        "--gemini-model-agent",
        type=str,
        default="gemini-1.5-pro-latest",
        help="Gemini model used by the Browser-Use agent (via Langchain). Needs good tool-calling.",
    )
    parser.add_argument(
        "--gemini-model-report-generation",
        type=str,
        default="gemini-1.5-pro-latest",
        help="Gemini model for generating the final report answers.",
    )

    # Browser-Use options
    parser.add_argument("--headless", action="store_true", help="Run the browser in headless mode (no UI).")

    args = parser.parse_args()

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Run the asynchronous main function
    asyncio.run(main(args))
