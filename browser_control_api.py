# --- START OF REVISED v2.py ---

import os
import asyncio
import traceback  # For better error printing
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from browser_use import Agent, Browser
except ImportError:
    print("ERROR: Please install the 'browser-use' library (pip install browser-use)")
    exit()

# --- Configuration ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

REPORT_FILENAME = "report.txt"
COMPLETED_REPORT_FILENAME = "completed_report_BROWSER_USE_v2.txt"
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"  # Or try "gemini-1.5-pro-latest" if flash struggles


# --- Load Report Template ---
def load_report_template(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"✓ Report template loaded from '{filename}'")
        return content
    except FileNotFoundError:
        print(f"✗ ERROR: Report template file '{filename}' not found.")
        exit()
    except Exception as e:
        print(f"✗ ERROR: Could not read {filename}: {e}")
        exit()


# --- Create the Comprehensive Task Prompt (with strengthened output instruction) ---
def create_main_task_prompt(template_content):
    # Extract target URL (simple regex for prompt context)
    target_url_match = re.search(
        r"(?:website|site) you need to test is:\s*(http[s]?://[^\s\n]+)", template_content, re.IGNORECASE
    )
    target_url = target_url_match.group(1).strip() if target_url_match else "http://www.benjerry.co.uk/"  # Default

    main_prompt = f"""
**Overall Objective:**
Act as an expert usability tester. Your goal is to interact with the website located at **{target_url}** according to the tasks described in the **Usability Report Template** provided below. After performing all interactions, your **SOLE AND FINAL TASK** is to generate the **completed report** as your output, filling in all sections based on your actions and observations.

**General Instructions (Action Phase):**
1.  Analyze the Template: Understand all Use Cases (UC1-UCN) and required actions.
2.  Execute Tasks Sequentially: Perform actions for UC1, then UC2, etc.
3.  Use Browser Tools: Interact using available tools (navigate, click elements identified by text/description/selector, type text, scroll, get page content/text, take screenshots). Handle cookie banners proactively if they appear.
4.  Persistence & Error Recovery: If actions fail (e.g., element not found, click intercepted), do not give up immediately. Analyze the current page content/text. Try scrolling, looking for alternative links/buttons with similar names, or refining your element description. Attempt 1-2 reasonable recovery actions before concluding a specific step failed.
5.  Information Gathering: Use content/text retrieval tools to find specific info needed for tasks (ingredients, year, contact details).
6.  Screenshots: Use the screenshot tool when required, generating logical filenames (e.g., `uc1_climate.png`, `uc5_section1.png`). Remember the exact filenames.

**Final Output Generation Phase (CRITICAL - Perform AFTER all browser actions):**
1.  Review all actions taken and information gathered during the interaction phase for UC1-UCN.
2.  Take the original empty report template provided below.
3.  Fill in **every section** accurately based *only* on the browser interactions performed:
    *   **Steps:** Describe actions/failures in natural language.
    *   **Screenshots:** Insert exact filenames generated or specific error messages.
    *   **Ratings:** Fill in R1, R2, R3 based on simulated experience.
    *   **Open Questions:** Answer Q1, Q2, Q3 citing specific examples from interactions. Fill R3.1 if needed.
    *   Fill remaining fields (e.g., "Any Bugs?" -> "No").
4.  Your ABSOLUTE FINAL output for this entire task MUST BE ONLY the completed report text, starting exactly with "--- START OF FILE report.txt ---" and ending exactly with "--- END OF FILE report.txt ---".
5.  **DO NOT** output any other text, summaries of actions, confirmations, comments, or code snippets before or after the completed report block.

**Usability Report Template to Complete:**
--- START REPORT TEMPLATE ---
{template_content}
--- END REPORT TEMPLATE ---
"""
    return main_prompt


# --- Main Execution ---
async def main():
    print("--- Starting Automated Tester with browser-use ---")
    browser = None  # Initialize browser variable for finally block

    try:
        # 1. Load Template
        report_template = load_report_template(REPORT_FILENAME)

        # 2. Create Main Task Prompt
        task_prompt = create_main_task_prompt(report_template)

        # 3. Initialize Browser and LLM (via LangChain)
        print("\n--- Initializing Browser and LLM Agent ---")
        browser = Browser()  # Initialize browser-use browser instance

        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_NAME,
            temperature=0.3,  # Keep temp low for focus
            # convert_system_message_to_human=True # Only if needed
        )

        agent = Agent(
            task=task_prompt,
            llm=llm,
            browser=browser,
            # max_iterations=30 # Optional: Limit agent turns
        )
        print("✓ Browser and Agent initialized.")

        # 4. Run the Agent and Capture History
        print("\n--- Running Agent (This may take some time)... ---")
        # *** CAPTURE the return value of agent.run() ***
        history = await agent.run()
        print("\n✓ Agent run finished.")

        # 5. Extract Final Result from History
        final_result = None
        if history:
            print("--- Analyzing Agent History for Final Output ---")
            final_result = history.final_result()
        else:
            print("  ⚠ Warning: agent.run() did not return a history object.")

        # 6. Process and Save Final Result
        if final_result:
            # Attempt to clean up potential "Okay, here is the report..." preamble
            report_start_marker = "--- START OF FILE report.txt ---"
            start_index = final_result.find(report_start_marker)
            if start_index != -1:
                final_result = final_result[start_index:]

            print("\n🏁 Agent's Final Output (Potential Completed Report):")
            print("=" * 50)
            print(final_result)
            print("=" * 50)
            with open(COMPLETED_REPORT_FILENAME, "w", encoding="utf-8") as f:
                f.write(final_result)
            print(f"\n✓ Completed report saved to '{COMPLETED_REPORT_FILENAME}'")
        else:
            print(
                "\n✗ ERROR: Could not automatically extract the final completed report from the agent's execution history."
            )
            print(
                "--- Please review the full console output above for the agent's interactions and potential report text. ---"
            )

    except Exception as e:
        print(f"✗ ERROR during agent run or processing: {e}")
        print(traceback.format_exc())  # Print full traceback

    finally:
        # 7. Close Browser
        print("\n--- Closing Browser ---")
        # input('Press Enter to close the browser...') # Remove user wait for automation
        if browser:
            try:
                await browser.close()
                print("✓ Browser closed.")
            except Exception as close_e:
                print(f"  ⚠ Warning: Error closing browser: {close_e}")
        else:
            print("  Browser instance not found for closing.")


if __name__ == "__main__":
    asyncio.run(main())

# --- END OF SINGLE FILE SCRIPT ---
