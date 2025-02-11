import google.generativeai as genai

# Keep types import - it might be needed for Tool even if not for response Part
from google.generativeai import types
import os
import requests
import json
import time
from dotenv import load_dotenv
import pprint

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)

FASTAPI_BASE_URL = "http://127.0.0.1:8000"
REPORT_FILENAME = "report.txt"

# --- Load Report Template ---
try:
    with open(REPORT_FILENAME, "r", encoding="utf-8") as f:
        report_template_content = f.read()
    print(f"✓ Report template loaded from '{REPORT_FILENAME}'")
except FileNotFoundError:
    print(f"✗ ERROR: Report template file '{REPORT_FILENAME}' not found.")
    exit()
# ... (rest of error handling) ...

# --- Define Function Declarations as Dictionaries ---
# (Keep the dictionary definitions as before - structure is likely correct)
start_browser_func_dict = {
    "name": "start_browser",
    "description": "Starts the Firefox browser.",
    "parameters": {"type": "object", "properties": {}},
}
# ... (include all other function dictionaries: stop_browser, navigate, click, type, scroll, screenshot) ...
stop_browser_func_dict = {
    "name": "stop_browser",
    "description": "Closes the browser.",
    "parameters": {"type": "object", "properties": {}},
}
navigate_func_dict = {
    "name": "navigate_to_url",
    "description": "Navigates to a URL.",
    "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "URL"}},
        "required": ["url"],
    },
}
click_func_dict = {
    "name": "click_element_by_description",
    "description": "Clicks element by description.",
    "parameters": {
        "type": "object",
        "properties": {"description": {"type": "string", "description": "Element text/label"}},
        "required": ["description"],
    },
}
type_func_dict = {
    "name": "type_into_element_by_description",
    "description": "Types text into element.",
    "parameters": {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Input field description"},
            "text_to_type": {"type": "string", "description": "Text to type"},
        },
        "required": ["description", "text_to_type"],
    },
}
scroll_func_dict = {
    "name": "scroll_page",
    "description": "Scrolls the page.",
    "parameters": {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["up", "down", "top", "bottom"], "description": "Scroll direction"}
        },
        "required": ["direction"],
    },
}
screenshot_func_dict = {
    "name": "capture_full_screenshot",
    "description": "Captures full screen screenshot.",
    "parameters": {
        "type": "object",
        "properties": {"filename": {"type": "string", "description": "Filename (e.g. image.png)"}},
        "required": ["filename"],
    },
}
get_text_func_dict = {
    "name": "get_page_text",
    "description": "Retrieves the visible text content from the current web page body. Use this to find specific information like ingredients, dates, contact details, etc.",
    "parameters": {"type": "object", "properties": {}},  # No parameters needed
}

all_function_declarations = [
    start_browser_func_dict,
    stop_browser_func_dict,
    navigate_func_dict,
    click_func_dict,
    type_func_dict,
    scroll_func_dict,
    screenshot_func_dict,
    get_text_func_dict,
]

# --- Prepare Tools Configuration ---
# Use types.Tool if available and correct based on previous steps
try:
    browser_tool = types.Tool(function_declarations=all_function_declarations)
    tools_arg = [browser_tool]
except NameError:  # If types wasn't imported or Tool isn't in it
    print("Warning: types.Tool structure not used, preparing fallback.")
    tools_arg = [{"function_declarations": all_function_declarations}]


# --- Choose and Configure the Gemini Model ---
MODEL_NAME = "models/gemini-1.5-flash-latest"

print(f"\nInitializing Model: {MODEL_NAME}")
model = genai.GenerativeModel(model_name=MODEL_NAME, tools=tools_arg)

chat = model.start_chat(history=[])
print("✓ Model initialized and chat session started.")

# --- Construct the Prompt ---
general_prompt_structure = """
You are an expert website tester simulating actions on a target website using provided tools and then filling out a given usability report template based *only* on your actions and findings.

**Your Goal:** Complete the provided Usability Report template thoroughly and accurately, using the tools persistently to find the required information or complete the task for EACH Use Case.

**Available Tools:** `start_browser`, `stop_browser`, `navigate_to_url`, `click_element_by_description`, `type_into_element_by_description`, `scroll_page`, `capture_full_screenshot`, `get_page_text`.

**General Workflow & Error Handling (CRITICAL - Follow Sequentially):**

**Phase 1: Setup**
1. Call `start_browser`.
2. Navigate to the primary target URL from the report template using `navigate_to_url`.

**Phase 2: Iterative Task Execution & Summarization**
*   Process Tasks Sequentially: Address tasks/Use Cases from the template in order.
*   **For EACH Task/Use Case:**
    a. **Analyze Task Goal:** Understand what this specific task requires.
    b. **Execute Actions Step-by-Step until Goal Met or Exhausted:**
        i.  **Determine Action:** Decide the single function call most likely to achieve the *current step* of the task (e.g., navigate, click link A, get text).
        ii. **Execute:** Issue ONLY that function call.
        iii. **Receive Result:** Get the result ('status', 'message', 'page_text', 'filename_saved', etc.).
        iv. **Analyze Result & Handle Errors:**
            *   **Success:** Note the success. If the result contains `page_text`, **actively search** this text for keywords or information related to the task goal. If the required info is found, the task step is complete. If a `filename_saved` is returned, note it.
            *   **Error (e.g., 'Element not found'):** Note the specific error. Call `get_page_text` (if not just done). Search the returned `page_text` for alternative, related element descriptions (e.g., link text, button text). If a plausible alternative is found, **retry the action in step (ii)** using the *alternative description* as your **one retry attempt** for this specific failed action. If the retry fails or no alternative is found, document the *original specific error* and move to the next logical action for the task, if any.
        v.  **Determine NEXT Action within Task:** Based on the analysis of the result (or documented failure):
            *   If the task's goal is met (e.g., information found in `page_text`, correct page reached): Move to step (c) Task Completion.
            *   If the goal is NOT met, but more actions are needed (e.g., need to scroll down after `get_page_text` showed nothing relevant yet, need to click a different link found in `page_text`, need to try searching after initial navigation): **Loop back to step (i)** to determine and execute the *next relevant action* for **this same task**. Try reasonable actions like scrolling and getting text again, or clicking relevant links found in the text. Attempt 2-3 relevant actions within a task before concluding it cannot be completed.
            *   If the goal is not met and no further reasonable actions can be identified (after trying scrolls, analyzing text, attempting alternatives): Move to step (c) Task Completion and document the specific point of failure.
    c. **Task Completion & Internal Summary:** Once the task goal is met OR reasonable attempts (including recovery) have failed, **mentally summarize** the outcome: successful steps, specific information found (e.g., year, ingredients), documented specific failures, and any screenshot filenames generated *during this task*. Note ratings if applicable to this task.
    d. **Move to Next Task:** Proceed sequentially to the next task/Use Case in the template.

*   **Specific Instruction for UC5 Screenshots:** During your exploration actions in UC5 (navigating, clicking, scrolling), call `capture_full_screenshot` **at least 4 times** at meaningful points in the exploration. Use distinct filenames (e.g., `uc5_section1_main.png`, `uc5_section2_subpage.png`). Remember the exact `filename_saved` values returned.

**Phase 3: Reporting (Assemble Final Output)**
1. Call `stop_browser`.
2. Take the empty report template.
3. **Assemble the Report:** Fill **every placeholder** using your Internal Summaries:
    *   Steps: Describe sequence accurately, including analysis of `page_text` results ("Used get_page_text, searched text, found 'Ingredients:'...") and specific failures ("Attempted click 'History', failed: Element not found. Used get_page_text, found no alternative."). Use natural language.
    *   Screenshots: Insert exact `filename_saved` where required (UC1) and list all generated filenames under the UC5 "Screenshots" heading. Note errors if screenshot failed.
    *   Answer ALL questions and ratings based on the full experience.
4. **Output:** ONLY the completed report text... [rest of constraint]...

**Constraints:** One primary action per turn (plus error recovery calls). Persistently attempt task goals using text analysis and reasonable follow-up actions before concluding failure. Complete ALL tasks sequentially. Summarize internally. Report errors specifically.

Here is the empty report template:
--- START REPORT TEMPLATE ---
{report_content_placeholder}
--- END REPORT TEMPLATE ---
"""
final_prompt = general_prompt_structure.format(report_content_placeholder=report_template_content)

# --- Interaction Loop ---
print("\n=========================================")
print("🚀 Starting Interaction Loop (v0.8.4 Response Structure)")
print("=========================================")
print(f"➤ You (Initial Prompt): [Sending instructions and report template...]")

try:
    response = chat.send_message(final_prompt)
except Exception as e:
    print(f"✗ ERROR: Calling Gemini API on initial prompt failed: {e}")
    exit()

MAX_TURNS = 30
turn_count = 0

while turn_count < MAX_TURNS:
    turn_count += 1
    print(f"\n----------- Turn {turn_count} -----------")
    # Store structured parts to send back, not just raw results
    parts_to_send_back_to_gemini = []

    try:
        # (Keep checks for candidates and parts as before)
        if not response.candidates:
            print("✗ ERROR: No candidates.")
            break
        candidate = response.candidates[0]
        if not candidate.content.parts:
            print("✗ ERROR: No parts.")
            break  # Add text check here later

        # --- Process All Parts in the Response ---
        contains_function_call = False
        final_text_part = None

        for part in candidate.content.parts:
            # --- Check for Function Call ---
            if hasattr(part, "function_call") and part.function_call:
                contains_function_call = True
                function_call = part.function_call
                function_name = function_call.name
                args = dict(function_call.args)

                print(f"➤ Gemini -> Tool:")
                print(f"   Function: {function_name}")
                # ... (print args) ...
                pprint.pprint(args, indent=6)

                # --- Call Local FastAPI Server ---
                # ... (endpoint mapping and FastAPI call logic remains the same) ...
                endpoint_path = function_name
                if function_name == "capture_full_screenshot":
                    endpoint_path = "screenshot"
                elif function_name == "navigate_to_url":
                    endpoint_path = "navigate"
                elif function_name == "click_element_by_description":
                    endpoint_path = "click"
                elif function_name == "type_into_element_by_description":
                    endpoint_path = "type"
                elif function_name == "scroll_page":
                    endpoint_path = "scroll"
                elif function_name == "start_browser":
                    endpoint_path = "start"
                elif function_name == "stop_browser":
                    endpoint_path = "stop"
                elif function_name == "get_page_text":
                    endpoint_path = "get_text"
                api_endpoint = f"{FASTAPI_BASE_URL}/{endpoint_path}"
                print(f"   Calling FastAPI: POST {api_endpoint}")
                api_result = None

                try:
                    # ... (FastAPI request logic remains the same) ...
                    if function_name in ["start_browser", "stop_browser"]:
                        api_response = requests.post(api_endpoint, timeout=30)
                    else:
                        api_response = requests.post(api_endpoint, json=args, timeout=60)
                    api_response.raise_for_status()
                    api_result = api_response.json()  # Get result dict from FastAPI
                    print(f"✓ FastAPI -> Script:")
                    # ... (print result) ...
                    pprint.pprint(api_result, indent=6)
                    if function_name not in ["start_browser", "scroll_page"]:
                        time.sleep(1.5)

                except Exception as e:  # Catch network, JSON, other errors
                    # ... (Error handling remains the same, setting api_result to an error dict) ...
                    error_detail = f"FastAPI call failed for {function_name}: {e}"
                    print(f"✗ ERROR: {error_detail}")
                    api_result = {"status": "error", "message": error_detail}

                # --- Prepare **STRUCTURED** part for sending back ---
                # Create the dictionary structure expected by the SDK for a function response part
                if api_result is not None:
                    function_response_part_dict = {
                        "function_response": {
                            "name": function_name,
                            "response": api_result,  # The dictionary result from FastAPI
                        }
                    }
                    parts_to_send_back_to_gemini.append(function_response_part_dict)
                else:
                    # Optionally send back an error structure if the call failed critically
                    print(f"✗ ERROR: No result for {function_name}, skipping sending back.")
                    # error_part = {"function_response": {"name": function_name,"response": {"status": "error", "message": "Failed to get result from API"}}}
                    # parts_to_send_back_to_gemini.append(error_part)

            # --- Check for Text Part ---
            elif hasattr(part, "text") and part.text:
                final_text_part = part.text  # Store potential final text

        # --- After Processing All Parts ---
        if contains_function_call and parts_to_send_back_to_gemini:
            # Send the list of *structured response dictionaries* back to Gemini
            print(f"\n✓ Sending {len(parts_to_send_back_to_gemini)} structured function result(s) back to Gemini...")
            # *** CORRECTION V0.8.4: Send the list of dictionaries structured as function responses ***
            response = chat.send_message(parts_to_send_back_to_gemini)
            print("✓ Results sent.")

        elif final_text_part:
            # (Handle final text response as before)
            print("\n🏁 Gemini's Final Response (Completed Report):")
            # ... (print and save report) ...
            print("=" * 50)
            print(final_text_part)
            print("=" * 50)
            try:
                with open("completed_report.txt", "w", encoding="utf-8") as f:
                    f.write(final_text_part)
                print(f"\n✓ Completed report saved to 'completed_report.txt'")
            except Exception as e:
                print(f"\n✗ Warning: Could not save completed report: {e}")
            break

        elif not contains_function_call and not final_text_part:
            print("⚠ Warning: No function calls or text found in this turn's response parts.")
            break

    except Exception as e:
        # (Outer loop error handling remains the same)
        print(f"\n✗ ERROR: Unexpected error processing Gemini's response on turn {turn_count}: {e}")
        # (Print problematic response object)
        break

# [ Rest of the script - MAX_TURNS check, final cleanup attempt ]

if turn_count >= MAX_TURNS:
    print("\n✗ ERROR: Reached maximum interaction turns. Process stopped.")

print("\n=========================================")
print("✅ Interaction Loop Finished")
print("=========================================")

# Attempt to gracefully stop the browser if the loop finishes or breaks
# This might fail if stop_browser wasn't called by Gemini or if FastAPI is down
print("\nAttempting final browser cleanup (sending stop_browser)...")
try:
    stop_response = requests.post(f"{FASTAPI_BASE_URL}/stop", timeout=10)
    if stop_response.ok:
        print("✓ Stop request sent successfully.")
    else:
        print(f"⚠ Warning: Stop request sent, but received status {stop_response.status_code}.")
except Exception as e:
    print(f"⚠ Warning: Failed to send stop request during cleanup: {e}")
    print("   (FastAPI server might be down or browser already closed).")
