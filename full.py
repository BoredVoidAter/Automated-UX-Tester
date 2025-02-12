# --- START OF SINGLE FILE SCRIPT ---

import google.generativeai as genai
from google.generativeai import types
import os
import requests
import json
import time
from dotenv import load_dotenv
import pprint
import re
import threading  # To run FastAPI server in background
import uvicorn  # To run FastAPI programmatically

# FastAPI & Selenium related imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException, WebDriverException
from selenium.webdriver.remote.webelement import WebElement
import mss  # For screenshots
import mss.tools
import traceback

# Optional: Use BeautifulSoup for potentially cleaner text extraction
try:
    from bs4 import BeautifulSoup

    USE_BS4 = True
except ImportError:
    USE_BS4 = False
    print(
        "Warning: BeautifulSoup4 not found (pip install beautifulsoup4 lxml). Falling back to basic Selenium text extraction."
    )


# --- Shared Configuration & State ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=API_KEY)

FASTAPI_BASE_URL = "http://127.0.0.1:8000"  # Server address
REPORT_FILENAME = "report.txt"  # Input report template
COMPLETED_REPORT_FILENAME = "completed_report_COMBINED.txt"  # Output file
SCREENSHOT_DIR = "api_screenshots"  # Directory for screenshots
if not os.path.exists(SCREENSHOT_DIR):
    os.makedirs(SCREENSHOT_DIR)

# Global WebDriver instance (managed by FastAPI endpoints)
driver: webdriver.Firefox | None = None

# --- FastAPI Application Setup ---
app = FastAPI(title="Browser Control API", description="API to control a browser via Selenium and take screenshots.")


# --- Pydantic Models for FastAPI Request Bodies ---
class NavigateRequest(BaseModel):
    url: str


class ElementRequest(BaseModel):
    description: str


class TypeRequest(ElementRequest):
    text_to_type: str


class ScrollRequest(BaseModel):
    direction: str


class ScreenshotRequest(BaseModel):
    filename: str


# --- FastAPI Helper Functions ---


def get_safe_current_url(driver_instance):
    """Safely attempts to get the current URL."""
    if not driver_instance:
        return "Browser not running"
    try:
        return driver_instance.current_url
    except WebDriverException as e:
        print(f"⚠ Warning: Error getting current URL (browser might be unresponsive): {e}")
        return f"Error getting URL: {str(e)}"
    except Exception as e:
        print(f"⚠ Warning: Unexpected error getting current URL: {e}")
        return "Error getting URL"


def check_driver_state():
    """Checks if the global driver is initialized and responsive."""
    if not driver:
        print("✗ ERROR (API): Driver object is None or invalid.")
        raise HTTPException(status_code=503, detail="Browser not started or has been closed. Please call /start.")
    try:
        _ = driver.window_handles  # Accessing a property to check connection
    except WebDriverException as e:
        print(f"✗ ERROR (API): Driver seems unresponsive: {e}")
        raise HTTPException(status_code=503, detail=f"Browser session seems unresponsive: {e}")


def find_element_by_description(desc: str) -> WebElement:
    """Attempts to find a web element based on various text strategies."""
    check_driver_state()
    desc_lower = desc.lower()
    element = None
    strategies = [
        (By.LINK_TEXT, desc),
        (By.PARTIAL_LINK_TEXT, desc),
        (By.XPATH, f"//button[normalize-space()='{desc}']"),
        (
            By.XPATH,
            f"//button[contains(translate(normalize-space(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{desc_lower}')]",
        ),
        (By.ID, desc),
        (By.NAME, desc),
        (By.XPATH, f"//input[@placeholder='{desc}' or @aria-label='{desc}']"),
        (By.XPATH, f"//input[contains(@placeholder, '{desc}') or contains(@aria-label, '{desc}')]"),
        (By.XPATH, f"//textarea[@placeholder='{desc}' or @aria-label='{desc}']"),
        (
            By.XPATH,
            f"//*[(self::a or self::button or self::input[@type='submit' or @type='button']) and contains(text(), '{desc}')]",
        ),
        (By.XPATH, f"//*[@title='{desc}' or @aria-label='{desc}']"),
    ]
    print(f"   API: Searching for element described as: '{desc}'")
    for strategy, value in strategies:
        try:
            element = driver.find_element(strategy, value)
            if element.is_displayed():
                print(f"   API: Found potential element using {strategy}: {value}")
                return element
            else:
                print(f"   API: Found element via {strategy} but it's not displayed.")
        except NoSuchElementException:
            continue
        except Exception as e:
            print(f"   API: Error during search with {strategy}='{value}': {e}")
    print(f"   API: Element not found using any strategy for description: '{desc}'")
    raise NoSuchElementException(f"Could not find element based on description: '{desc}'")


# --- FastAPI Endpoints ---


@app.post("/start", summary="Start the Firefox browser")
async def api_start_browser():
    global driver
    if driver:
        try:
            _ = driver.window_handles
            print("✓ API (/start): Browser already running and seems responsive.")
            return {"status": "success", "message": "Browser already running."}
        except WebDriverException:
            print("⚠ API (/start): Stale browser session detected. Closing and restarting.")
            try:
                driver.quit()
            except Exception:
                pass
            driver = None
        except Exception as e:
            print(f"⚠ API (/start): Error checking existing browser state: {e}")
            driver = None  # Reset cautiously
    print("✓ API (/start): Starting Firefox browser...")
    try:
        service = FirefoxService(GeckoDriverManager().install())
        options = webdriver.FirefoxOptions()
        # options.add_argument("--headless")
        driver = webdriver.Firefox(service=service, options=options)
        driver.maximize_window()
        driver.implicitly_wait(5)
        print("✓ API (/start): Browser started successfully.")
        return {"status": "success", "message": "Browser started successfully."}
    except Exception as e:
        print(f"✗ ERROR (API /start): Failed to start browser: {e}")
        print(traceback.format_exc())
        driver = None
        raise HTTPException(status_code=500, detail=f"Failed to start browser: {str(e)}")


@app.post("/stop", summary="Stop the browser")
async def api_stop_browser():
    global driver
    if driver:
        print("✓ API (/stop): Stopping browser...")
        try:
            driver.quit()
            print("✓ API (/stop): Browser stopped successfully.")
            driver = None
            return {"status": "success", "message": "Browser stopped successfully."}
        except Exception as e:
            print(f"✗ ERROR (API /stop): Error stopping browser: {e}")
            driver = None
            return {"status": "error", "message": f"Error stopping browser: {str(e)}"}
    else:
        print("✓ API (/stop): Browser not running.")
        return {"status": "success", "message": "Browser not running."}


@app.post("/navigate", summary="Navigate to URL and attempt to accept cookies")
async def api_navigate(request: NavigateRequest):
    check_driver_state()
    target_url = request.url
    print(f"--- API (/navigate) Request ---")
    print(f"   Target URL: {target_url}")
    try:
        driver.get(target_url)
        print(f"   Page requested. Waiting for load...")
        time.sleep(3)
        cookie_accepted = False
        try:
            print(f"   Looking for cookie acceptance button...")
            possible_selectors = [
                (By.ID, "onetrust-accept-btn-handler"),
                (
                    By.XPATH,
                    "//button[contains(translate(normalize-space(), 'ACGEKLOPRTUVWY', 'acgekloprtuvwy'), 'accept')]",
                ),
                (By.XPATH, "//button[contains(translate(normalize-space(), 'AGRE', 'agre'), 'agree')]"),
                (By.XPATH, "//button[contains(translate(normalize-space(), 'ALOW', 'alow'), 'allow')]"),
                (By.XPATH, "//button[contains(translate(normalize-space(), 'OKGT', 'okgt'), 'ok')]"),
                (By.XPATH, "//button[contains(translate(normalize-space(), 'OKGT', 'okgt'), 'got it')]"),
                (By.CSS_SELECTOR, "[data-testid='accept-button']"),
                (By.CSS_SELECTOR, "button.cookie-accept"),
                (By.CSS_SELECTOR, "button[id*='accept'][id*='cookie']"),
            ]
            accept_button = None
            for strategy, value in possible_selectors:
                try:
                    elements = driver.find_elements(strategy, value)
                    for element in elements:
                        if element.is_displayed() and element.is_enabled():
                            accept_button = element
                            print(f"   Found potential accept button using {strategy}: {value}")
                            break
                    if accept_button:
                        break
                except Exception:
                    continue
            if accept_button:
                try:
                    driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", accept_button
                    )
                    time.sleep(0.5)
                    accept_button.click()
                    cookie_accepted = True
                    print(f"   Clicked cookie acceptance button. Waiting...")
                    time.sleep(2)
                except ElementNotInteractableException as e_interact:
                    print(f"   ⚠ Warning: Cookie button found but not interactable on click: {e_interact}")
                except Exception as e_click:
                    print(f"   ⚠ Warning: Error clicking cookie button: {e_click}")
            else:
                print(f"   Cookie acceptance button not found or interactable.")
        except Exception as e_cookie:
            print(f"   ⚠ Warning: Error during cookie handling: {e_cookie}")

        current_url = get_safe_current_url(driver)
        status_message = f"Navigated to {target_url}"
        if cookie_accepted:
            status_message += " and attempted cookie acceptance."
        else:
            status_message += ". Cookie banner not found or interaction failed."
        print(f"✓ API (/navigate): Navigation complete. Current URL: {current_url}")
        return {"status": "success", "message": status_message, "current_url": current_url}
    except Exception as e:
        print(f"✗ ERROR (API /navigate): Critical error navigating to {target_url}: {e}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "message": f"Critical error navigating: {str(e)}",
            "current_url": get_safe_current_url(driver),
        }


@app.post("/click", summary="Click an element by description")
async def api_click_element(request: ElementRequest):
    check_driver_state()
    current_url_before = get_safe_current_url(driver)
    description = request.description
    print(f"--- API (/click) Request ---")
    print(f"   Attempting click: '{description}'")
    try:
        element = find_element_by_description(description)
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", element)
            time.sleep(0.5)
        except Exception as scroll_e:
            print(f"   ⚠ Warning: Could not scroll element '{description}' into view: {scroll_e}")
        element.click()
        time.sleep(1.5)
        current_url_after = get_safe_current_url(driver)
        print(f"✓ API (/click): Clicked '{description}'. URL changed: {current_url_before != current_url_after}")
        return {"status": "success", "message": f"Clicked element: '{description}'", "current_url": current_url_after}
    except NoSuchElementException:
        err_msg = f"Element not found with description: '{description}'"
        print(f"✗ ERROR (API /click): {err_msg}")
        return {"status": "error", "message": err_msg, "current_url": current_url_before}
    except ElementNotInteractableException:
        err_msg = f"Element '{description}' found but not interactable (hidden/obscured?)."
        print(f"✗ ERROR (API /click): {err_msg}")
        return {"status": "error", "message": err_msg, "current_url": current_url_before}
    except Exception as e:
        err_msg = f"Error clicking '{description}': {str(e)}"
        print(f"✗ ERROR (API /click): {err_msg}")
        print(traceback.format_exc())
        return {"status": "error", "message": err_msg, "current_url": current_url_before}


@app.post("/type", summary="Type text into an element")
async def api_type_into_element(request: TypeRequest):
    check_driver_state()
    current_url = get_safe_current_url(driver)
    description = request.description
    text_to_type = request.text_to_type
    print(f"--- API (/type) Request ---")
    print(f"   Attempting type '{text_to_type}' into '{description}'")
    try:
        element = find_element_by_description(description)
        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});", element)
            time.sleep(0.5)
        except Exception as scroll_e:
            print(f"   ⚠ Warning: Could not scroll element '{description}' into view: {scroll_e}")
        element.clear()
        element.send_keys(text_to_type)
        print(f"✓ API (/type): Typed text into: '{description}'")
        return {"status": "success", "message": f"Typed text into: '{description}'", "current_url": current_url}
    except NoSuchElementException:
        err_msg = f"Element not found for typing: '{description}'"
        print(f"✗ ERROR (API /type): {err_msg}")
        return {"status": "error", "message": err_msg, "current_url": current_url}
    except Exception as e:
        err_msg = f"Error typing into '{description}': {str(e)}"
        print(f"✗ ERROR (API /type): {err_msg}")
        print(traceback.format_exc())
        return {"status": "error", "message": err_msg, "current_url": current_url}


@app.post("/scroll", summary="Scroll the page")
async def api_scroll_page(request: ScrollRequest):
    check_driver_state()
    direction = request.direction.lower()
    print(f"--- API (/scroll) Request ---")
    print(f"   Direction: {direction}")
    try:
        if direction == "down":
            driver.execute_script("window.scrollBy(0, window.innerHeight);")
        elif direction == "up":
            driver.execute_script("window.scrollBy(0, -window.innerHeight);")
        elif direction == "bottom":
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        elif direction == "top":
            driver.execute_script("window.scrollTo(0, 0);")
        else:
            raise ValueError("Invalid scroll direction")
        time.sleep(0.7)
        print(f"✓ API (/scroll): Scrolled {direction}")
        return {"status": "success", "message": f"Scrolled {direction}", "current_url": get_safe_current_url(driver)}
    except Exception as e:
        err_msg = f"Error scrolling {direction}: {str(e)}"
        print(f"✗ ERROR (API /scroll): {err_msg}")
        print(traceback.format_exc())
        return {"status": "error", "message": err_msg, "current_url": get_safe_current_url(driver)}


@app.post("/screenshot", summary="Take a full-screen screenshot using MSS")
async def api_capture_screenshot(request: ScreenshotRequest):
    filename = request.filename
    filepath = os.path.join(SCREENSHOT_DIR, filename)
    abs_filepath = os.path.abspath(filepath)
    print(f"--- API (/screenshot) Request ---")
    print(f"   Filename: {filename}")
    print(f"   Saving to: {abs_filepath}")
    try:
        if not os.path.exists(SCREENSHOT_DIR):
            os.makedirs(SCREENSHOT_DIR)
    except Exception as e:
        print(f"✗ ERROR (API /screenshot): Could not create directory '{SCREENSHOT_DIR}': {e}")
        raise HTTPException(status_code=500, detail=f"Server error creating screenshot directory: {e}")
    try:
        with mss.mss() as sct:
            monitor_number = 1
            if monitor_number >= len(sct.monitors):
                print(
                    f"✗ ERROR (API /screenshot): Monitor {monitor_number} does not exist. Monitors available: {len(sct.monitors)}"
                )
                raise HTTPException(status_code=500, detail=f"Monitor {monitor_number} not found.")
            mon = sct.monitors[monitor_number]
            im = sct.grab(mon)
            mss.tools.to_png(im.rgb, im.size, output=filepath)
        if os.path.exists(filepath):
            print(f"✓ API (/screenshot): Screenshot saved successfully.")
            return {
                "status": "success",
                "message": f"Screenshot saved.",
                "filename_saved": filename,
                "filepath": filepath,
            }
        else:
            print(f"✗ ERROR (API /screenshot): File not found after saving attempt.")
            raise HTTPException(status_code=500, detail="Server error: Screenshot file not found after saving.")
    except Exception as e:
        print(f"✗ ERROR (API /screenshot): Capture/save failed: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Server error during screenshot: {e}")


@app.post("/get_text", summary="Get visible text content of the current page")
async def api_get_page_text():
    check_driver_state()
    current_url = get_safe_current_url(driver)
    print(f"--- API (/get_text) Request ---")
    print(f"   Getting text from: {current_url}")
    try:
        page_text = ""
        if USE_BS4:
            try:
                html_source = driver.page_source
                soup = BeautifulSoup(html_source, "lxml")
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()
                page_text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
                print(f"   API: Extracted text using BeautifulSoup.")
            except Exception as bs4_e:
                print(f"   ⚠ Warning: BeautifulSoup parsing failed ({bs4_e}), falling back to Selenium .text")
                body_element = driver.find_element(By.TAG_NAME, "body")
                page_text = body_element.text
        else:
            body_element = driver.find_element(By.TAG_NAME, "body")
            page_text = body_element.text
            print(f"   API: Extracted text using Selenium body.text.")

        max_len = 6000
        truncated = len(page_text) > max_len
        page_text_to_return = page_text[:max_len]
        print(f"✓ API (/get_text): Retrieved text (truncated: {truncated}). Length: {len(page_text_to_return)}")
        return {
            "status": "success",
            "message": f"Retrieved page text (truncated: {truncated}).",
            "current_url": current_url,
            "page_text": page_text_to_return,
        }
    except NoSuchElementException:
        err_msg = "Could not find body element on page."
        print(f"✗ ERROR (API /get_text): {err_msg}")
        return {"status": "error", "message": err_msg, "current_url": current_url}
    except Exception as e:
        err_msg = f"Server error during get_page_text: {str(e)}"
        print(f"✗ ERROR (API /get_text): {err_msg}")
        print(traceback.format_exc())
        return {"status": "error", "message": err_msg, "current_url": current_url}


# --- Function to run FastAPI server in a thread ---
def run_fastapi_server():
    """Runs the FastAPI app using Uvicorn."""
    print("[FastAPI Server Thread] Starting Uvicorn on http://127.0.0.1:8000...")
    try:
        # Reduce log level to avoid cluttering main thread output
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    except Exception as e:
        print(f"[FastAPI Server Thread] ✗ ERROR running Uvicorn: {e}")
    finally:
        print("[FastAPI Server Thread] Uvicorn finished.")


# --- Gemini Controller Logic ---

# --- LLM Prompt Templates ---

EXTRACTION_PROMPT_TEMPLATE = """
Analyze the provided Usability Report Template text below. Your goal is to identify:
1.  The primary target URL the testing should be performed on.
2.  A list of distinct tasks, Use Cases, or sections that require specific actions (like navigation, clicking, finding information, taking screenshots, answering questions based on interaction).

**Instructions:**
*   Find the target URL (usually mentioned near the beginning, often after 'website you need to test is:'). If none is clearly specified, output "URL_NOT_FOUND".
*   Identify each distinct task/Use Case (often marked like "UC X:" or "Use Case X:"). For each task:
    *   Assign a short, unique identifier (e.g., "UC1", "UC2", "UC3", "UC4", "UC5"). If markers aren't present, use logical IDs like "Task_Find_Contact", "Task_Explore_Section".
    *   Write a concise goal description summarizing what needs to be done for that task, based *only* on the text within that task's section in the template. Include key details mentioned (e.g., "Find climate protection info and screenshot the page.", "Find Ingredients/Nutrition for a product using search/filter.", "Explore two new sections and take 4 screenshots.").
*   Ignore instructional text sections ("How to Fill out...", "Checklist", "Common Mistakes", tool descriptions) unless they contain a specific action task. Ignore overall subjective questions (like Q1-Q3, R3) that will be answered *after* all interactive tasks are done. Focus only on tasks requiring browser interaction or observation *during* the test.
*   **Output Format:** Respond ONLY with a single JSON object containing two keys:
    *   `"target_url"`: A string containing the identified URL (or "URL_NOT_FOUND").
    *   `"tasks"`: A JSON array where each element is an object with two keys: `"id"` (the unique identifier string) and `"goal"` (the concise goal description string). Ensure the task order matches the template.

**Example JSON Output:**
```json
{
  "target_url": "https://www.example.com/test",
  "tasks": [
    {
      "id": "UC1",
      "goal": "Navigate to the 'About Us' page and take a screenshot named 'about_us.png'."
    },
    {
      "id": "UC2",
      "goal": "Find the primary contact email address on the 'Contact' page."
    }
  ]
}```

**Now, analyze this report template:**
--- START REPORT TEMPLATE ---
{report_content_placeholder}
--- END REPORT TEMPLATE ---
"""

TASK_EXECUTION_PROMPT_TEMPLATE = """
You are an expert website tester performing ONLY the following specific task on the target website using the available tools. Focus solely on achieving this task's goal. Assume the browser is already open and navigated to the target site or the relevant page from the previous task.

**Current Task Goal:** {task_goal}

**Available Tools:** `navigate_to_url`, `click_element_by_description`, `type_into_element_by_description`, `scroll_page`, `capture_full_screenshot`, `get_page_text`.

**Procedure:**
1. Determine the single best function call to start or progress on the task goal.
2. Execute ONLY that function call.
3. Analyze the result ('status', 'message', 'page_text', 'filename_saved', etc.).
4. If Error: Note specific error. Use `get_page_text` to understand state. If text reveals an alternative element, try ONE related action (e.g., click different link). If recovery fails/impossible, note original error.
5. If Success: Check result. If `page_text` received, search it for task info. If `filename_saved` received, note it.
6. Decide Next Step for THIS TASK:
    * If goal met: STOP execution & proceed to Output.
    * If goal not met but more actions needed (scroll, click link from text, analyze more text): Go back to step 1 for next action for THIS task. Persist for 2-3 reasonable attempts.
    * If goal not met & attempts exhausted: STOP execution & proceed to Output, noting failure point.
7. **Output:** Once execution stops, provide ONLY this summary:
    * Concise, numbered list of steps taken for *this task* (e.g., "1. Navigated to /abc.", "2. Used get_page_text, found 'XYZ'.", "3. Attempted click 'Link', failed: Element not found."). Include specific errors noted after recovery attempts.
    * If finding specific info was goal, state if found and what (e.g., "Found: 123 Main St", "Could not locate phone number.").
    * If screenshot was goal/part, state filename (e.g., "Screenshot saved: task1.png") or specific error.
    * If Rating associated (implied by template structure for this task ID, e.g., UC3 might imply R1, UC4 might imply R2), provide rating (1-6). Example: "Rating: 5". Infer if necessary.

Do NOT include other commentary.
"""

FINAL_ASSEMBLY_PROMPT_TEMPLATE = """
You are assembling a final Usability Report based on summaries from previously attempted tasks.

**Task:** Fill in the original empty report template below completely and accurately using ONLY the provided task summaries. Also answer the overall subjective questions based on the collective experience.

**Instructions:**
1. Review the **Collected Task Summaries** provided below. Each summary corresponds to a specific task/Use Case from the original template (e.g., summary for 'UC1' goes into the UC1 section).
2. Take the **Original Empty Report Template**.
3. Fill in each section (UC1, UC2, UC3, UC4, UC5 sections, Rating questions R1, R2): Use the corresponding Task Summary to insert the generated steps, findings, ratings, and screenshot filenames exactly as reported in the summary. Match the Task ID (e.g., "UC1") to the section header in the template. If a summary indicates failure, report that specific failure using the exact error message provided in the summary.
4. Screenshot Placeholders: For UC1's placeholder `[Screenshot: ...]`, insert the filename reported in the UC1 summary. For the UC5 "Screenshots" section, list all distinct screenshot filenames mentioned in the UC5 summary. If a summary reported a screenshot error, write the error message instead.
5. Open Questions (Q1, Q2, Q3) & Overall Rating (R3, R3.1): Synthesize answers based on the overall experience reflected *across all* task summaries. Use specific examples of success/failure mentioned in the summaries. Be objective based *only* on the provided summaries.
6. Fill other sections ("Any Bugs?", etc.) appropriately based on the summaries (likely "No" for bugs unless summaries mention specific issues found that seem like bugs).
7. Ensure ALL placeholders corresponding to the attempted tasks or overall questions are filled. Use "N/A" or indicate if a task could not be completed based on the summary, but avoid leaving template placeholders like `[...]`.
8. Output MUST be ONLY the completed report text, starting with "--- START OF FILE report.txt ---" and ending with "--- END OF FILE report.txt ---". No extra commentary.

**Original Empty Report Template:**
--- START REPORT TEMPLATE ---
{report_content_placeholder}
--- END REPORT TEMPLATE ---

**Collected Task Summaries:**
--- TASK SUMMARIES START ---
{task_summaries_placeholder}
--- TASK SUMMARIES END ---
"""


# --- LLM Task Identification Function ---
def llm_extract_tasks(template_content, model_instance):
    """Uses LLM to parse template and extract tasks."""
    print("\n--- Identifying Tasks via LLM ---")
    extraction_prompt = EXTRACTION_PROMPT_TEMPLATE.format(report_content_placeholder=template_content)
    # Use a model instance configured *without* tools for simple generation
    extraction_model = genai.GenerativeModel(model_name=MODEL_NAME)

    try:
        response = extraction_model.generate_content(extraction_prompt)
        # Robustly extract text, checking for safety blocks etc.
        extracted_json_text = ""
        if not response.candidates:
            print("✗ ERROR: LLM Extraction - No candidates returned.")
            return [], "URL_NOT_FOUND"
        candidate = response.candidates[0]
        if hasattr(candidate, "finish_reason") and candidate.finish_reason.name == "SAFETY":
            print(
                f"✗ ERROR: LLM Extraction - Blocked by safety settings: {getattr(candidate, 'safety_ratings', 'No details')}"
            )
            return [], "URL_NOT_FOUND"
        if hasattr(candidate.content, "parts") and candidate.content.parts:
            if hasattr(candidate.content.parts[0], "text"):
                extracted_json_text = candidate.content.parts[0].text
            else:
                print("⚠ Warning: LLM Extraction - First part has no text attribute.")
                return [], "URL_NOT_FOUND"  # Cannot proceed without text
        elif hasattr(candidate.content, "text"):
            extracted_json_text = candidate.content.text
        else:
            print("✗ ERROR: LLM Extraction - No parts or text found in candidate content.")
            pprint.pprint(response)
            return [], "URL_NOT_FOUND"

        print(f"  Raw extraction response text (first 500 chars):\n{extracted_json_text[:500]}...")

        # Attempt to clean potential markdown ```json ... ``` blocks or isolate JSON object
        json_match = re.search(r"```json\s*([\s\S]+?)\s*```", extracted_json_text)
        if json_match:
            extracted_json_text = json_match.group(1)
        else:
            start_index = extracted_json_text.find("{")
            end_index = extracted_json_text.rfind("}")
            if start_index != -1 and end_index != -1 and end_index > start_index:
                extracted_json_text = extracted_json_text[start_index : end_index + 1]
            else:
                print("⚠ Warning: Could not isolate JSON object in response text.")

        data = json.loads(extracted_json_text)

        # Basic Validation
        if (
            not isinstance(data, dict)
            or "target_url" not in data
            or "tasks" not in data
            or not isinstance(data["tasks"], list)
        ):
            print("✗ ERROR: LLM extraction response JSON structure is invalid.")
            return [], "URL_NOT_FOUND"

        # Deeper Task Validation
        tasks = []
        for i, task_item in enumerate(data["tasks"]):
            if isinstance(task_item, dict) and "id" in task_item and "goal" in task_item:
                task_item["id"] = re.sub(r"\W+", "_", task_item["id"])  # Sanitize ID
                tasks.append(task_item)
            else:
                print(f"⚠ Warning: Invalid task structure at index {i} from LLM: {task_item}")

        target_url = data["target_url"]
        if not isinstance(target_url, str) or not target_url or target_url == "URL_NOT_FOUND":
            print("⚠ Warning: LLM did not return a valid target URL. Falling back.")
            url_match_fallback = re.search(
                r"website.*?test is:\s*(http[s]?://[^\s\n]+)", template_content, re.IGNORECASE
            )
            target_url = url_match_fallback.group(1).strip() if url_match_fallback else "https://example.com/fallback"
            print(f"   Using fallback URL: {target_url}")

        if not tasks:
            print("⚠ Warning: LLM did not identify any tasks.")
        print(f"✓ LLM identified {len(tasks)} tasks. Target URL: {target_url}")
        return tasks, target_url

    except json.JSONDecodeError as e:
        print(f"✗ ERROR: Failed to decode JSON from LLM: {e}\nText was:{extracted_json_text}")
        return [], "URL_NOT_FOUND"
    except Exception as e:
        print(f"✗ ERROR: Unexpected error during LLM task extraction: {e}")
        try:
            pprint.pprint(response)
        except:
            print("(Could not print response object)")
        return [], "URL_NOT_FOUND"


# --- Function to Execute a Single Task ---
def execute_single_task(task_chat, initial_prompt):
    """Handles the interaction loop for one task and returns the summary text."""
    try:
        print(f"  Sending task execution prompt to Gemini...")
        response = task_chat.send_message(initial_prompt)
    except Exception as e:
        print(f"  ✗ ERROR: Initial Gemini call for task failed: {e}")
        return f"Error: Initial model call failed for task: {str(e)}"

    task_turn = 0
    MAX_TASK_TURNS = 15  # Limit turns per individual task

    while task_turn < MAX_TASK_TURNS:
        task_turn += 1
        print(f"\n  -- Task Turn {task_turn} --")
        api_result = None
        structured_response_parts = []  # List to hold parts structured for sending back

        try:
            # --- Process Gemini Response ---
            if not response.candidates:
                return "Error: No candidates in response."
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason.name

            # Handle safety blocks or unexpected stops
            if finish_reason not in ["STOP", "MAX_TOKENS", "FUNCTION_CALL", "UNSPECIFIED"]:
                if finish_reason == "SAFETY":
                    safety_info = getattr(candidate, "safety_ratings", "No details")
                    print(f"  ✗ ERROR: Response blocked by safety settings: {safety_info}")
                    return f"Error: Response blocked by safety settings."
                else:
                    print(f"  ⚠ Warning: Unexpected finish reason '{finish_reason}'.")

            if not hasattr(candidate.content, "parts") or not candidate.content.parts:
                if hasattr(candidate.content, "text") and candidate.content.text:
                    return candidate.content.text
                if finish_reason == "STOP":
                    return "Task completed with no textual summary output."
                print("✗ ERROR: Response candidate has no parts and no fallback text.")
                pprint.pprint(response)
                return "Error: Response candidate has no parts."

            # --- Process Parts ---
            contains_function_call = False
            final_text_part = None  # Store potential text summary

            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    contains_function_call = True
                    function_call = part.function_call
                    function_name = function_call.name
                    args = dict(function_call.args)

                    print(f"  ➤ Gemini -> Tool:")
                    print(f"     Function: {function_name}")
                    pprint.pprint(args, indent=8)

                    # Call Local FastAPI Server
                    endpoint_path = function_name
                    # (Endpoint mapping)
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
                    else:
                        print(f"   ⚠ Warning: Unknown function '{function_name}' called by LLM.")  # Handle unknown

                    api_endpoint = f"{FASTAPI_BASE_URL}/{endpoint_path}"
                    print(f"     Calling FastAPI: POST {api_endpoint}")
                    api_result = None
                    try:
                        # Determine method/body based on function name
                        if function_name in ["start_browser", "stop_browser", "get_page_text"]:
                            api_response = requests.post(api_endpoint, timeout=60)
                        # Check if function expects args based on our definition map
                        elif function_name in all_function_declarations_map and all_function_declarations_map[
                            function_name
                        ].get("parameters", {}).get("properties"):
                            api_response = requests.post(api_endpoint, json=args, timeout=60)
                        # Handle known functions that don't expect args but weren't caught above
                        elif function_name in all_function_declarations_map:
                            api_response = requests.post(api_endpoint, timeout=60)
                        else:  # Handle unknown function call attempt if needed
                            print(f"   Skipping API call for unknown function: {function_name}")
                            api_result = {
                                "status": "error",
                                "message": f"Unknown function '{function_name}' requested.",
                            }

                        # Only process response if api_result is still None (meaning call was made)
                        if api_result is None:
                            api_response.raise_for_status()
                            api_result = api_response.json()
                            print(f"  ✓ FastAPI -> Script:")
                            pprint.pprint(api_result, indent=8)
                            # Add delay for non-query actions
                            if function_name not in ["scroll_page", "get_page_text", "start_browser", "stop_browser"]:
                                time.sleep(1.5)

                    except requests.exceptions.RequestException as e:
                        error_detail = f"FastAPI call failed for {function_name}: {str(e)}"
                        print(f"  ✗ ERROR: {error_detail}")
                        api_result = {"status": "error", "message": error_detail}
                    except json.JSONDecodeError as e:
                        error_detail = (
                            f"FastAPI response not valid JSON from {endpoint_path}. Text: {api_response.text[:200]}..."
                        )
                        print(f"  ✗ ERROR: {error_detail}")
                        api_result = {"status": "error", "message": error_detail}
                    except Exception as e:
                        error_detail = f"Unexpected error during FastAPI call for {function_name}: {e}"
                        print(f"  ✗ ERROR: {error_detail}")
                        api_result = {"status": "error", "message": error_detail}

                    # --- Prepare structured part for sending back ---
                    if api_result is not None:
                        structured_response_parts.append(
                            {"function_response": {"name": function_name, "response": api_result}}
                        )
                    else:  # Should not happen if error handling above works, but as safeguard
                        print(f"  ✗ ERROR: No result captured for {function_name}, sending internal error back.")
                        structured_response_parts.append(
                            {
                                "function_response": {
                                    "name": function_name,
                                    "response": {
                                        "status": "error",
                                        "message": "Internal script error: No result obtained from API call",
                                    },
                                }
                            }
                        )

                elif hasattr(part, "text") and part.text:
                    # Store potential summary text
                    final_text_part = part.text

            # --- After Processing All Parts for this turn ---
            if contains_function_call and structured_response_parts:
                # If function calls were made, send results back for next step
                print(f"  ✓ Sending {len(structured_response_parts)} function result(s) back to Gemini...")
                response = task_chat.send_message(structured_response_parts)  # Send structured list
                print("  ✓ Results sent.")
                # Continue loop

            elif final_text_part and finish_reason == "STOP":
                # If we got text AND the model stopped, assume it's the summary
                print("  ✓ Received final text summary for task.")
                return final_text_part

            elif finish_reason == "STOP":  # Model stopped without function call or text part
                print("  ⚠ Warning: Finish reason STOP but no usable output found in parts.")
                return "Task completed with no textual summary output."

            elif not contains_function_call and not final_text_part:
                # If no function calls were handled and no text received, it's an issue
                print("  ⚠ Warning: No function calls or text found in this turn's response parts.")
                return "Error: Model returned no action or text for task."

        except Exception as e:
            print(f"\n  ✗ ERROR: Unexpected error processing Gemini response on task turn {task_turn}: {e}")
            try:
                pprint.pprint(response)
            except Exception:
                print("(Could not dump response object)")
            return f"Error: Unexpected error processing response: {str(e)}"  # Task failed

    # If loop finishes due to max turns
    print(f"✗ ERROR: Reached max turns ({MAX_TASK_TURNS}) for task.")
    return f"Error: Task failed to complete within max turns."


# --- Main Execution Flow ---
if __name__ == "__main__":
    print("--- Starting Automated Tester ---")

    # --- Start FastAPI Server in Background Thread ---
    server_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    server_thread.start()
    print("✓ FastAPI server thread started. Waiting for server startup...")
    time.sleep(7)  # Increased wait time slightly for server init

    # 1. Load Template
    report_template = load_report_template(REPORT_FILENAME)

    # 2. Identify Tasks & Target URL using LLM
    # Initialize main model with tools (needed for task execution later)
    try:
        # Pass tools arg during init
        model = genai.GenerativeModel(model_name=MODEL_NAME, tools=tools_arg)
        print("✓ Main Model (with tools) initialized.")
    except Exception as model_init_e:
        print(f"✗ CRITICAL ERROR: Failed to initialize main Gemini Model: {model_init_e}")
        exit()

    # Create a map for faster lookup later if needed
    all_function_declarations_map = {f["name"]: f for f in all_function_declarations}

    tasks_to_do, target_url = llm_extract_tasks(report_template, model)  # Pass initialized model
    task_results = {}  # Stores summaries keyed by task ID

    if not tasks_to_do or not target_url.startswith("http"):
        print("✗ CRITICAL ERROR: Could not identify tasks or valid target URL. Exiting.")
        # Attempt to stop server if it started
        try:
            requests.post(f"{FASTAPI_BASE_URL}/stop", timeout=5)
        except Exception:
            pass
        exit()

    # 3. Start Browser
    print("\n--- Executing Setup Task: start_browser ---")
    # Use a clean chat for this setup task, using the main model with tools
    start_result = execute_single_task(model.start_chat(history=[]), "Call the `start_browser` function.")
    # Check result more carefully - assuming success message is specific
    if (
        "error" in start_result.lower()
        or "failed" in start_result.lower()
        or ("already running" not in start_result.lower() and "started successfully" not in start_result.lower())
    ):
        print(f"✗ CRITICAL ERROR starting browser: {start_result}")
        exit()
    print(f"✓ Browser start command executed.")

    # 4. Initial Navigation
    print(f"\n--- Executing Setup Task: Initial Navigation to {target_url} ---")
    nav_prompt = f"Call `navigate_to_url` to go to the target URL: {target_url}"
    # Use the main model with tools for subsequent tasks
    nav_result = execute_single_task(model.start_chat(history=[]), nav_prompt)
    if "error" in nav_result.lower() or "failed" in nav_result.lower() or "navigated" not in nav_result.lower():
        print(f"✗ CRITICAL ERROR during initial navigation: {nav_result}")
        # Attempt to stop browser before exiting
        print("\nAttempting emergency browser stop...")
        execute_single_task(model.start_chat(history=[]), "Call `stop_browser` function.")
        exit()
    print(f"✓ Initial navigation command executed.")

    # 5. Loop Through Dynamically Identified Tasks
    for task_info in tasks_to_do:
        task_id = task_info["id"]  # Use the ID extracted by LLM
        task_goal = task_info["goal"]
        print(f"\n--- Executing Task: {task_id} ---")
        print(f"   Goal: {task_goal}")

        current_task_prompt = TASK_EXECUTION_PROMPT_TEMPLATE.format(task_goal=task_goal)
        # Start fresh chat for task isolation, using the main model instance with tools
        task_summary = execute_single_task(model.start_chat(history=[]), current_task_prompt)
        task_results[task_id] = task_summary  # Store the summary string
        print(f"--- Task {task_id} Finished. Result Summary: ---")
        print(task_summary)
        print("-" * 40)

    # 6. Stop Browser
    print("\n--- Executing Cleanup Task: stop_browser ---")
    stop_result = execute_single_task(model.start_chat(history=[]), "Call the `stop_browser` function.")
    print(f"✓ Browser stop command executed.")

    # 7. Assemble Final Report
    print("\n--- Assembling Final Report ---")
    formatted_summaries = ""
    for task_id, summary in task_results.items():
        # Use the task ID from the LLM extraction (e.g., "UC1", "Task_Find_Email")
        # Make ID more readable for the prompt context
        readable_task_id = task_id.replace("_", " ").replace("UC", "Use Case ")
        formatted_summaries += f"**Summary for Task '{readable_task_id}':**\n{summary}\n\n---\n"

    final_assembly_prompt = FINAL_ASSEMBLY_PROMPT_TEMPLATE.format(
        report_content_placeholder=report_template, task_summaries_placeholder=formatted_summaries
    )

    # Initialize a model *without* tools for the final assembly (pure text gen)
    assembly_model = genai.GenerativeModel(model_name=MODEL_NAME)
    assembly_chat = assembly_model.start_chat(history=[])
    try:
        print("➤ Sending final assembly request to Gemini...")
        # Configure for potentially long output
        generation_config_assembly = genai.GenerationConfig(
            max_output_tokens=8192, temperature=0.5  # Maximize output tokens  # Lower temp for factual assembly
        )
        final_response = assembly_chat.send_message(
            final_assembly_prompt, generation_config=generation_config_assembly
        )

        # Extract final text safely
        final_report_text = f"Error: Could not extract final report text. Finish Reason: {final_response.candidates[0].finish_reason.name if final_response.candidates else 'N/A'}"  # Default error
        if final_response.candidates:
            candidate = final_response.candidates[0]
            # Check finish reason first
            finish_reason = candidate.finish_reason.name
            if finish_reason == "STOP" or finish_reason == "MAX_TOKENS":
                if (
                    hasattr(candidate.content, "parts")
                    and candidate.content.parts
                    and hasattr(candidate.content.parts[0], "text")
                ):
                    final_report_text = candidate.content.parts[0].text
                elif hasattr(candidate.content, "text"):  # Fallback if no parts but text exists
                    final_report_text = candidate.content.text
                else:
                    # If STOP/MAX_TOKENS but no text, report that
                    final_report_text = f"Report assembly finished ({finish_reason}) but no text content was found."
            else:
                # Handle safety or other non-STOP reasons
                final_report_text = f"Error: Report assembly failed or was blocked. Finish Reason: {candidate.finish_reason.name}. Safety Ratings: {getattr(candidate, 'safety_ratings', 'N/A')}"

        print("\n🏁 Gemini's Final Assembled Report:")
        print("=" * 50)
        print(final_report_text)
        print("=" * 50)

        # Save final report
        with open(COMPLETED_REPORT_FILENAME, "w", encoding="utf-8") as f:
            f.write(final_report_text)
        print(f"\n✓ Completed report saved to '{COMPLETED_REPORT_FILENAME}'")

    except Exception as e:
        print(f"✗ ERROR during final report assembly: {e}")
        try:
            pprint.pprint(final_response)
        except:
            print("(Could not print final response object)")

    print("\n=========================================")
    print("✅ Script Finished")
    print("=========================================")
    # Server thread is daemon, will exit when main thread exits.

# --- END OF SINGLE FILE SCRIPT ---
