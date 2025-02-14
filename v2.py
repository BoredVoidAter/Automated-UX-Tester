import asyncio
import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain_openai import ChatOpenAI # Optional
from browser_use import Agent as BrowserAgent, Browser, BrowserConfig, BrowserContextConfig
from pydantic import SecretStr
import logging

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Optional

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# --- Constants ---
INPUT_REPORT_FILE = "report.txt"
OUTPUT_REPORT_FILE = "report_output.txt"
SCREENSHOTS_DIR = Path("screenshots")
MAX_RETRIES = 2
# Initial task for agent before starting questions (can be simple navigation)
INITIAL_AGENT_TASK = "Navigate to {target_url} and confirm the page title. Output 'Ready.' when done."


# Ensure screenshots directory exists
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper Functions ---
# (read_report, get_target_website, parse_report_with_gemini,
#  crosscheck_report_with_gemini, generate_output_report remain the same
#  as the previous *full* script version. Make sure they are included here.)
# --- Placeholder for Helper Functions (Keep implementations from previous full script) ---
def read_report(filename: str) -> str:
    """Reads the content of the report file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Error: Input report file '{filename}' not found.")
        return ""
    except Exception as e:
        logging.error(f"Error reading report file '{filename}': {e}")
        return ""


def get_target_website() -> str:
    """Gets the target website URL from the user."""
    url = input("Please enter the target website URL: ")
    if not url.startswith(("http://", "https://")):
        if "." in url:
            logging.warning("URL might be missing http:// or https://. Adding https://")
            url = "https://" + url
        else:
            logging.error("Invalid URL format provided.")
            return ""
    return url


def prompt_user_for_cookie_handling(target_url: str):
    """Uses console input to pause execution for manual cookie handling."""
    print("\n" + "=" * 50)
    print(">>> ACTION REQUIRED <<<")
    print(f"\nA browser window should now be open (navigating to: {target_url}).")
    print(
        "Please switch to the browser window and MANUALLY handle any cookie consent banners or popups (e.g., click 'Accept All')."
    )
    input("\n>>> Press Enter here in the console ONLY AFTER you have handled cookies on the website... <<<")
    print("=" * 50 + "\n")
    logging.info("User pressed Enter, proceeding with automated tasks.")


def parse_report_with_gemini(report_content: str, llm: ChatGoogleGenerativeAI) -> list[dict]:
    """Uses Gemini to parse the report."""
    logging.info("Parsing report with Gemini...")
    prompt = f"""
    Analyze the following usability report text meticulously. Identify each distinct question, task, confirmation, rating scale, or section requiring user input based on website interaction.

    For each identified item, determine its primary response type. Possible types:
    - 'confirmation': Simple yes/no or checkbox confirmation.
    - 'screenshot': Requires ONLY a screenshot.
    - 'text': Requires a step-by-step textual description.
    - 'text_and_screenshot': Requires BOTH text steps AND one or more screenshots.
    - 'rating': Requires selecting a rating. Extract the full question text including the scale.
    - 'exploration': Requires exploring sections, describing steps textually, AND providing multiple screenshots. Note if multiple screenshots are needed.
    - 'open_feedback': Requires open-ended textual feedback.
    - 'other': If none of the above fit perfectly.

    Extract the following for each item:
    1.  'id': A unique identifier (e.g., "Confirmation1", "UC1", "UC2", "UC3", "R1", "UC4", "R2", "UC5", "ScreenshotsUC5", "Q1", "Q2", "Q3", "R3", "R3.1", "BugsConfirmation"). Infer from headings. Make sequential IDs if none obvious. Use specific ID for UC5 screenshots.
    2.  'question_text': The core question, instruction, or task statement.
    3.  'type': The determined response type.
    4.  'requires_multiple_screenshots': boolean (true/false).
    5.  'full_section_text': The complete text of the section for context.

    Ignore generic instructions, checklists (unless part of task), examples (unless the task), IGNORE_WHEN_COPYING blocks, AI warnings. Focus on actionable items.

    Return the result as a valid JSON list of objects. Each object must contain all 5 keys ('id', 'question_text', 'type', 'requires_multiple_screenshots', 'full_section_text').

    Report Content:
    ```text
    {report_content}
    ```

    JSON Output:
    """
    try:
        response = llm.invoke(prompt)
        content = response.content
        logging.debug(f"Gemini Raw Parse Response: {content}")
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        if json_match:
            json_string = json_match.group(1)
        else:
            first_brace = -1
            last_brace = -1
            json_string = content.strip()
            if not json_string.startswith("["):
                first_brace = json_string.find("[")
            if first_brace != -1:
                json_string = json_string[first_brace:]
            if not json_string.endswith("]"):
                last_brace = json_string.rfind("]")
            if last_brace != -1:
                json_string = json_string[: last_brace + 1]
            else:
                json_string = "[]"

        try:
            parsed_questions = json.loads(json_string)
        except json.JSONDecodeError as e:
            logging.error(f"Primary JSON decoding failed: {e}. Attempting extraction.")
            list_match = re.search(r"(\[[\s\S]*?\])", json_string, re.DOTALL)
            if list_match:
                json_string = list_match.group(1)
                try:
                    parsed_questions = json.loads(json_string)
                except json.JSONDecodeError as e2:
                    raise ValueError("Could not decode JSON list.") from e2
            else:
                raise ValueError("Could not find JSON list structure.") from e

        if not isinstance(parsed_questions, list):
            raise ValueError("Did not return JSON list.")

        validated_questions = []
        for i, q in enumerate(parsed_questions):
            if not isinstance(q, dict):
                continue
            q.setdefault("id", f"Task{i+1}")
            q.setdefault("question_text", "[PARSE ERROR]")
            q.setdefault("type", "other")
            q.setdefault("requires_multiple_screenshots", False)
            q.setdefault("full_section_text", q.get("question_text", ""))
            required_keys = ["id", "question_text", "type", "requires_multiple_screenshots", "full_section_text"]
            if not all(k in q for k in required_keys):
                logging.warning(f"Q ID {q['id']} missing keys: {q}")
            q["answer"] = None
            q["screenshot_paths"] = []
            validated_questions.append(q)
        logging.info(f"Parsed and validated {len(validated_questions)} questions.")
        return validated_questions
    except Exception as e:
        logging.error(f"Error during Gemini report parsing: {e}", exc_info=True)
        return []


def crosscheck_report_with_gemini(parsed_questions: list[dict], llm: ChatGoogleGenerativeAI) -> list[dict]:
    """Uses Gemini to identify questions needing retry."""
    logging.info("Cross-checking report...")
    report_summary = ""
    # ... (Build summary as before) ...
    for i, q in enumerate(parsed_questions):
        q_id = q.get("id", f"Task{i+1}")
        answer = q.get("answer", "")
        screenshots = q.get("screenshot_paths", [])
        q_type = q.get("type", "other")
        status = "UNANSWERED/ERROR"
        if answer and "[AUTO-TESTER-ERROR]" not in answer:
            requires_ss = q_type in ["screenshot", "text_and_screenshot", "exploration"]
            if requires_ss and not screenshots:
                status = "ANSWERED_TEXT_ONLY (Screenshot Missing/Failed)"
            elif q_type == "screenshot" and screenshots:
                status = "ANSWERED (Screenshot Only)"  # Simplified check
            elif not requires_ss:
                status = "ANSWERED (Text Only)"
            elif requires_ss and screenshots:
                status = "ANSWERED (Text and Screenshot)"
            else:
                status = "ANSWERED (Check Validity)"
        elif screenshots and q_type in ["screenshot", "text_and_screenshot", "exploration"]:
            status = "ANSWERED_SCREENSHOT_ONLY (Text Error/Missing)"
        elif not answer and not screenshots:
            status = "UNANSWERED (No Data)"
        screenshot_status = f"{len(screenshots)} Verified" if screenshots else "None Provided/Verified"
        report_summary += f"Q_ID: {q_id}, Type: {q_type}, Status: {status}, Screenshots: {screenshot_status}, Preview: {str(answer)[:50]}...\n---\n"

    prompt = f"""
    Analyze the summary of answered questions. Identify questions to RETRY if:
    - Status indicates "ERROR" or "UNANSWERED".
    - Status indicates a missing component essential for the type (e.g., "Screenshot Missing/Failed" for screenshot types).
    - Answer Preview is clearly irrelevant or excessively short for detailed types ('text', 'open_feedback', 'exploration', 'rating').

    Return a JSON list containing ONLY the 'id' field (string) for each question to re-attempt. Return `[]` if none need retry based on these rules.

    Summary:
    ```
    {report_summary}
    ```
    Retry IDs JSON:
    """
    try:
        response = llm.invoke(prompt)
        content = response.content
        logging.debug(f"Gemini Raw Crosscheck: {content}")
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
        if json_match:
            json_string = json_match.group(1)
        else:
            json_string = content.strip()
            start, end = json_string.find("["), json_string.rfind("]")
            if start != -1 and end != -1:
                json_string = json_string[start : end + 1]
            else:
                json_string = "[]" if json_string == "[]" else "[]"
        unanswered_ids = json.loads(json_string)
        if not isinstance(unanswered_ids, list):
            raise ValueError("Invalid JSON list.")
        retry_ids = set(
            item if isinstance(item, str) else item.get("id")
            for item in unanswered_ids
            if isinstance(item, (str, dict)) and (isinstance(item, str) or item.get("id"))
        )
        logging.info(f"Gemini identified {len(retry_ids)} IDs to retry: {retry_ids}")
        original_map = {q.get("id"): q for q in parsed_questions}
        questions_to_retry = [original_map[qid] for qid in retry_ids if qid in original_map]
        logging.info(f"Found {len(questions_to_retry)} full objects to retry.")
        return questions_to_retry
    except Exception as e:
        logging.error(f"Error during Gemini crosscheck: {e}", exc_info=True)
        return []


def generate_output_report(original_report_content: str, answered_questions: list[dict]) -> str:
    """Generates the final report text."""
    logging.info("Generating output report...")
    # ... (Keep the implementation from the previous full script version) ...
    output_content = list(original_report_content.splitlines())
    processed_indices = set()
    questions_map = {q.get("id"): q for q in answered_questions if q.get("id")}
    processed_qids_in_insertion = set()

    for line_idx, line in enumerate(output_content):
        if line_idx in processed_indices:
            continue
        best_match_q_id, best_match_line_idx = None, -1
        # Match ID markers
        for q_id, q_data in questions_map.items():
            if q_id in processed_qids_in_insertion:
                continue
            q_id_num = "".join(filter(str.isdigit, q_id))
            q_id_prefix = "".join(filter(str.isalpha, q_id)).lower()
            id_patterns = []
            if q_id_prefix == "uc" and q_id_num:
                id_patterns.append(re.compile(r"^\s*([Uu][Cc]\s*(?:Case)?\s*" + re.escape(q_id_num) + r")\b"))
            elif q_id_prefix == "q" and q_id_num:
                id_patterns.append(re.compile(r"^\s*([Qq]\s*" + re.escape(q_id_num) + r")\b"))
            elif q_id_prefix == "r" and q_id_num:
                id_patterns.append(re.compile(r"^\s*([Rr]\s*" + re.escape(q_id_num) + r")\b"))
            elif q_id == "ScreenshotsUC5":
                id_patterns.append(re.compile(r"^\s*Screenshots:?\s*Please upload"))
            elif q_id.lower().startswith("confirmation"):
                id_patterns.append(re.compile(r"^\s*Confirmation\s+Required"))
            for pattern in id_patterns:
                if pattern.search(line):
                    best_match_q_id, best_match_line_idx = q_id, line_idx
                    break
            if best_match_q_id:
                break
        # Match text snippet if no ID found
        if not best_match_q_id:
            for q_id, q_data in questions_map.items():
                if q_id in processed_qids_in_insertion:
                    continue
                q_text_full = q_data.get("full_section_text", q_data.get("question_text", ""))
                snippet = q_text_full.splitlines()[0].strip()[:60] if q_text_full else ""
                if snippet and snippet in line:
                    best_match_q_id, best_match_line_idx = q_id, line_idx
                    break
        if not best_match_q_id:
            continue

        q_data = questions_map[best_match_q_id]
        answer = q_data.get("answer", f"[ERROR] Answer missing for {best_match_q_id}.")
        screenshot_paths = q_data.get("screenshot_paths", [])
        insert_after_line_idx = best_match_line_idx
        # Find best insertion point
        for idx in range(best_match_line_idx + 1, len(output_content)):
            line_content = output_content[idx].strip()
            if re.match(r"^\s*([Uu][Cc]\s*\d|[QqRr]\s*\d|---)", line_content):
                break
            if not line_content or line_content.startswith(("1.", "...", "​")):
                insert_after_line_idx = idx
                break
            insert_after_line_idx = idx
        # Construct and insert answer lines
        answer_lines = [f">>> AUTO-TESTER ANSWER FOR {best_match_q_id} <<<"] + answer.splitlines()
        if screenshot_paths:
            answer_lines.append("[SCREENSHOTS]:")
            for spath_str in screenshot_paths:
                try:
                    rel_path = Path(spath_str).relative_to(Path.cwd())
                except ValueError:
                    answer_lines.append(f"- {Path(spath_str).as_posix()}")
                else:
                    answer_lines.append(f"- {rel_path.as_posix()}")
        answer_lines.append(">>> AUTO-TESTER ANSWER END <<<")
        for i, line_to_insert in enumerate(reversed(answer_lines)):
            output_content.insert(insert_after_line_idx + 1, line_to_insert)
        for i in range(len(answer_lines) + 1):
            processed_indices.add(insert_after_line_idx + i)
        processed_qids_in_insertion.add(best_match_q_id)
        logging.info(f"Inserted answer for Q_ID: {best_match_q_id} after line {insert_after_line_idx}")

    final_output_lines = list(output_content)
    unprocessed_qids = set(questions_map.keys()) - processed_qids_in_insertion
    if unprocessed_qids:
        logging.warning(f"Could not find insertion points for Q_IDs: {', '.join(sorted(list(unprocessed_qids)))}")
        final_output_lines.append("\n\n--- ANSWERS FOR UNPLACED QUESTIONS ---")
        for qid in sorted(list(unprocessed_qids)):
            if qid not in questions_map:
                continue
            q_data = questions_map[qid]
            answer, screenshots = q_data.get("answer", "[N/A]"), q_data.get("screenshot_paths", [])
            final_output_lines.append(f"\n-- Q_ID: {qid} --")
            final_output_lines.extend(answer.splitlines())
            if screenshots:
                final_output_lines.append("[SCREENSHOTS]:")
                final_output_lines.extend([f"- {Path(p).as_posix()}" for p in screenshots])
            final_output_lines.append("-- END Q_ID --\n")
    return "\n".join(final_output_lines)


# --- Main Execution Logic ---
async def main():
    """Main async function to run the web tester."""
    target_url = get_target_website()
    if not target_url:
        return

    original_report_content = read_report(INPUT_REPORT_FILE)
    if not original_report_content:
        return

    # --- Initialize LLMs ---
    try:
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            convert_system_message_to_human=True,
            temperature=0.1,
        )
        browser_llm = gemini_llm  # Using Gemini for both roles
        logging.info(f"Using {gemini_llm.model} for parsing and {browser_llm.model} for browser control.")
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}", exc_info=True)
        return

    # 1. Parse the report
    parsed_questions = parse_report_with_gemini(original_report_content, gemini_llm)
    if not parsed_questions:
        return

    # --- Initialize Browser and Agent ONCE ---
    browser_instance = None
    agent = None  # Initialize agent variable
    try:
        logging.info("Launching browser (headless=False)...")
        browser_instance = Browser(config=BrowserConfig(headless=False, new_context_config=BrowserContextConfig()))
        logging.info(f"Browser process launched. Instance ID: {id(browser_instance)}")

        # Create the BrowserAgent ONCE
        # We give it an initial placeholder task; it will be updated in the loop
        logging.info("Initializing Browser Agent...")
        agent = BrowserAgent(
            llm=browser_llm,
            browser=browser_instance,
            task="Initial placeholder task. Will be updated.",  # Placeholder task
        )
        logging.info(f"Browser Agent initialized. Agent ID: {id(agent)}")

        # --- Initial Navigation & User Cookie Prompt ---
        logging.info(f"Performing initial navigation via agent to {target_url}...")
        agent.task = INITIAL_AGENT_TASK.format(target_url=target_url)  # Set the initial task
        try:
            await agent.run(max_steps=5)
            logging.info(f"Initial navigation likely complete.")
        except Exception as nav_e:
            logging.error(f"Initial navigation via agent failed: {nav_e}", exc_info=True)
            # Consider if script should exit here

        # --- Prompt user using console input ---
        prompt_user_for_cookie_handling(target_url)
        # --- Execution resumes here AFTER user presses Enter ---

        logging.info("User interaction complete. Starting automated tasks...")

        # --- Main Question Answering Loop ---
        questions_to_attempt_in_cycle = list(parsed_questions)
        all_question_ids = [q.get("id") for q in parsed_questions]

        for attempt in range(MAX_RETRIES + 1):
            if not questions_to_attempt_in_cycle:
                logging.info("All questions processed or max retries reached.")
                break

            logging.info(
                f"--- Starting Answer Cycle {attempt + 1} / {MAX_RETRIES + 1} for {len(questions_to_attempt_in_cycle)} questions ---"
            )
            ids_failed_in_this_cycle = set()

            for i, question_data_to_attempt in enumerate(questions_to_attempt_in_cycle):
                q_id = question_data_to_attempt.get("id", f"Unknown_{i}")
                logging.info(f"Processing Q_ID: {q_id} ({i+1}/{len(questions_to_attempt_in_cycle)})")

                # --- Generate Task Prompt for the CURRENT question ---
                # (This reuses the prompt generation logic from the *previous* answer_question_with_browser_use function)
                q_text = question_data_to_attempt.get("question_text", "No question text provided.")
                q_type = question_data_to_attempt.get("type", "text")
                q_context = question_data_to_attempt.get("full_section_text", q_text)
                q_multi_screenshot = question_data_to_attempt.get("requires_multiple_screenshots", False)
                screenshot_base_filename = f"{q_id.replace(' ', '_').replace('/', '_')}_{Path(target_url).stem}"
                expected_screenshot_paths = []
                if q_type in ["screenshot", "text_and_screenshot", "exploration"]:
                    if q_multi_screenshot:
                        expected_screenshot_paths = [SCREENSHOTS_DIR / f"{screenshot_base_filename}_1.png"]
                    else:
                        expected_screenshot_paths = [SCREENSHOTS_DIR / f"{screenshot_base_filename}.png"]
                # --- Construct the Task Prompt Dynamically ---
                current_task_prompt = f"""
Your Goal: You are an automated web usability tester continuing a session. Use the CURRENT browser window/tab and its state (cookies, etc.) to interact with the website '{target_url}' and answer the question/task below. Do NOT open new browser windows. Navigate within the current tab as needed.

Context from Usability Report Section:
\"\"\"
{q_context}
\"\"\"
Specific Question/Task to Address NOW: "{q_text}"
Question Type: '{q_type}'
Requires Multiple Screenshots: {q_multi_screenshot}
--- Required Output Format (Strict JSON) ---
Output ONLY a single JSON object: {{ "answer_text": "...", "screenshot_paths": ["...", "..."] }}
--- Detailed Instructions based on Type ---
"""  # (Append specific type instructions + screenshot instructions + general instructions as before)
                # --- Append Specific Instructions based on Type (Copied from previous function) ---
                screenshot_instruction = ""
                paths_str_prompt = [
                    p.absolute().as_posix() for p in expected_screenshot_paths
                ]  # Example paths for prompt
                base_path_str_prompt = (
                    (SCREENSHOTS_DIR / screenshot_base_filename).absolute().as_posix()
                )  # Base path without extension for multi
                if q_type in ["screenshot", "text_and_screenshot", "exploration"]:
                    if q_multi_screenshot:
                        screenshot_instruction = f"\n- IMPORTANT MULTI-SCREENSHOT TASK:\n - Take MULTIPLE screenshots ({question_data_to_attempt.get('screenshot_count', 'e.g., 4')}).\n - Save sequentially: '{base_path_str_prompt}_1.png', '{base_path_str_prompt}_2.png', etc.\n - List ALL saved paths accurately in 'screenshot_paths'."
                    else:
                        screenshot_instruction = f"\n- IMPORTANT SCREENSHOT TASK:\n - Take ONE FULL screenshot.\n - Save EXACTLY to: '{paths_str_prompt[0]}'\n - Verify save.\n - List this path in 'screenshot_paths': [\"{paths_str_prompt[0]}\"]"
                # Append instructions based on q_type...
                if q_type == "text":
                    current_task_prompt += "\n- Provide detailed numbered steps in 'answer_text'."
                elif q_type == "screenshot":
                    current_task_prompt += (
                        screenshot_instruction
                        + "\n- Navigate to fulfill request.\n- 'answer_text' confirms screenshot content."
                    )
                elif q_type == "text_and_screenshot":
                    current_task_prompt += (
                        "\n- Provide detailed numbered steps in 'answer_text'." + screenshot_instruction
                    )
                elif q_type == "rating":
                    current_task_prompt += "\n- Provide rating and justification in 'answer_text': \"Rating: [Number] - [Justification]\"."
                elif q_type == "confirmation":
                    current_task_prompt += "\n- Verify condition. Output confirmation in 'answer_text'."
                elif q_type == "exploration":
                    current_task_prompt += (
                        "\n- Explore TWO distinct sections.\n- Provide numbered steps for Section 1 & Section 2 in 'answer_text'."
                        + screenshot_instruction
                    )
                elif q_type == "open_feedback":
                    current_task_prompt += (
                        f"\n- Provide detailed feedback for '{q_text}' in 'answer_text'. Use numbered list if needed."
                    )
                else:
                    current_task_prompt += "\n- Provide textual answer/description in 'answer_text'."
                current_task_prompt += f"""
--- General Instructions ---
- Operate within the existing browser session for {target_url}.
- Be thorough. State clearly in 'answer_text' if info truly cannot be found.
- CRITICAL: Respond ONLY with the single, valid JSON object. No extra text.
"""
                # --- Update Agent Task and Run ---
                agent.task = current_task_prompt  # Update the task of the *existing* agent
                agent_response_text = ""
                updated_question_data = question_data_to_attempt.copy()  # Work on a copy for safety in this attempt

                try:
                    logging.info(f"Running agent {id(agent)} for Q_ID: {q_id}")
                    run_result = await agent.run(max_steps=40)
                    agent_response_text = run_result.output if hasattr(run_result, "output") else str(run_result)
                    logging.debug(f"Raw agent output for Q_ID {q_id} (Attempt {attempt+1}):\n{agent_response_text}")

                    # --- Extract JSON ---
                    json_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", agent_response_text, re.DOTALL)
                    if not json_match:
                        start_brace, end_brace = agent_response_text.find("{"), agent_response_text.rfind("}")
                        if start_brace != -1 and end_brace != -1:
                            json_string = agent_response_text[start_brace : end_brace + 1]
                        else:
                            json_string = None
                    else:
                        json_string = json_match.group(1)

                    if json_string:
                        result_data = json.loads(json_string)
                        if not isinstance(result_data, dict):
                            raise ValueError("JSON not a dict.")
                        updated_question_data["answer"] = result_data.get("answer_text")
                        paths_from_agent = result_data.get("screenshot_paths", [])
                        updated_question_data["screenshot_paths"] = (
                            [str(p).strip(" '\"") for p in paths_from_agent if isinstance(p, str) and p.strip(" '\"")]
                            if isinstance(paths_from_agent, list)
                            else []
                        )

                        # --- Verify Screenshots ---
                        if updated_question_data["screenshot_paths"]:
                            verified_paths = []
                            for p_str in updated_question_data["screenshot_paths"]:
                                p = Path(p_str)
                                if not p.is_absolute():
                                    p_abs_cwd = Path.cwd() / p
                                    p_abs_ssdir = SCREENSHOTS_DIR / p.name
                                    if p_abs_cwd.is_file():
                                        p = p_abs_cwd
                                    elif p_abs_ssdir.is_file():
                                        p = p_abs_ssdir
                                    else:
                                        p = Path(p_str)
                                if p.is_file():
                                    verified_paths.append(p.absolute().as_posix())
                                else:
                                    logging.warning(
                                        f"Screenshot path provided but file NOT found: {p.absolute().as_posix()} (Original: '{p_str}')"
                                    )
                            updated_question_data["screenshot_paths"] = verified_paths

                    else:  # No JSON found
                        logging.warning(f"No JSON block found in agent response for Q_ID {q_id}.")
                        if (
                            q_type in ["text", "open_feedback", "rating", "confirmation"]
                            and not requires_ss
                            and agent_response_text
                        ):
                            updated_question_data["answer"] = agent_response_text.strip()

                except Exception as agent_e:
                    logging.error(
                        f"Error during agent run or JSON processing for Q_ID {q_id} (Attempt {attempt+1}): {agent_e}",
                        exc_info=True,
                    )
                    updated_question_data["answer"] = (
                        f"[AUTO-TESTER-ERROR] Agent run/parse failed on attempt {attempt+1}: {type(agent_e).__name__}"
                    )
                    updated_question_data["screenshot_paths"] = []

                # --- Update Master List with Result ---
                updated = False
                for master_idx, master_q in enumerate(parsed_questions):
                    if master_q.get("id") == q_id:
                        parsed_questions[master_idx] = updated_question_data
                        updated = True
                        break
                if not updated:
                    logging.error(f"Consistency error: Q_ID {q_id} not found in master list.")

                # --- Check Result for Retry ---
                answer = updated_question_data.get("answer", "")
                screenshots = updated_question_data.get("screenshot_paths", [])
                is_error = "[AUTO-TESTER-ERROR]" in answer
                is_incomplete = requires_ss and not screenshots and not is_error

                if is_error or is_incomplete:
                    ids_failed_in_this_cycle.add(q_id)
                    logging.warning(
                        f"Q_ID {q_id} failed/incomplete. Reason: {'Error' if is_error else 'Missing/Failed screenshot(s)'}"
                    )
                else:
                    logging.info(f"Q_ID {q_id} processed successfully.")
                # await asyncio.sleep(1) # Optional delay

            # --- Prepare for Next Retry Cycle ---
            if not ids_failed_in_this_cycle:
                logging.info(f"All questions in cycle {attempt + 1} processed successfully.")
                break
            questions_to_attempt_in_cycle = [q for q in parsed_questions if q.get("id") in ids_failed_in_this_cycle]
            if attempt < MAX_RETRIES:
                logging.info(
                    f"{len(questions_to_attempt_in_cycle)} questions failed/incomplete. Preparing for retry {attempt + 2}..."
                )
                await asyncio.sleep(10)
            else:
                logging.warning(
                    f"Max retries ({MAX_RETRIES + 1}) reached. {len(questions_to_attempt_in_cycle)} questions have errors or are incomplete."
                )

        # 5. Generate Final Report
        final_report_content = generate_output_report(original_report_content, parsed_questions)

        # 6. Write Output Report
        try:
            with open(OUTPUT_REPORT_FILE, "w", encoding="utf-8") as f:
                f.write(final_report_content)
            logging.info(f"Successfully generated output report: {OUTPUT_REPORT_FILE}")
        except Exception as e:
            logging.error(f"Error writing output report file: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"An critical error occurred in the main execution: {e}", exc_info=True)
    finally:
        # 7. Clean up - Ensure the SINGLE browser instance is closed
        if browser_instance:
            logging.info("Closing browser...")
            await browser_instance.close()
            logging.info("Browser closed.")


if __name__ == "__main__":
    asyncio.run(main())
