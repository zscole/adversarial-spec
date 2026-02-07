"""Model calling, cost tracking, and response handling."""

from __future__ import annotations

import concurrent.futures
import difflib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ["LITELLM_LOG"] = "ERROR"

try:
    import litellm
    from litellm import completion

    litellm.suppress_debug_info = True
except ImportError:
    print(
        "Error: litellm package not installed. Run: pip install litellm",
        file=sys.stderr,
    )
    sys.exit(1)

from prompts import (
    FOCUS_AREAS,
    PRESERVE_INTENT_PROMPT,
    PRESS_PROMPT_TEMPLATE,
    REVIEW_PROMPT_TEMPLATE,
    get_doc_type_name,
    get_system_prompt,
)
from providers import (
    CLAUDE_CLI_AVAILABLE,
    CODEX_AVAILABLE,
    DEFAULT_CODEX_REASONING,
    DEFAULT_COST,
    GEMINI_CLI_AVAILABLE,
    MODEL_COSTS,
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds


def is_o_series_model(model: str) -> bool:
    """
    Check if a model is an OpenAI O-series model.

    O-series models (o1, o1-mini, o1-preview) don't support custom temperature.
    They only accept temperature=1 or no temperature parameter.

    Args:
        model: Model identifier string.

    Returns:
        True if the model is an O-series model.
    """
    model_lower = model.lower()
    return model_lower.startswith("o1") or "/o1" in model_lower or "-o1" in model_lower


@dataclass
class ModelResponse:
    """Response from a model critique."""

    model: str
    response: str
    agreed: bool
    spec: Optional[str]
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class CostTracker:
    """Track token usage and costs across model calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Add usage for a model call and return the cost."""
        costs = MODEL_COSTS.get(model, DEFAULT_COST)
        cost = (input_tokens / 1_000_000 * costs["input"]) + (
            output_tokens / 1_000_000 * costs["output"]
        )

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        if model not in self.by_model:
            self.by_model[model] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        self.by_model[model]["input_tokens"] += input_tokens
        self.by_model[model]["output_tokens"] += output_tokens
        self.by_model[model]["cost"] += cost

        return cost

    def summary(self) -> str:
        """Generate cost summary string."""
        lines = ["", "=== Cost Summary ==="]
        lines.append(
            f"Total tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out"
        )
        lines.append(f"Total cost: ${self.total_cost:.4f}")
        if len(self.by_model) > 1:
            lines.append("")
            lines.append("By model:")
            for model, data in self.by_model.items():
                lines.append(
                    f"  {model}: ${data['cost']:.4f} ({data['input_tokens']:,} in / {data['output_tokens']:,} out)"
                )
        return "\n".join(lines)


# Global cost tracker instance
cost_tracker = CostTracker()


def load_context_files(context_paths: list[str]) -> str:
    """Load and format context files for inclusion in prompts."""
    if not context_paths:
        return ""

    sections = []
    for path in context_paths:
        try:
            content = Path(path).read_text()
            sections.append(f"### Context: {path}\n```\n{content}\n```")
        except Exception as e:
            sections.append(f"### Context: {path}\n[Error loading file: {e}]")

    return (
        "## Additional Context\nThe following documents are provided as context:\n\n"
        + "\n\n".join(sections)
    )


def detect_agreement(response: str) -> bool:
    """Check if response indicates agreement."""
    return "[AGREE]" in response


def extract_spec(response: str) -> Optional[str]:
    """Extract spec content from [SPEC]...[/SPEC] tags."""
    if "[SPEC]" not in response or "[/SPEC]" not in response:
        return None
    start = response.find("[SPEC]") + len("[SPEC]")
    end = response.find("[/SPEC]")
    return response[start:end].strip()


def extract_tasks(response: str) -> list[dict]:
    """Extract tasks from export-tasks response."""
    tasks = []
    parts = response.split("[TASK]")
    for part in parts[1:]:
        if "[/TASK]" not in part:
            continue
        task_text = part.split("[/TASK]")[0].strip()
        task: dict[str, str | list[str]] = {}
        current_key: Optional[str] = None
        current_value: list[str] = []

        for line in task_text.split("\n"):
            line = line.strip()
            if line.startswith("title:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "title"
                current_value = [line[6:].strip()]
            elif line.startswith("type:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "type"
                current_value = [line[5:].strip()]
            elif line.startswith("priority:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "priority"
                current_value = [line[9:].strip()]
            elif line.startswith("description:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "description"
                current_value = [line[12:].strip()]
            elif line.startswith("acceptance_criteria:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "acceptance_criteria"
                current_value = []
            elif line.startswith("- ") and current_key == "acceptance_criteria":
                current_value.append(line[2:])
            elif current_key:
                current_value.append(line)

        if current_key:
            task[current_key] = (
                current_value
                if current_key == "acceptance_criteria"
                else "\n".join(current_value).strip()
            )

        if task.get("title"):
            tasks.append(task)

    return tasks


def get_critique_summary(response: str, max_length: int = 300) -> str:
    """Get a summary of the critique portion of a response."""
    spec_start = response.find("[SPEC]")
    if spec_start > 0:
        critique = response[:spec_start].strip()
    else:
        critique = response

    if len(critique) > max_length:
        critique = critique[:max_length] + "..."
    return critique


def generate_diff(previous: str, current: str) -> str:
    """Generate unified diff between two specs."""
    prev_lines = previous.splitlines(keepends=True)
    curr_lines = current.splitlines(keepends=True)

    diff = difflib.unified_diff(
        prev_lines, curr_lines, fromfile="previous", tofile="current", lineterm=""
    )
    return "".join(diff)


def call_codex_model(
    system_prompt: str,
    user_message: str,
    model: str,
    reasoning_effort: str = DEFAULT_CODEX_REASONING,
    timeout: int = 600,
    search: bool = False,
) -> tuple[str, int, int]:
    """
    Call Codex CLI in headless mode using ChatGPT subscription.

    Args:
        system_prompt: System instructions for the model
        user_message: User prompt to send
        model: Model name (e.g., "codex/gpt-5.2-codex" -> uses "gpt-5.2-codex")
        reasoning_effort: Thinking level (minimal, low, medium, high, xhigh). Default: xhigh
        timeout: Timeout in seconds (default 10 minutes)
        search: Enable web search capability for Codex

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)

    Raises:
        RuntimeError: If Codex CLI is not available or fails
    """
    if not CODEX_AVAILABLE:
        raise RuntimeError(
            "Codex CLI not found. Install with: npm install -g @openai/codex"
        )

    # Extract actual model name from "codex/model" format
    actual_model = model.split("/", 1)[1] if "/" in model else model

    # Combine system prompt and user message for Codex
    full_prompt = f"""SYSTEM INSTRUCTIONS:
{system_prompt}

USER REQUEST:
{user_message}"""

    try:
        cmd = [
            "codex",
            "exec",
            "--json",
            "--full-auto",
            "--skip-git-repo-check",
            "--model",
            actual_model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
        ]
        if search:
            cmd.append("--search")
        cmd.append(full_prompt)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            error_msg = (
                result.stderr.strip() or f"Codex exited with code {result.returncode}"
            )
            raise RuntimeError(f"Codex CLI failed: {error_msg}")

        # Parse JSONL output to extract agent messages
        response_text = ""
        input_tokens = 0
        output_tokens = 0

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)

                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        response_text = item.get("text", "")

                if event.get("type") == "turn.completed":
                    usage = event.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

            except json.JSONDecodeError:
                continue

        if not response_text:
            raise RuntimeError("No agent message found in Codex output")

        return response_text, input_tokens, output_tokens

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Codex CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Codex CLI not found in PATH")


def call_claude_cli_model(
    system_prompt: str,
    user_message: str,
    model: str,
    timeout: int = 600,
) -> tuple[str, int, int]:
    """
    Call Claude CLI in print mode.

    Args:
        system_prompt: System instructions for the model
        user_message: User prompt to send
        model: Model name (e.g., "claude-cli/sonnet" -> uses "sonnet")
        timeout: Timeout in seconds (default 10 minutes)

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)

    Raises:
        RuntimeError: If Claude CLI is not available or fails
    """
    if not CLAUDE_CLI_AVAILABLE:
        raise RuntimeError(
            "Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        )

    actual_model = model.split("/", 1)[1] if "/" in model else model

    try:
        cmd = [
            "claude",
            "-p",
            "--output-format",
            "text",
            "--model",
            actual_model,
            "--append-system-prompt",
            system_prompt,
            user_message,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode != 0:
            error_msg = (
                result.stderr.strip() or f"Claude CLI exited with code {result.returncode}"
            )
            raise RuntimeError(f"Claude CLI failed: {error_msg}")

        response_text = result.stdout.strip()
        if not response_text:
            raise RuntimeError("No response from Claude CLI")

        input_tokens = (len(system_prompt) + len(user_message)) // 4
        output_tokens = len(response_text) // 4

        return response_text, input_tokens, output_tokens

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Claude CLI not found in PATH")


def call_gemini_cli_model(
    system_prompt: str,
    user_message: str,
    model: str,
    timeout: int = 600,
) -> tuple[str, int, int]:
    """
    Call Gemini CLI for model inference using Google account authentication.

    Args:
        system_prompt: System instructions for the model
        user_message: User prompt to send
        model: Model name (e.g., "gemini-cli/gemini-3-pro-preview" -> uses "gemini-3-pro-preview")
        timeout: Timeout in seconds (default 10 minutes)

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
        Note: Gemini CLI doesn't report token usage, so tokens are estimated.

    Raises:
        RuntimeError: If Gemini CLI is not available or fails
    """
    if not GEMINI_CLI_AVAILABLE:
        raise RuntimeError(
            "Gemini CLI not found. Install with: npm install -g @google/gemini-cli"
        )

    # Extract actual model name from "gemini-cli/model" format
    actual_model = model.split("/", 1)[1] if "/" in model else model

    # Combine system prompt and user message
    full_prompt = f"""SYSTEM INSTRUCTIONS:
{system_prompt}

USER REQUEST:
{user_message}"""

    try:
        # Use gemini CLI with the prompt passed via stdin and -p flag
        cmd = [
            "gemini",
            "-m",
            actual_model,
            "-y",
        ]  # -y for auto-approve (no tool calls expected)

        result = subprocess.run(
            cmd, input=full_prompt, capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            error_msg = (
                result.stderr.strip()
                or f"Gemini CLI exited with code {result.returncode}"
            )
            raise RuntimeError(f"Gemini CLI failed: {error_msg}")

        response_text = result.stdout.strip()

        # Filter out noise lines from gemini CLI output
        lines = response_text.split("\n")
        filtered_lines = []
        skip_prefixes = ("Loaded cached", "Server ", "Loading extension")
        for line in lines:
            if not any(line.startswith(prefix) for prefix in skip_prefixes):
                filtered_lines.append(line)
        response_text = "\n".join(filtered_lines).strip()

        if not response_text:
            raise RuntimeError("No response from Gemini CLI")

        # Estimate tokens (Gemini CLI doesn't report actual usage)
        # Rough estimate: 4 chars per token
        input_tokens = len(full_prompt) // 4
        output_tokens = len(response_text) // 4

        return response_text, input_tokens, output_tokens

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Gemini CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Gemini CLI not found in PATH")


def call_single_model(
    model: str,
    spec: str,
    round_num: int,
    doc_type: str,
    press: bool = False,
    focus: Optional[str] = None,
    persona: Optional[str] = None,
    context: Optional[str] = None,
    preserve_intent: bool = False,
    codex_reasoning: str = DEFAULT_CODEX_REASONING,
    codex_search: bool = False,
    timeout: int = 600,
    bedrock_mode: bool = False,
    bedrock_region: Optional[str] = None,
) -> ModelResponse:
    """Send spec to a single model and return response with retry on failure."""
    # Handle Bedrock routing
    actual_model = model
    if bedrock_mode:
        if bedrock_region:
            os.environ["AWS_REGION"] = bedrock_region
        if not model.startswith("bedrock/"):
            actual_model = f"bedrock/{model}"

    system_prompt = get_system_prompt(doc_type, persona)
    doc_type_name = get_doc_type_name(doc_type)

    focus_section = ""
    if focus and focus.lower() in FOCUS_AREAS:
        focus_section = FOCUS_AREAS[focus.lower()]
    elif focus:
        focus_section = f"**CRITICAL FOCUS: {focus.upper()}**\nPrioritize analysis of {focus} concerns above all else."

    if preserve_intent:
        focus_section = PRESERVE_INTENT_PROMPT + "\n\n" + focus_section

    context_section = context if context else ""

    template = PRESS_PROMPT_TEMPLATE if press else REVIEW_PROMPT_TEMPLATE
    user_message = template.format(
        round=round_num,
        doc_type_name=doc_type_name,
        spec=spec,
        focus_section=focus_section,
        context_section=context_section,
    )

    # Route Codex CLI models to dedicated handler
    if model.startswith("codex/"):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                content, input_tokens, output_tokens = call_codex_model(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    model=model,
                    reasoning_effort=codex_reasoning,
                    timeout=timeout,
                    search=codex_search,
                )
                agreed = "[AGREE]" in content
                extracted = extract_spec(content)

                if not agreed and not extracted:
                    print(
                        f"Warning: {model} provided critique but no [SPEC] tags found. Response may be malformed.",
                        file=sys.stderr,
                    )

                cost = cost_tracker.add(model, input_tokens, output_tokens)

                return ModelResponse(
                    model=model,
                    response=content,
                    agreed=agreed,
                    spec=extracted,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    print(
                        f"Warning: {model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"Error: {model} failed after {MAX_RETRIES} attempts: {last_error}",
                        file=sys.stderr,
                    )

        return ModelResponse(
            model=model, response="", agreed=False, spec=None, error=last_error
        )

    # Route Gemini CLI models to dedicated handler
    if model.startswith("gemini-cli/"):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                content, input_tokens, output_tokens = call_gemini_cli_model(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    model=model,
                    timeout=timeout,
                )
                agreed = "[AGREE]" in content
                extracted = extract_spec(content)

                if not agreed and not extracted:
                    print(
                        f"Warning: {model} provided critique but no [SPEC] tags found. Response may be malformed.",
                        file=sys.stderr,
                    )

                cost = cost_tracker.add(model, input_tokens, output_tokens)

                return ModelResponse(
                    model=model,
                    response=content,
                    agreed=agreed,
                    spec=extracted,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    print(
                        f"Warning: {model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"Error: {model} failed after {MAX_RETRIES} attempts: {last_error}",
                        file=sys.stderr,
                    )

        return ModelResponse(
            model=model, response="", agreed=False, spec=None, error=last_error
        )

    # Route Claude CLI models to dedicated handler
    if model.startswith("claude-cli/"):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                content, input_tokens, output_tokens = call_claude_cli_model(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    model=model,
                    timeout=timeout,
                )
                agreed = "[AGREE]" in content
                extracted = extract_spec(content)

                if not agreed and not extracted:
                    print(
                        f"Warning: {model} provided critique but no [SPEC] tags found. Response may be malformed.",
                        file=sys.stderr,
                    )

                cost = cost_tracker.add(model, input_tokens, output_tokens)

                return ModelResponse(
                    model=model,
                    response=content,
                    agreed=agreed,
                    spec=extracted,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2**attempt)
                    print(
                        f"Warning: {model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                else:
                    print(
                        f"Error: {model} failed after {MAX_RETRIES} attempts: {last_error}",
                        file=sys.stderr,
                    )

        return ModelResponse(
            model=model, response="", agreed=False, spec=None, error=last_error
        )

    # Standard litellm path for all other providers
    last_error = None
    display_model = model

    for attempt in range(MAX_RETRIES):
        try:
            # Build completion kwargs
            completion_kwargs = {
                "model": actual_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "max_tokens": 8000,
                "timeout": timeout,
            }

            # O-series models don't support custom temperature
            if not is_o_series_model(actual_model):
                completion_kwargs["temperature"] = 0.7

            response = completion(**completion_kwargs)
            content = response.choices[0].message.content
            agreed = "[AGREE]" in content
            extracted = extract_spec(content)

            if not agreed and not extracted:
                print(
                    f"Warning: {display_model} provided critique but no [SPEC] tags found. Response may be malformed.",
                    file=sys.stderr,
                )

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            cost = cost_tracker.add(display_model, input_tokens, output_tokens)

            return ModelResponse(
                model=display_model,
                response=content,
                agreed=agreed,
                spec=extracted,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )
        except Exception as e:
            last_error = str(e)
            if bedrock_mode:
                if "AccessDeniedException" in last_error:
                    last_error = (
                        f"Model not enabled in your Bedrock account: {display_model}"
                    )
                elif "ValidationException" in last_error:
                    last_error = f"Invalid Bedrock model ID: {display_model}"

            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"Warning: {display_model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(
                    f"Error: {display_model} failed after {MAX_RETRIES} attempts: {last_error}",
                    file=sys.stderr,
                )

    return ModelResponse(
        model=display_model, response="", agreed=False, spec=None, error=last_error
    )


def call_models_parallel(
    models: list[str],
    spec: str,
    round_num: int,
    doc_type: str,
    press: bool = False,
    focus: Optional[str] = None,
    persona: Optional[str] = None,
    context: Optional[str] = None,
    preserve_intent: bool = False,
    codex_reasoning: str = DEFAULT_CODEX_REASONING,
    codex_search: bool = False,
    timeout: int = 600,
    bedrock_mode: bool = False,
    bedrock_region: Optional[str] = None,
) -> list[ModelResponse]:
    """Call multiple models in parallel and collect responses."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {
            executor.submit(
                call_single_model,
                model,
                spec,
                round_num,
                doc_type,
                press,
                focus,
                persona,
                context,
                preserve_intent,
                codex_reasoning,
                codex_search,
                timeout,
                bedrock_mode,
                bedrock_region,
            ): model
            for model in models
        }
        for future in concurrent.futures.as_completed(future_to_model):
            results.append(future.result())
    return results
