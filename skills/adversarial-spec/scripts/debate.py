#!/usr/bin/env python3
"""
Adversarial spec debate script.
Sends specs to multiple LLMs for critique using LiteLLM.

Usage:
    echo "spec" | python3 debate.py critique --models gpt-4o
    echo "spec" | python3 debate.py critique --models gpt-4o,gemini/gemini-2.0-flash,xai/grok-3 --doc-type prd
    echo "spec" | python3 debate.py critique --models codex/gpt-5.3-codex,gemini/gemini-2.0-flash --doc-type tech
    echo "spec" | python3 debate.py critique --models gpt-4o --focus security
    echo "spec" | python3 debate.py critique --models gpt-4o --persona "security engineer"
    echo "spec" | python3 debate.py critique --models gpt-4o --context ./api.md --context ./schema.sql
    echo "spec" | python3 debate.py critique --models gpt-4o --profile strict-security
    echo "spec" | python3 debate.py critique --models gpt-4o --preserve-intent
    echo "spec" | python3 debate.py critique --models gpt-4o --session my-debate
    python3 debate.py critique --resume my-debate
    echo "spec" | python3 debate.py diff --previous prev.md --current current.md
    echo "spec" | python3 debate.py export-tasks --doc-type prd
    python3 debate.py providers
    python3 debate.py profiles
    python3 debate.py sessions

Supported providers (set corresponding API key):
    OpenAI:     OPENAI_API_KEY       models: gpt-4o, gpt-4-turbo, o1, etc.
    Anthropic:  ANTHROPIC_API_KEY    models: claude-sonnet-4-20250514, claude-opus-4-20250514, etc.
    Google:     GEMINI_API_KEY       models: gemini/gemini-2.0-flash, gemini/gemini-pro, etc.
    xAI:        XAI_API_KEY          models: xai/grok-3, xai/grok-beta, etc.
    Mistral:    MISTRAL_API_KEY      models: mistral/mistral-large, etc.
    Groq:       GROQ_API_KEY         models: groq/llama-3.3-70b, etc.
    OpenRouter: OPENROUTER_API_KEY   models: openrouter/openai/gpt-4o, openrouter/anthropic/claude-3.5-sonnet, etc.
    Codex CLI:  (ChatGPT subscription) models: codex/gpt-5.3-codex, codex/gpt-5.2-codex
                Install: npm install -g @openai/codex && codex login
                Reasoning: --codex-reasoning xhigh (minimal, low, medium, high, xhigh)

Document types:
    prd   - Product Requirements Document (business/product focus)
    tech  - Technical Specification / Architecture Document (engineering focus)

Exit codes:
    0 - Success
    1 - API error
    2 - Missing API key or config error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
os.environ["LITELLM_LOG"] = "ERROR"

try:
    from litellm import completion
except ImportError:
    print(
        "Error: litellm package not installed. Run: pip install litellm",
        file=sys.stderr,
    )
    sys.exit(1)

from models import (  # noqa: E402
    ModelResponse,
    call_models_parallel,
    cost_tracker,
    extract_tasks,
    generate_diff,
    get_critique_summary,
    is_o_series_model,
    load_context_files,
)
from prompts import EXPORT_TASKS_PROMPT, get_doc_type_name  # noqa: E402
from providers import (  # noqa: E402
    DEFAULT_CODEX_REASONING,
    get_bedrock_config,
    get_default_model,
    handle_bedrock_command,
    list_focus_areas,
    list_personas,
    list_profiles,
    list_providers,
    load_profile,
    save_profile,
    validate_bedrock_models,
    validate_model_credentials,
)
from session import SESSIONS_DIR, SessionState, save_checkpoint  # noqa: E402


def send_telegram_notification(
    models: list[str], round_num: int, results: list[ModelResponse], poll_timeout: int
) -> Optional[str]:
    """Send Telegram notification with all model responses and poll for feedback.

    Args:
        models: List of model identifiers used.
        round_num: Current round number.
        results: List of model responses.
        poll_timeout: Seconds to wait for user reply.

    Returns:
        User feedback text if received, None otherwise.
    """
    try:
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        import telegram_bot

        token, chat_id = telegram_bot.get_config()
        if not token or not chat_id:
            print(
                "Warning: Telegram not configured. Skipping notification.",
                file=sys.stderr,
            )
            return None

        summaries = []
        all_agreed = True
        for r in results:
            if r.error:
                summaries.append(f"`{r.model}`: ERROR - {r.error[:100]}")
                all_agreed = False
            elif r.agreed:
                summaries.append(f"`{r.model}`: AGREE")
            else:
                all_agreed = False
                summary = get_critique_summary(r.response, 200)
                summaries.append(f"`{r.model}`: {summary}")

        status = "ALL AGREE" if all_agreed else "Critiques received"
        notification = f"""*Round {round_num} complete*

Status: {status}
Models: {len(results)}
Cost: ${cost_tracker.total_cost:.4f}

"""
        notification += "\n\n".join(summaries)

        last_update = telegram_bot.get_last_update_id(token)

        full_notification = (
            notification
            + f"\n\n_Reply within {poll_timeout}s to add feedback, or wait to continue._"
        )
        if not telegram_bot.send_long_message(token, chat_id, full_notification):
            print("Warning: Failed to send Telegram notification.", file=sys.stderr)
            return None

        feedback = telegram_bot.poll_for_reply(
            token, chat_id, poll_timeout, last_update
        )
        return feedback

    except ImportError:
        print(
            "Warning: telegram_bot.py not found. Skipping notification.",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(f"Warning: Telegram error: {e}", file=sys.stderr)
        return None


def send_final_spec_to_telegram(
    spec: str, rounds: int, models: list[str], doc_type: str
) -> bool:
    """Send the final converged spec to Telegram.

    Args:
        spec: The final spec content.
        rounds: Number of rounds completed.
        models: List of model identifiers used.
        doc_type: Document type (prd or tech).

    Returns:
        True on success, False on failure.
    """
    try:
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        import telegram_bot

        token, chat_id = telegram_bot.get_config()
        if not token or not chat_id:
            print(
                "Warning: Telegram not configured. Skipping final spec notification.",
                file=sys.stderr,
            )
            return False

        doc_type_name = get_doc_type_name(doc_type)
        models_str = ", ".join(f"`{m}`" for m in models)
        header = f"""*Debate complete!*

Document: {doc_type_name}
Rounds: {rounds}
Models: Claude vs {models_str}
Total cost: ${cost_tracker.total_cost:.4f}

Final document:
---"""

        if not telegram_bot.send_message(token, chat_id, header):
            return False

        return telegram_bot.send_long_message(token, chat_id, spec)

    except Exception as e:
        print(f"Warning: Failed to send final spec to Telegram: {e}", file=sys.stderr)
        return False


def add_core_arguments(parser: argparse.ArgumentParser) -> None:
    """Add core critique arguments to parser."""
    parser.add_argument(
        "--models",
        "-m",
        default=None,
        help="Comma-separated list of models (e.g., gpt-4o,gemini/gemini-2.0-flash,xai/grok-3)",
    )
    parser.add_argument(
        "--doc-type",
        "-d",
        choices=["prd", "tech"],
        default="tech",
        help="Document type: prd or tech (default: tech)",
    )
    parser.add_argument(
        "--round", "-r", type=int, default=1, help="Current round number"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Total rounds completed (used with send-final)",
    )


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    """Add output formatting arguments to parser."""
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--show-cost", action="store_true", help="Show cost summary after critique"
    )


def add_telegram_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Telegram notification arguments to parser."""
    parser.add_argument(
        "--telegram",
        "-t",
        action="store_true",
        help="Send Telegram notifications and poll for feedback",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=60,
        help="Seconds to wait for Telegram reply (default: 60)",
    )


def add_critique_modifiers(parser: argparse.ArgumentParser) -> None:
    """Add critique modification arguments to parser."""
    parser.add_argument(
        "--press",
        "-p",
        action="store_true",
        help="Press models to confirm they read the full document (anti-laziness check)",
    )
    parser.add_argument(
        "--focus",
        "-f",
        help="Focus area for critique (security, scalability, performance, ux, reliability, cost)",
    )
    parser.add_argument(
        "--persona",
        help="Persona for critique (security-engineer, oncall-engineer, junior-developer, etc.)",
    )
    parser.add_argument(
        "--context",
        "-c",
        action="append",
        default=[],
        help="Additional context file(s) to include (can be used multiple times)",
    )
    parser.add_argument(
        "--preserve-intent",
        action="store_true",
        help="Require explicit justification for any removal or substantial modification",
    )


def add_session_arguments(parser: argparse.ArgumentParser) -> None:
    """Add session management arguments to parser."""
    parser.add_argument(
        "--session",
        "-s",
        help="Session ID for state persistence (enables checkpointing and resume)",
    )
    parser.add_argument("--resume", help="Resume a previous session by ID")


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Add profile management arguments to parser."""
    parser.add_argument("--profile", help="Load settings from a saved profile")


def add_diff_arguments(parser: argparse.ArgumentParser) -> None:
    """Add diff command arguments to parser."""
    parser.add_argument("--previous", help="Previous spec file (for diff action)")
    parser.add_argument("--current", help="Current spec file (for diff action)")


def add_codex_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Codex CLI arguments to parser."""
    parser.add_argument(
        "--codex-reasoning",
        default=DEFAULT_CODEX_REASONING,
        choices=["low", "medium", "high", "xhigh"],
        help=f"Reasoning effort for Codex CLI models (default: {DEFAULT_CODEX_REASONING})",
    )
    parser.add_argument(
        "--codex-search",
        action="store_true",
        help="Enable web search for Codex CLI models",
    )


def add_bedrock_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Bedrock arguments to parser."""
    parser.add_argument("--region", help="AWS region for Bedrock (e.g., us-east-1)")
    parser.add_argument(
        "bedrock_arg",
        nargs="?",
        help="Additional argument for bedrock subcommands (model name or alias target)",
    )


def add_misc_arguments(parser: argparse.ArgumentParser) -> None:
    """Add miscellaneous arguments to parser."""
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout in seconds for model API/CLI calls (default: 600 = 10 minutes)",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial spec debate with multiple LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  echo "spec" | python3 debate.py critique --models gpt-4o
  echo "spec" | python3 debate.py critique --models gpt-4o --focus security
  echo "spec" | python3 debate.py critique --models gpt-4o --persona "security engineer"
  echo "spec" | python3 debate.py critique --models gpt-4o --context ./api.md
  echo "spec" | python3 debate.py critique --profile my-security-profile
  python3 debate.py diff --previous old.md --current new.md
  echo "spec" | python3 debate.py export-tasks --doc-type prd
  python3 debate.py providers
  python3 debate.py focus-areas
  python3 debate.py personas
  python3 debate.py profiles
  python3 debate.py save-profile myprofile --models gpt-4o,gemini/gemini-2.0-flash --focus security

Bedrock commands:
  python3 debate.py bedrock status                           # Show Bedrock config
  python3 debate.py bedrock enable --region us-east-1        # Enable Bedrock mode
  python3 debate.py bedrock disable                          # Disable Bedrock mode
  python3 debate.py bedrock add-model claude-3-sonnet        # Add model to available list
  python3 debate.py bedrock remove-model claude-3-haiku      # Remove model from list
  python3 debate.py bedrock alias mymodel anthropic.claude-3-sonnet-20240229-v1:0  # Add custom alias

Document types:
  prd   - Product Requirements Document (business/product focus)
  tech  - Technical Specification / Architecture Document (engineering focus)
        """,
    )

    # Positional arguments
    parser.add_argument(
        "action",
        choices=[
            "critique",
            "providers",
            "send-final",
            "diff",
            "export-tasks",
            "focus-areas",
            "personas",
            "profiles",
            "save-profile",
            "sessions",
            "bedrock",
        ],
        help="Action to perform",
    )
    parser.add_argument(
        "profile_name",
        nargs="?",
        help="Profile name (for save-profile action) or bedrock subcommand",
    )

    # Add argument groups
    add_core_arguments(parser)
    add_output_arguments(parser)
    add_telegram_arguments(parser)
    add_critique_modifiers(parser)
    add_session_arguments(parser)
    add_profile_arguments(parser)
    add_diff_arguments(parser)
    add_codex_arguments(parser)
    add_bedrock_arguments(parser)
    add_misc_arguments(parser)

    return parser


def handle_info_command(args: argparse.Namespace) -> bool:
    """Handle info commands (providers, focus-areas, personas, profiles, sessions).

    Args:
        args: Parsed command-line arguments.

    Returns:
        True if command was handled, False otherwise.
    """
    if args.action == "providers":
        list_providers()
        return True

    if args.action == "focus-areas":
        list_focus_areas()
        return True

    if args.action == "personas":
        list_personas()
        return True

    if args.action == "profiles":
        list_profiles()
        return True

    if args.action == "sessions":
        sessions = SessionState.list_sessions()
        print("Saved Sessions:\n")
        if not sessions:
            print("  No sessions found.")
            print(f"\n  Sessions are stored in: {SESSIONS_DIR}")
            print("\n  Start a session with: --session <name>")
        else:
            for s in sessions:
                print(f"  {s['id']}")
                print(f"    round: {s['round']}, type: {s['doc_type']}")
                print(
                    f"    updated: {s['updated_at'][:19] if s['updated_at'] else 'unknown'}"
                )
                print()
        return True

    return False


def handle_utility_command(args: argparse.Namespace) -> bool:
    """Handle utility commands (bedrock, save-profile, diff).

    Args:
        args: Parsed command-line arguments.

    Returns:
        True if command was handled, False otherwise.
    """
    if args.action == "bedrock":
        subcommand = args.profile_name or "status"
        handle_bedrock_command(subcommand, args.bedrock_arg, args.region)
        return True

    if args.action == "save-profile":
        if not args.profile_name:
            print("Error: Profile name required", file=sys.stderr)
            sys.exit(1)
        config = {
            "models": args.models,
            "doc_type": args.doc_type,
            "focus": args.focus,
            "persona": args.persona,
            "context": args.context,
            "preserve_intent": args.preserve_intent,
        }
        save_profile(args.profile_name, config)
        return True

    if args.action == "diff":
        if not args.previous or not args.current:
            print("Error: --previous and --current required for diff", file=sys.stderr)
            sys.exit(1)
        try:
            prev_content = Path(args.previous).read_text()
            curr_content = Path(args.current).read_text()
            diff = generate_diff(prev_content, curr_content)
            if diff:
                print(diff)
            else:
                print("No differences found.")
        except OSError as e:
            print(f"Error reading files: {e}", file=sys.stderr)
            sys.exit(1)
        return True

    return False


def apply_profile(args: argparse.Namespace) -> None:
    """Apply profile settings to args if --profile specified.

    Args:
        args: Parsed command-line arguments (modified in place).
    """
    if not args.profile:
        return

    profile = load_profile(args.profile)
    if "models" in profile and args.models is None:
        args.models = profile["models"]
    if "doc_type" in profile and args.doc_type == "tech":
        args.doc_type = profile["doc_type"]
    if "focus" in profile and not args.focus:
        args.focus = profile["focus"]
    if "persona" in profile and not args.persona:
        args.persona = profile["persona"]
    if "context" in profile and not args.context:
        args.context = profile["context"]
    if profile.get("preserve_intent") and not args.preserve_intent:
        args.preserve_intent = profile["preserve_intent"]


def parse_models(args: argparse.Namespace) -> list[str]:
    """Parse and validate models list from args.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of model identifiers.
    """
    # If no models specified, use default based on available API keys
    if args.models is None:
        default_model = get_default_model()
        if default_model is None:
            print(
                "Error: No API keys configured and no models specified.",
                file=sys.stderr,
            )
            print("\nAvailable providers:", file=sys.stderr)
            print(
                "  OpenAI:    Set OPENAI_API_KEY for gpt-4o, o1, etc.", file=sys.stderr
            )
            print(
                "  Anthropic: Set ANTHROPIC_API_KEY for claude-sonnet-4-20250514, etc.",
                file=sys.stderr,
            )
            print(
                "  Google:    Set GEMINI_API_KEY for gemini/gemini-2.0-flash, etc.",
                file=sys.stderr,
            )
            print("  xAI:       Set XAI_API_KEY for xai/grok-3, etc.", file=sys.stderr)
            print(
                "  Mistral:   Set MISTRAL_API_KEY for mistral/mistral-large, etc.",
                file=sys.stderr,
            )
            print(
                "  Groq:      Set GROQ_API_KEY for groq/llama-3.3-70b-versatile, etc.",
                file=sys.stderr,
            )
            print(
                "  Deepseek:  Set DEEPSEEK_API_KEY for deepseek/deepseek-chat, etc.",
                file=sys.stderr,
            )
            print(
                "  Zhipu:     Set ZHIPUAI_API_KEY for zhipu/glm-4, etc.",
                file=sys.stderr,
            )
            print("\nOr specify models explicitly: --models gpt-4o", file=sys.stderr)
            print(
                "\nRun 'python3 debate.py providers' to see which keys are set.",
                file=sys.stderr,
            )
            sys.exit(2)
        args.models = default_model

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("Error: No models specified", file=sys.stderr)
        sys.exit(1)
    return models


def setup_bedrock(
    args: argparse.Namespace, models: list[str]
) -> tuple[list[str], bool, Optional[str]]:
    """Configure Bedrock mode and validate models.

    Args:
        args: Parsed command-line arguments.
        models: List of model identifiers.

    Returns:
        Tuple of (validated_models, bedrock_mode, bedrock_region).
    """
    bedrock_config = get_bedrock_config()
    bedrock_mode = bedrock_config.get("enabled", False)
    bedrock_region = bedrock_config.get("region")

    if not bedrock_mode or args.action != "critique":
        return models, bedrock_mode, bedrock_region

    available = bedrock_config.get("available_models", [])
    if not available:
        print(
            "Error: Bedrock mode is enabled but no models are configured.",
            file=sys.stderr,
        )
        print(
            "Add models with: python3 debate.py bedrock add-model claude-3-sonnet",
            file=sys.stderr,
        )
        print("Or disable Bedrock: python3 debate.py bedrock disable", file=sys.stderr)
        sys.exit(2)

    valid_models, invalid_models = validate_bedrock_models(models, bedrock_config)

    if invalid_models:
        print(
            "Error: The following models are not available in your Bedrock configuration:",
            file=sys.stderr,
        )
        for m in invalid_models:
            print(f"  - {m}", file=sys.stderr)
        print(f"\nAvailable models: {', '.join(available)}", file=sys.stderr)
        print(
            "Add models with: python3 debate.py bedrock add-model <model>",
            file=sys.stderr,
        )
        print("Or disable Bedrock: python3 debate.py bedrock disable", file=sys.stderr)
        sys.exit(2)

    print(
        f"Bedrock mode: routing through AWS Bedrock ({bedrock_region})",
        file=sys.stderr,
    )
    return valid_models, bedrock_mode, bedrock_region


def handle_send_final(args: argparse.Namespace, models: list[str]) -> None:
    """Handle send-final action.

    Args:
        args: Parsed command-line arguments.
        models: List of model identifiers.
    """
    spec = sys.stdin.read().strip()
    if not spec:
        print("Error: No spec provided via stdin", file=sys.stderr)
        sys.exit(1)
    if send_final_spec_to_telegram(spec, args.rounds, models, args.doc_type):
        print("Final document sent to Telegram.")
    else:
        print("Failed to send final document to Telegram.", file=sys.stderr)
        sys.exit(1)


def handle_export_tasks(args: argparse.Namespace, models: list[str]) -> None:
    """Handle export-tasks action.

    Args:
        args: Parsed command-line arguments.
        models: List of model identifiers.
    """
    spec = sys.stdin.read().strip()
    if not spec:
        print("Error: No spec provided via stdin", file=sys.stderr)
        sys.exit(1)

    doc_type_name = get_doc_type_name(args.doc_type)
    prompt = EXPORT_TASKS_PROMPT.format(doc_type_name=doc_type_name, spec=spec)

    try:
        # Build completion kwargs
        completion_kwargs = {
            "model": models[0],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8000,
        }

        # O-series models don't support custom temperature
        if not is_o_series_model(models[0]):
            completion_kwargs["temperature"] = 0.3

        response = completion(**completion_kwargs)
        content = response.choices[0].message.content
        tasks = extract_tasks(content)

        if args.json:
            print(json.dumps({"tasks": tasks}, indent=2))
        else:
            print(f"\n=== Extracted {len(tasks)} Tasks ===\n")
            for i, task in enumerate(tasks, 1):
                print(
                    f"{i}. [{task.get('type', 'task')}] [{task.get('priority', 'medium')}] {task.get('title', 'Untitled')}"
                )
                if task.get("description"):
                    print(f"   {task['description'][:100]}...")
                if task.get("acceptance_criteria"):
                    print(
                        f"   Acceptance criteria: {len(task['acceptance_criteria'])} items"
                    )
                print()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def load_or_resume_session(
    args: argparse.Namespace, models: list[str]
) -> tuple[str, Optional[SessionState], list[str]]:
    """Load session from resume or stdin, optionally creating new session.

    Args:
        args: Parsed command-line arguments.
        models: List of model identifiers.

    Returns:
        Tuple of (spec, session_state, models).
    """
    session_state = None

    if args.resume:
        try:
            session_state = SessionState.load(args.resume)
            print(
                f"Resuming session '{args.resume}' at round {session_state.round}",
                file=sys.stderr,
            )
            spec = session_state.spec
            args.round = session_state.round
            args.doc_type = session_state.doc_type
            args.models = ",".join(session_state.models)
            if session_state.focus:
                args.focus = session_state.focus
            if session_state.persona:
                args.persona = session_state.persona
            if session_state.preserve_intent:
                args.preserve_intent = session_state.preserve_intent
            models = session_state.models
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        spec = sys.stdin.read().strip()
        if not spec:
            print("Error: No spec provided via stdin", file=sys.stderr)
            sys.exit(1)

    if args.session and not session_state:
        session_state = SessionState(
            session_id=args.session,
            spec=spec,
            round=args.round,
            doc_type=args.doc_type,
            models=models,
            focus=args.focus,
            persona=args.persona,
            preserve_intent=args.preserve_intent,
            created_at=datetime.now().isoformat(),
        )
        session_state.save()
        print(f"Session '{args.session}' created", file=sys.stderr)

    return spec, session_state, models


def run_critique(
    args: argparse.Namespace,
    spec: str,
    models: list[str],
    session_state: Optional[SessionState],
    context: Optional[str],
    bedrock_mode: bool,
    bedrock_region: Optional[str],
) -> None:
    """Execute the critique workflow and output results.

    Args:
        args: Parsed command-line arguments.
        spec: The specification to critique.
        models: List of model identifiers.
        session_state: Optional session state for persistence.
        context: Optional context string.
        bedrock_mode: Whether Bedrock mode is enabled.
        bedrock_region: AWS region for Bedrock.
    """
    mode = "pressing for confirmation" if args.press else "critiquing"
    focus_info = f" (focus: {args.focus})" if args.focus else ""
    persona_info = f" (persona: {args.persona})" if args.persona else ""
    preserve_info = " (preserve-intent)" if args.preserve_intent else ""
    search_info = " (search)" if args.codex_search else ""
    print(
        f"Calling {len(models)} model(s) ({mode}){focus_info}{persona_info}{preserve_info}{search_info}: {', '.join(models)}...",
        file=sys.stderr,
    )

    results = call_models_parallel(
        models,
        spec,
        args.round,
        args.doc_type,
        args.press,
        args.focus,
        args.persona,
        context,
        args.preserve_intent,
        args.codex_reasoning,
        args.codex_search,
        args.timeout,
        bedrock_mode,
        bedrock_region,
    )

    errors = [r for r in results if r.error]
    for err_result in errors:
        print(
            f"Warning: {err_result.model} returned error: {err_result.error}",
            file=sys.stderr,
        )

    successful = [r for r in results if not r.error]
    all_agreed = all(r.agreed for r in successful) if successful else False

    session_id = session_state.session_id if session_state else args.session
    if session_id or args.session:
        save_checkpoint(spec, args.round, session_id)

    latest_spec = spec
    for r in successful:
        if r.spec:
            latest_spec = r.spec
            break

    if session_state:
        session_state.spec = latest_spec
        session_state.round = args.round + 1
        session_state.history.append(
            {
                "round": args.round,
                "all_agreed": all_agreed,
                "models": [
                    {"model": r.model, "agreed": r.agreed, "error": r.error}
                    for r in results
                ],
            }
        )
        session_state.save()

    user_feedback = None
    if args.telegram:
        user_feedback = send_telegram_notification(
            models, args.round, results, args.poll_timeout
        )
        if user_feedback:
            print(f"Received feedback: {user_feedback}", file=sys.stderr)

    output_results(args, results, models, all_agreed, user_feedback, session_state)


def output_results(
    args: argparse.Namespace,
    results: list[ModelResponse],
    models: list[str],
    all_agreed: bool,
    user_feedback: Optional[str],
    session_state: Optional[SessionState],
) -> None:
    """Output critique results in JSON or text format.

    Args:
        args: Parsed command-line arguments.
        results: List of model responses.
        models: List of model identifiers.
        all_agreed: Whether all models agreed.
        user_feedback: Optional user feedback from Telegram.
        session_state: Optional session state.
    """
    if args.json:
        output: dict[str, Any] = {
            "all_agreed": all_agreed,
            "round": args.round,
            "doc_type": args.doc_type,
            "models": models,
            "focus": args.focus,
            "persona": args.persona,
            "preserve_intent": args.preserve_intent,
            "session": session_state.session_id if session_state else args.session,
            "results": [
                {
                    "model": r.model,
                    "agreed": r.agreed,
                    "response": r.response,
                    "spec": r.spec,
                    "error": r.error,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cost": r.cost,
                }
                for r in results
            ],
            "cost": {
                "total": cost_tracker.total_cost,
                "input_tokens": cost_tracker.total_input_tokens,
                "output_tokens": cost_tracker.total_output_tokens,
                "by_model": cost_tracker.by_model,
            },
        }
        if user_feedback:
            output["user_feedback"] = user_feedback
        print(json.dumps(output, indent=2))
    else:
        doc_type_name = get_doc_type_name(args.doc_type)
        print(f"\n=== Round {args.round} Results ({doc_type_name}) ===\n")

        for r in results:
            print(f"--- {r.model} ---")
            if r.error:
                print(f"ERROR: {r.error}")
            elif r.agreed:
                print("[AGREE]")
            else:
                print(r.response)
            print()

        if all_agreed:
            print("=== ALL MODELS AGREE ===")
        else:
            successful = [r for r in results if not r.error]
            agreed_models = [r.model for r in successful if r.agreed]
            disagreed_models = [r.model for r in successful if not r.agreed]
            if agreed_models:
                print(f"Agreed: {', '.join(agreed_models)}")
            if disagreed_models:
                print(f"Critiqued: {', '.join(disagreed_models)}")

        if user_feedback:
            print()
            print("=== User Feedback ===")
            print(user_feedback)

        if args.show_cost:
            print(cost_tracker.summary())


def validate_models_before_run(models: list[str], bedrock_mode: bool) -> None:
    """
    Validate that models have required credentials before running critique.

    Args:
        models: List of model identifiers.
        bedrock_mode: Whether Bedrock mode is enabled.
    """
    if bedrock_mode:
        # Bedrock validation is handled in setup_bedrock
        return

    valid, invalid = validate_model_credentials(models)

    if invalid:
        print("Error: The following models lack required API keys:", file=sys.stderr)
        for model in invalid:
            # Determine which key is needed
            if model.startswith("gpt-") or model.startswith("o1"):
                print(f"  - {model} (requires OPENAI_API_KEY)", file=sys.stderr)
            elif model.startswith("claude-"):
                print(f"  - {model} (requires ANTHROPIC_API_KEY)", file=sys.stderr)
            elif model.startswith("gemini/"):
                print(f"  - {model} (requires GEMINI_API_KEY)", file=sys.stderr)
            elif model.startswith("xai/"):
                print(f"  - {model} (requires XAI_API_KEY)", file=sys.stderr)
            elif model.startswith("mistral/"):
                print(f"  - {model} (requires MISTRAL_API_KEY)", file=sys.stderr)
            elif model.startswith("groq/"):
                print(f"  - {model} (requires GROQ_API_KEY)", file=sys.stderr)
            elif model.startswith("deepseek/"):
                print(f"  - {model} (requires DEEPSEEK_API_KEY)", file=sys.stderr)
            elif model.startswith("zhipu/"):
                print(f"  - {model} (requires ZHIPUAI_API_KEY)", file=sys.stderr)
            elif model.startswith("codex/"):
                print(
                    f"  - {model} (requires Codex CLI: npm install -g @openai/codex && codex login)",
                    file=sys.stderr,
                )
            else:
                print(f"  - {model} (unknown provider)", file=sys.stderr)

        print(
            "\nRun 'python3 debate.py providers' to see which API keys are configured.",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> None:
    """Entry point for the debate CLI."""
    parser = create_parser()
    args = parser.parse_args()

    if handle_info_command(args):
        return

    if handle_utility_command(args):
        return

    apply_profile(args)
    models = parse_models(args)
    context = load_context_files(args.context) if args.context else None
    models, bedrock_mode, bedrock_region = setup_bedrock(args, models)

    # Validate models have required credentials
    validate_models_before_run(models, bedrock_mode)

    if args.action == "send-final":
        handle_send_final(args, models)
        return

    if args.action == "export-tasks":
        handle_export_tasks(args, models)
        return

    spec, session_state, models = load_or_resume_session(args, models)
    run_critique(
        args, spec, models, session_state, context, bedrock_mode, bedrock_region
    )


if __name__ == "__main__":
    main()
