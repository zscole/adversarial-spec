"""Provider configuration, Bedrock support, and profile management."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from prompts import FOCUS_AREAS, PERSONAS

PROFILES_DIR = Path.home() / ".config" / "adversarial-spec" / "profiles"
GLOBAL_CONFIG_PATH = Path.home() / ".claude" / "adversarial-spec" / "config.json"

# Cost per 1M tokens (approximate, as of 2024)
MODEL_COSTS = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "gemini/gemini-2.0-flash": {"input": 0.075, "output": 0.30},
    "gemini/gemini-pro": {"input": 0.50, "output": 1.50},
    "xai/grok-3": {"input": 3.00, "output": 15.00},
    "xai/grok-beta": {"input": 5.00, "output": 15.00},
    "mistral/mistral-large": {"input": 2.00, "output": 6.00},
    "groq/llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "deepseek/deepseek-chat": {"input": 0.14, "output": 0.28},
    "zhipu/glm-4": {"input": 1.40, "output": 1.40},
    "zhipu/glm-4-plus": {"input": 7.00, "output": 7.00},
    # Codex CLI models (uses ChatGPT subscription, no per-token cost)
    "codex/gpt-5.3-codex": {"input": 0.0, "output": 0.0},
    "codex/gpt-5.2-codex": {"input": 0.0, "output": 0.0},
    "codex/gpt-5.1-codex-max": {"input": 0.0, "output": 0.0},
    "codex/gpt-5.1-codex-mini": {"input": 0.0, "output": 0.0},
    # Gemini CLI models (uses Google account, no per-token cost)
    "gemini-cli/gemini-3-pro-preview": {"input": 0.0, "output": 0.0},
    "gemini-cli/gemini-3-flash-preview": {"input": 0.0, "output": 0.0},
}

DEFAULT_COST = {"input": 5.00, "output": 15.00}

# Check if Codex CLI is available
CODEX_AVAILABLE = shutil.which("codex") is not None

# Check if Gemini CLI is available
GEMINI_CLI_AVAILABLE = shutil.which("gemini") is not None

# Default reasoning effort for Codex CLI (minimal, low, medium, high, xhigh)
DEFAULT_CODEX_REASONING = "xhigh"

# Bedrock model mapping: friendly names -> Bedrock model IDs
BEDROCK_MODEL_MAP = {
    # Anthropic Claude models
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3.5-sonnet-v2": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3.5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    # Meta Llama models
    "llama-3-8b": "meta.llama3-8b-instruct-v1:0",
    "llama-3-70b": "meta.llama3-70b-instruct-v1:0",
    "llama-3.1-8b": "meta.llama3-1-8b-instruct-v1:0",
    "llama-3.1-70b": "meta.llama3-1-70b-instruct-v1:0",
    "llama-3.1-405b": "meta.llama3-1-405b-instruct-v1:0",
    # Mistral models
    "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
    "mistral-large": "mistral.mistral-large-2402-v1:0",
    "mixtral-8x7b": "mistral.mixtral-8x7b-instruct-v0:1",
    # Amazon Titan models
    "titan-text-express": "amazon.titan-text-express-v1",
    "titan-text-lite": "amazon.titan-text-lite-v1",
    # Cohere models
    "cohere-command": "cohere.command-text-v14",
    "cohere-command-light": "cohere.command-light-text-v14",
    "cohere-command-r": "cohere.command-r-v1:0",
    "cohere-command-r-plus": "cohere.command-r-plus-v1:0",
    # AI21 models
    "ai21-jamba": "ai21.jamba-instruct-v1:0",
}


def load_global_config() -> dict:
    """Load global config from ~/.claude/adversarial-spec/config.json."""
    if not GLOBAL_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(GLOBAL_CONFIG_PATH.read_text())
    except json.JSONDecodeError as e:
        print(f"Warning: Invalid JSON in global config: {e}", file=sys.stderr)
        return {}


def save_global_config(config: dict):
    """Save global config to ~/.claude/adversarial-spec/config.json."""
    GLOBAL_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    GLOBAL_CONFIG_PATH.write_text(json.dumps(config, indent=2))


def is_bedrock_enabled() -> bool:
    """Check if Bedrock mode is enabled in global config."""
    config = load_global_config()
    return config.get("bedrock", {}).get("enabled", False)


def get_bedrock_config() -> dict:
    """Get Bedrock configuration from global config."""
    config = load_global_config()
    return config.get("bedrock", {})


def resolve_bedrock_model(
    friendly_name: str, config: Optional[dict] = None
) -> Optional[str]:
    """
    Resolve a friendly model name to a Bedrock model ID.

    Checks in order:
    1. If already a full Bedrock ID (contains '.'), return as-is
    2. Built-in BEDROCK_MODEL_MAP
    3. Custom aliases in config

    Returns None if not found.
    """
    # If it looks like a full Bedrock ID, return as-is
    if "." in friendly_name and not friendly_name.startswith("bedrock/"):
        return friendly_name

    # Check built-in map
    if friendly_name in BEDROCK_MODEL_MAP:
        return BEDROCK_MODEL_MAP[friendly_name]

    # Check custom aliases in config
    if config is None:
        config = get_bedrock_config()
    custom_aliases = config.get("custom_aliases", {})
    if friendly_name in custom_aliases:
        return custom_aliases[friendly_name]

    return None


def validate_bedrock_models(
    models: list[str], config: Optional[dict] = None
) -> tuple[list[str], list[str]]:
    """
    Validate that requested models are available in Bedrock config.

    Returns (valid_models, invalid_models) where valid_models are resolved to Bedrock IDs.
    """
    if config is None:
        config = get_bedrock_config()

    available = config.get("available_models", [])
    valid = []
    invalid = []

    for model in models:
        # Check if model is in available list (by friendly name or full ID)
        if model in available:
            resolved = resolve_bedrock_model(model, config)
            if resolved:
                valid.append(resolved)
            else:
                invalid.append(model)
        else:
            # Also check if it's a full Bedrock ID that matches an available friendly name
            resolved = resolve_bedrock_model(model, config)
            if resolved:
                # Check if the friendly name version is available
                for avail in available:
                    if resolve_bedrock_model(avail, config) == resolved:
                        valid.append(resolved)
                        break
                else:
                    invalid.append(model)
            else:
                invalid.append(model)

    return valid, invalid


def load_profile(profile_name: str) -> dict:
    """Load a saved profile by name."""
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    if not profile_path.exists():
        print(
            f"Error: Profile '{profile_name}' not found at {profile_path}",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        return json.loads(profile_path.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in profile '{profile_name}': {e}", file=sys.stderr)
        sys.exit(2)


def save_profile(profile_name: str, config: dict):
    """Save a profile to disk."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    profile_path.write_text(json.dumps(config, indent=2))
    print(f"Profile saved to {profile_path}")


def list_profiles():
    """List all saved profiles."""
    print("Saved Profiles:\n")
    if not PROFILES_DIR.exists():
        print("  No profiles found.")
        print(f"\n  Profiles are stored in: {PROFILES_DIR}")
        print(
            "\n  Create a profile with: python3 debate.py save-profile <name> --models ... --focus ..."
        )
        return

    profiles = list(PROFILES_DIR.glob("*.json"))
    if not profiles:
        print("  No profiles found.")
        return

    for p in sorted(profiles):
        try:
            config = json.loads(p.read_text())
            name = p.stem
            models = config.get("models", "not set")
            focus = config.get("focus", "none")
            persona = config.get("persona", "none")
            preserve = "yes" if config.get("preserve_intent") else "no"
            print(f"  {name}")
            print(f"    models: {models}")
            print(f"    focus: {focus}")
            print(f"    persona: {persona}")
            print(f"    preserve-intent: {preserve}")
            print()
        except Exception:
            print(f"  {p.stem} [error reading]")


def list_providers():
    """List all supported providers and their API key status."""
    # Show Bedrock status first if configured
    bedrock_config = get_bedrock_config()
    if bedrock_config.get("enabled"):
        print("AWS Bedrock (Active):\n")
        print("  Status:  ENABLED - All models route through Bedrock")
        print(f"  Region:  {bedrock_config.get('region', 'not set')}")
        available = bedrock_config.get("available_models", [])
        print(
            f"  Models:  {', '.join(available) if available else '(none configured)'}"
        )

        # Check AWS credentials
        aws_creds = bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            or os.environ.get("AWS_PROFILE")
            or os.environ.get("AWS_ROLE_ARN")
        )
        print(f"  AWS Credentials: {'[available]' if aws_creds else '[not detected]'}")
        print()
        print(
            "  Run 'python3 debate.py bedrock status' for full Bedrock configuration."
        )
        print(
            "  Run 'python3 debate.py bedrock disable' to use direct API keys instead.\n"
        )
        print("-" * 60 + "\n")

    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-4o, gpt-4-turbo, o1"),
        (
            "Anthropic",
            "ANTHROPIC_API_KEY",
            "claude-sonnet-4-20250514, claude-opus-4-20250514",
        ),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-2.0-flash, gemini/gemini-pro"),
        ("xAI", "XAI_API_KEY", "xai/grok-3, xai/grok-beta"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large, mistral/codestral"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("Together", "TOGETHER_API_KEY", "together_ai/meta-llama/Llama-3-70b"),
        (
            "OpenRouter",
            "OPENROUTER_API_KEY",
            "openrouter/openai/gpt-4o, openrouter/anthropic/claude-3.5-sonnet",
        ),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
        ("Zhipu", "ZHIPUAI_API_KEY", "zhipu/glm-4, zhipu/glm-4-plus"),
    ]

    if bedrock_config.get("enabled"):
        print("Direct API Providers (inactive while Bedrock is enabled):\n")
    else:
        print("Supported providers:\n")

    for name, key, models in providers:
        status = "[set]" if os.environ.get(key) else "[not set]"
        print(f"  {name:12} {key:24} {status}")
        print(f"             Example models: {models}")
        print()

    # Codex CLI (uses ChatGPT subscription, not API key)
    codex_status = "[installed]" if CODEX_AVAILABLE else "[not installed]"
    print(f"  {'Codex CLI':12} {'(ChatGPT subscription)':24} {codex_status}")
    print("             Example models: codex/gpt-5.3-codex, codex/gpt-5.2-codex")
    print(
        "             Reasoning: --codex-reasoning (minimal, low, medium, high, xhigh)"
    )
    print("             Install: npm install -g @openai/codex && codex login")
    print()

    # Gemini CLI (uses Google account, not API key)
    gemini_cli_status = "[installed]" if GEMINI_CLI_AVAILABLE else "[not installed]"
    print(f"  {'Gemini CLI':12} {'(Google account)':24} {gemini_cli_status}")
    print(
        "             Example models: gemini-cli/gemini-3-pro-preview, gemini-cli/gemini-3-flash-preview"
    )
    print("             Install: npm install -g @google/gemini-cli && gemini auth")
    print()

    # Show Bedrock option if not enabled
    if not bedrock_config.get("enabled"):
        print("AWS Bedrock:\n")
        print(
            "  Not configured. Enable with: python3 debate.py bedrock enable --region us-east-1"
        )
        print()


def list_focus_areas():
    """List available focus areas."""
    print("Available focus areas (--focus):\n")
    for name, description in FOCUS_AREAS.items():
        first_line = (
            description.strip().split("\n")[1]
            if "\n" in description
            else description[:60]
        )
        print(f"  {name:15} {first_line.strip()[:60]}")
    print()


def list_personas():
    """List available personas."""
    print("Available personas (--persona):\n")
    for name, description in PERSONAS.items():
        print(f"  {name}")
        print(f"    {description[:80]}...")
        print()


def get_available_providers() -> list[tuple[str, Optional[str], str]]:
    """
    Get list of providers with configured API keys.

    Returns:
        List of (provider_name, env_var, default_model) tuples for providers with API keys set.
        Note: env_var can be None for providers like Codex CLI that use alternative auth.
    """
    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-4o"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-20250514"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-2.0-flash"),
        ("xAI", "XAI_API_KEY", "xai/grok-3"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("OpenRouter", "OPENROUTER_API_KEY", "openrouter/openai/gpt-4o"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
        ("Zhipu", "ZHIPUAI_API_KEY", "zhipu/glm-4"),
    ]

    available: list[tuple[str, Optional[str], str]] = []
    for name, key, model in providers:
        if os.environ.get(key):
            available.append((name, key, model))

    # Add Codex CLI if available
    if CODEX_AVAILABLE:
        available.append(("Codex CLI", None, "codex/gpt-5.3-codex"))

    # Add Gemini CLI if available
    if GEMINI_CLI_AVAILABLE:
        available.append(("Gemini CLI", None, "gemini-cli/gemini-3-pro-preview"))

    return available


def get_default_model() -> Optional[str]:
    """
    Get a default model based on available API keys.

    Checks Bedrock first, then API keys in priority order.

    Returns:
        Model name string, or None if no API keys are configured.
    """
    # Check Bedrock first
    bedrock_config = get_bedrock_config()
    if bedrock_config.get("enabled"):
        available_models = bedrock_config.get("available_models", [])
        if available_models:
            return available_models[0]

    # Check API keys
    available = get_available_providers()
    if available:
        return available[0][2]  # Return default model from first available provider

    return None


def validate_model_credentials(models: list[str]) -> tuple[list[str], list[str]]:
    """
    Validate that API keys are available for requested models.

    Args:
        models: List of model identifiers.

    Returns:
        Tuple of (valid_models, invalid_models) where invalid_models lack credentials.
    """
    bedrock_config = get_bedrock_config()

    # If Bedrock is enabled, validate against Bedrock models
    if bedrock_config.get("enabled"):
        return validate_bedrock_models(models, bedrock_config)

    valid = []
    invalid = []

    provider_map = {
        "gpt-": "OPENAI_API_KEY",
        "o1": "OPENAI_API_KEY",
        "claude-": "ANTHROPIC_API_KEY",
        "gemini/": "GEMINI_API_KEY",
        "xai/": "XAI_API_KEY",
        "mistral/": "MISTRAL_API_KEY",
        "groq/": "GROQ_API_KEY",
        "deepseek/": "DEEPSEEK_API_KEY",
        "zhipu/": "ZHIPUAI_API_KEY",
        "codex/": None,  # Uses ChatGPT subscription, not API key
        "gemini-cli/": None,  # Uses Google account, not API key
    }

    for model in models:
        # Check if it's a Codex model
        if model.startswith("codex/"):
            if CODEX_AVAILABLE:
                valid.append(model)
            else:
                invalid.append(model)
            continue

        # Check if it's a Gemini CLI model
        if model.startswith("gemini-cli/"):
            if GEMINI_CLI_AVAILABLE:
                valid.append(model)
            else:
                invalid.append(model)
            continue

        # Find matching provider
        required_key = None
        for prefix, key in provider_map.items():
            if model.startswith(prefix):
                required_key = key
                break

        # If no provider match found, assume it needs validation later
        if required_key is None:
            valid.append(model)
            continue

        # Check if API key is set
        if os.environ.get(required_key):
            valid.append(model)
        else:
            invalid.append(model)

    return valid, invalid


def handle_bedrock_command(subcommand: str, arg: Optional[str], region: Optional[str]):
    """Handle bedrock subcommands: status, enable, disable, add-model, remove-model, alias."""
    config = load_global_config()
    bedrock = config.get("bedrock", {})

    if subcommand == "status":
        print("Bedrock Configuration:\n")
        if not bedrock:
            print("  Status: Not configured")
            print(f"\n  Config path: {GLOBAL_CONFIG_PATH}")
            print("\n  To enable: python3 debate.py bedrock enable --region us-east-1")
            return

        enabled = bedrock.get("enabled", False)
        print(f"  Status: {'Enabled' if enabled else 'Disabled'}")
        print(f"  Region: {bedrock.get('region', 'not set')}")
        print(f"  Config path: {GLOBAL_CONFIG_PATH}")

        available = bedrock.get("available_models", [])
        print(f"\n  Available models ({len(available)}):")
        if available:
            for model in available:
                resolved = resolve_bedrock_model(model, bedrock)
                if resolved and resolved != model:
                    print(f"    - {model} -> {resolved}")
                else:
                    print(f"    - {model}")
        else:
            print("    (none configured)")
            print(
                "\n    Add models with: python3 debate.py bedrock add-model claude-3-sonnet"
            )

        aliases = bedrock.get("custom_aliases", {})
        if aliases:
            print(f"\n  Custom aliases ({len(aliases)}):")
            for alias, target in aliases.items():
                print(f"    - {alias} -> {target}")

        # Show available friendly names
        print(f"\n  Built-in model mappings ({len(BEDROCK_MODEL_MAP)}):")
        for name in sorted(BEDROCK_MODEL_MAP.keys())[:5]:
            print(f"    - {name}")
        if len(BEDROCK_MODEL_MAP) > 5:
            print(f"    ... and {len(BEDROCK_MODEL_MAP) - 5} more")

    elif subcommand == "enable":
        if not region:
            print("Error: --region is required for 'bedrock enable'", file=sys.stderr)
            print(
                "Example: python3 debate.py bedrock enable --region us-east-1",
                file=sys.stderr,
            )
            sys.exit(1)

        bedrock["enabled"] = True
        bedrock["region"] = region
        if "available_models" not in bedrock:
            bedrock["available_models"] = []
        if "custom_aliases" not in bedrock:
            bedrock["custom_aliases"] = {}

        config["bedrock"] = bedrock
        save_global_config(config)
        print(f"Bedrock mode enabled (region: {region})")
        print(f"Config saved to: {GLOBAL_CONFIG_PATH}")

        if not bedrock.get("available_models"):
            print(
                "\nNext: Add models with: python3 debate.py bedrock add-model claude-3-sonnet"
            )

    elif subcommand == "disable":
        bedrock["enabled"] = False
        config["bedrock"] = bedrock
        save_global_config(config)
        print("Bedrock mode disabled")

    elif subcommand == "add-model":
        if not arg:
            print("Error: Model name required for 'bedrock add-model'", file=sys.stderr)
            print(
                "Example: python3 debate.py bedrock add-model claude-3-sonnet",
                file=sys.stderr,
            )
            sys.exit(1)

        # Validate model name
        resolved = resolve_bedrock_model(arg, bedrock)
        if not resolved:
            print(
                f"Warning: '{arg}' is not a known Bedrock model. Adding anyway.",
                file=sys.stderr,
            )
            print(
                "Use 'python3 debate.py bedrock alias' to map it to a Bedrock model ID.",
                file=sys.stderr,
            )

        available = bedrock.get("available_models", [])
        if arg in available:
            print(f"Model '{arg}' is already in the available list")
            return

        available.append(arg)
        bedrock["available_models"] = available
        config["bedrock"] = bedrock
        save_global_config(config)

        if resolved:
            print(f"Added model: {arg} -> {resolved}")
        else:
            print(f"Added model: {arg}")

    elif subcommand == "remove-model":
        if not arg:
            print(
                "Error: Model name required for 'bedrock remove-model'", file=sys.stderr
            )
            sys.exit(1)

        available = bedrock.get("available_models", [])
        if arg not in available:
            print(f"Model '{arg}' is not in the available list", file=sys.stderr)
            sys.exit(1)

        available.remove(arg)
        bedrock["available_models"] = available
        config["bedrock"] = bedrock
        save_global_config(config)
        print(f"Removed model: {arg}")

    elif subcommand == "alias":
        if not arg:
            print(
                "Error: Alias name and target required for 'bedrock alias'",
                file=sys.stderr,
            )
            print(
                "Example: python3 debate.py bedrock alias mymodel anthropic.claude-3-sonnet-20240229-v1:0",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            "Error: 'bedrock alias' requires two arguments: alias_name and model_id",
            file=sys.stderr,
        )
        print(
            "Example: python3 debate.py bedrock alias mymodel anthropic.claude-3-sonnet-20240229-v1:0",
            file=sys.stderr,
        )
        print("\nAlternatively, edit the config file directly:", file=sys.stderr)
        print(f"  {GLOBAL_CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)

    elif subcommand == "list-models":
        print("Built-in Bedrock model mappings:\n")
        for name, bedrock_id in sorted(BEDROCK_MODEL_MAP.items()):
            print(f"  {name:25} -> {bedrock_id}")

    else:
        print(f"Unknown bedrock subcommand: {subcommand}", file=sys.stderr)
        print(
            "Available subcommands: status, enable, disable, add-model, remove-model, alias, list-models",
            file=sys.stderr,
        )
        sys.exit(1)
