#!/usr/bin/env python3
"""
Adversarial spec debate script.
Sends specs to multiple LLMs for critique using LiteLLM.

Usage:
    echo "spec" | python3 debate.py critique --models gpt-4o
    echo "spec" | python3 debate.py critique --models gpt-4o,gemini/gemini-2.0-flash,xai/grok-3 --doc-type prd
    echo "spec" | python3 debate.py critique --models codex/gpt-5.2-codex,gemini/gemini-2.0-flash --doc-type tech
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
    OpenAI:    OPENAI_API_KEY      models: gpt-4o, gpt-4-turbo, o1, etc.
    Anthropic: ANTHROPIC_API_KEY   models: claude-sonnet-4-20250514, claude-opus-4-20250514, etc.
    Google:    GEMINI_API_KEY      models: gemini/gemini-2.0-flash, gemini/gemini-pro, etc.
    xAI:       XAI_API_KEY         models: xai/grok-3, xai/grok-beta, etc.
    Mistral:   MISTRAL_API_KEY     models: mistral/mistral-large, etc.
    Groq:      GROQ_API_KEY        models: groq/llama-3.3-70b, etc.
    Codex CLI: (ChatGPT subscription) models: codex/gpt-5.2-codex, codex/gpt-5.1-codex-max
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

import os
import sys
import argparse
import json
import difflib
import time
import subprocess
import shutil
import concurrent.futures
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

os.environ["LITELLM_LOG"] = "ERROR"

try:
    import litellm
    from litellm import completion
    litellm.suppress_debug_info = True
except ImportError:
    print("Error: litellm package not installed. Run: pip install litellm", file=sys.stderr)
    sys.exit(1)

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
    # Codex CLI models (uses ChatGPT subscription, no per-token cost)
    "codex/gpt-5.2-codex": {"input": 0.0, "output": 0.0},
    "codex/gpt-5.1-codex-max": {"input": 0.0, "output": 0.0},
    "codex/gpt-5.1-codex-mini": {"input": 0.0, "output": 0.0},
}

# Default reasoning effort for Codex CLI (minimal, low, medium, high, xhigh)
DEFAULT_CODEX_REASONING = "xhigh"

DEFAULT_COST = {"input": 5.00, "output": 15.00}

# Check if Codex CLI is available
CODEX_AVAILABLE = shutil.which("codex") is not None

PROFILES_DIR = Path.home() / ".config" / "adversarial-spec" / "profiles"
SESSIONS_DIR = Path.home() / ".config" / "adversarial-spec" / "sessions"
CHECKPOINTS_DIR = Path.cwd() / ".adversarial-spec-checkpoints"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds

PRESERVE_INTENT_PROMPT = """
**PRESERVE ORIGINAL INTENT**
This document represents deliberate design choices. Before suggesting ANY removal or substantial modification:

1. ASSUME the author had good reasons for including each element
2. For EVERY removal or substantial change you propose, you MUST:
   - Quote the exact text you want to remove/change
   - Explain what problem it causes (not just "unnecessary" or "could be simpler")
   - Describe the concrete harm if it remains vs the benefit of removal
   - Consider: Is this genuinely wrong, or just different from what you'd write?

3. Distinguish between:
   - ERRORS: Factually wrong, contradictory, or technically broken (remove/fix these)
   - RISKS: Security holes, scalability issues, missing error handling (flag these)
   - PREFERENCES: Different style, structure, or approach (DO NOT remove these)

4. If something seems unusual but isn't broken, ASK about it rather than removing it:
   "The spec includes X which is unconventional. Was this intentional? If so, consider documenting the rationale."

5. Your critique should ADD protective detail, not sand off distinctive choices.

Treat removal like a code review: additions are cheap, deletions require justification.
"""

FOCUS_AREAS = {
    "security": """
**CRITICAL FOCUS: SECURITY**
Prioritize security analysis above all else. Specifically examine:
- Authentication and authorization mechanisms
- Input validation and sanitization
- SQL injection, XSS, CSRF, SSRF vulnerabilities
- Secret management and credential handling
- Data encryption at rest and in transit
- API security (rate limiting, authentication)
- Dependency vulnerabilities
- Privilege escalation risks
- Audit logging for security events
Flag any security gaps as blocking issues.""",

    "scalability": """
**CRITICAL FOCUS: SCALABILITY**
Prioritize scalability analysis above all else. Specifically examine:
- Horizontal vs vertical scaling strategy
- Database sharding and replication
- Caching strategy and invalidation
- Queue and async processing design
- Connection pooling and resource limits
- CDN and edge caching
- Microservices boundaries and communication
- Load balancing strategy
- Capacity planning and growth projections
Flag any scalability gaps as blocking issues.""",

    "performance": """
**CRITICAL FOCUS: PERFORMANCE**
Prioritize performance analysis above all else. Specifically examine:
- Latency targets (p50, p95, p99)
- Throughput requirements
- Database query optimization
- N+1 query problems
- Memory usage and leaks
- CPU-bound vs I/O-bound operations
- Caching effectiveness
- Network round trips
- Asset optimization
Flag any performance gaps as blocking issues.""",

    "ux": """
**CRITICAL FOCUS: USER EXPERIENCE**
Prioritize UX analysis above all else. Specifically examine:
- User journey clarity and completeness
- Error states and recovery flows
- Loading states and perceived performance
- Accessibility (WCAG compliance)
- Mobile vs desktop experience
- Internationalization readiness
- Onboarding flow
- Edge cases in user interactions
- Feedback and confirmation patterns
Flag any UX gaps as blocking issues.""",

    "reliability": """
**CRITICAL FOCUS: RELIABILITY**
Prioritize reliability analysis above all else. Specifically examine:
- Failure modes and recovery
- Circuit breakers and fallbacks
- Retry strategies with backoff
- Data consistency guarantees
- Backup and disaster recovery
- Health checks and readiness probes
- Graceful degradation
- SLA/SLO definitions
- Incident response procedures
Flag any reliability gaps as blocking issues.""",

    "cost": """
**CRITICAL FOCUS: COST EFFICIENCY**
Prioritize cost analysis above all else. Specifically examine:
- Infrastructure cost projections
- Resource utilization efficiency
- Auto-scaling policies
- Reserved vs on-demand resources
- Data transfer costs
- Third-party service costs
- Build vs buy decisions
- Operational overhead
- Cost monitoring and alerts
Flag any cost efficiency gaps as blocking issues.""",
}

PERSONAS = {
    "security-engineer": "You are a senior security engineer with 15 years of experience in application security, penetration testing, and secure architecture design. You think like an attacker and are paranoid about edge cases.",

    "oncall-engineer": "You are the on-call engineer who will be paged at 3am when this system fails. You care deeply about observability, clear error messages, runbooks, and anything that will help you debug production issues quickly.",

    "junior-developer": "You are a junior developer who will implement this spec. Flag anything that is ambiguous, assumes tribal knowledge, or would require you to make decisions that should be in the spec.",

    "qa-engineer": "You are a QA engineer responsible for testing this system. Identify missing test scenarios, edge cases, boundary conditions, and acceptance criteria. Flag anything untestable.",

    "site-reliability": "You are an SRE responsible for running this in production. Focus on operational concerns: deployment, rollback, monitoring, alerting, capacity planning, and incident response.",

    "product-manager": "You are a product manager reviewing this spec. Focus on user value, success metrics, scope clarity, and whether the spec actually solves the stated problem.",

    "data-engineer": "You are a data engineer. Focus on data models, data flow, ETL implications, analytics requirements, data quality, and downstream data consumer needs.",

    "mobile-developer": "You are a mobile developer. Focus on API design from a mobile perspective: payload sizes, offline support, battery impact, and mobile-specific UX concerns.",

    "accessibility-specialist": "You are an accessibility specialist. Focus on WCAG compliance, screen reader support, keyboard navigation, color contrast, and inclusive design patterns.",

    "legal-compliance": "You are a legal/compliance reviewer. Focus on data privacy (GDPR, CCPA), terms of service implications, liability, audit requirements, and regulatory compliance.",
}

SYSTEM_PROMPT_PRD = """You are a senior product manager participating in adversarial spec development.

You will receive a Product Requirements Document (PRD) from another AI model. Your job is to critique it rigorously.

Analyze the PRD for:
- Clear problem definition with evidence of real user pain
- Well-defined user personas with specific, believable characteristics
- User stories in proper format (As a... I want... So that...)
- Measurable success criteria and KPIs
- Explicit scope boundaries (what's in AND out)
- Realistic risk assessment with mitigations
- Dependencies identified
- NO technical implementation details (that belongs in a tech spec)

Expected PRD structure:
- Executive Summary
- Problem Statement / Opportunity
- Target Users / Personas
- User Stories / Use Cases
- Functional Requirements
- Non-Functional Requirements
- Success Metrics / KPIs
- Scope (In/Out)
- Dependencies
- Risks and Mitigations

If you find significant issues:
- Provide a clear critique explaining each problem
- Output your revised PRD that addresses these issues
- Format: First your critique, then the revised PRD between [SPEC] and [/SPEC] tags

If the PRD is solid and ready for stakeholder review:
- Output exactly [AGREE] on its own line
- Then output the final PRD between [SPEC] and [/SPEC] tags

Be rigorous. A good PRD should let any PM or designer understand exactly what to build and why.
Push back on vague requirements, unmeasurable success criteria, and missing user context."""

SYSTEM_PROMPT_TECH = """You are a senior software architect participating in adversarial spec development.

You will receive a Technical Specification from another AI model. Your job is to critique it rigorously.

Analyze the spec for:
- Clear architectural decisions with rationale
- Complete API contracts (endpoints, methods, request/response schemas, error codes)
- Data models that handle all identified use cases
- Security threats identified and mitigated (auth, authz, input validation, data protection)
- Error scenarios enumerated with handling strategy
- Performance targets that are specific and measurable
- Deployment strategy that is repeatable and reversible
- No ambiguity an engineer would need to resolve

Expected structure:
- Overview / Context
- Goals and Non-Goals
- System Architecture
- Component Design
- API Design (full schemas, not just endpoint names)
- Data Models / Database Schema
- Infrastructure Requirements
- Security Considerations
- Error Handling Strategy
- Performance Requirements / SLAs
- Observability (logging, metrics, alerting)
- Testing Strategy
- Deployment Strategy
- Migration Plan (if applicable)
- Open Questions / Future Considerations

If you find significant issues:
- Provide a clear critique explaining each problem
- Output your revised specification that addresses these issues
- Format: First your critique, then the revised spec between [SPEC] and [/SPEC] tags

If the spec is solid and production-ready:
- Output exactly [AGREE] on its own line
- Then output the final spec between [SPEC] and [/SPEC] tags

Be rigorous. A good tech spec should let any engineer implement the system without asking clarifying questions.
Push back on incomplete APIs, missing error handling, vague performance targets, and security gaps."""

SYSTEM_PROMPT_GENERIC = """You are a senior technical reviewer participating in adversarial spec development.

You will receive a specification from another AI model. Your job:

1. Analyze the spec rigorously for:
   - Gaps in requirements
   - Ambiguous language
   - Missing edge cases
   - Security vulnerabilities
   - Scalability concerns
   - Technical feasibility issues
   - Inconsistencies between sections
   - Missing error handling
   - Unclear data models or API designs

2. If you find significant issues:
   - Provide a clear critique explaining each problem
   - Output your revised specification that addresses these issues
   - Format: First your critique, then the revised spec between [SPEC] and [/SPEC] tags

3. If the spec is solid and production-ready with no material changes needed:
   - Output exactly [AGREE] on its own line
   - Then output the final spec between [SPEC] and [/SPEC] tags

Be rigorous and demanding. Do not agree unless the spec is genuinely complete and production-ready.
Push back on weak points. The goal is convergence on an excellent spec, not quick agreement."""

REVIEW_PROMPT_TEMPLATE = """This is round {round} of adversarial spec development.

Here is the current {doc_type_name}:

{spec}

{context_section}
{focus_section}
Review this document according to your criteria. Either critique and revise it, or say [AGREE] if it's production-ready."""

PRESS_PROMPT_TEMPLATE = """This is round {round} of adversarial spec development. You previously indicated agreement with this document.

Here is the current {doc_type_name}:

{spec}

{context_section}
**IMPORTANT: Please confirm your agreement by thoroughly reviewing the ENTIRE document.**

Before saying [AGREE], you MUST:
1. Confirm you have read every section of this document
2. List at least 3 specific sections you reviewed and what you verified in each
3. Explain WHY you agree - what makes this document complete and production-ready?
4. Identify ANY remaining concerns, however minor (even stylistic or optional improvements)

If after this thorough review you find issues you missed before, provide your critique.

If you genuinely agree after careful review, output:
1. Your verification (sections reviewed, reasons for agreement, minor concerns)
2. [AGREE] on its own line
3. The final spec between [SPEC] and [/SPEC] tags"""

EXPORT_TASKS_PROMPT = """Analyze this {doc_type_name} and extract all actionable tasks.

Document:
{spec}

For each task, output in this exact format:
[TASK]
title: <short task title>
type: <user-story | bug | task | spike>
priority: <high | medium | low>
description: <detailed description>
acceptance_criteria:
- <criterion 1>
- <criterion 2>
[/TASK]

Extract:
1. All user stories as individual tasks
2. Technical requirements as implementation tasks
3. Any identified risks as spike/investigation tasks
4. Non-functional requirements as tasks

Be thorough. Every actionable item in the spec should become a task."""


@dataclass
class ModelResponse:
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
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int):
        costs = MODEL_COSTS.get(model, DEFAULT_COST)
        cost = (input_tokens / 1_000_000 * costs["input"]) + (output_tokens / 1_000_000 * costs["output"])

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
        lines = ["", "=== Cost Summary ==="]
        lines.append(f"Total tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out")
        lines.append(f"Total cost: ${self.total_cost:.4f}")
        if len(self.by_model) > 1:
            lines.append("")
            lines.append("By model:")
            for model, data in self.by_model.items():
                lines.append(f"  {model}: ${data['cost']:.4f} ({data['input_tokens']:,} in / {data['output_tokens']:,} out)")
        return "\n".join(lines)


cost_tracker = CostTracker()


@dataclass
class SessionState:
    """Persisted state for resume functionality."""
    session_id: str
    spec: str
    round: int
    doc_type: str
    models: list
    focus: Optional[str] = None
    persona: Optional[str] = None
    preserve_intent: bool = False
    created_at: str = ""
    updated_at: str = ""
    history: list = field(default_factory=list)

    def save(self):
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        path = SESSIONS_DIR / f"{self.session_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, session_id: str) -> "SessionState":
        path = SESSIONS_DIR / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found")
        data = json.loads(path.read_text())
        return cls(**data)

    @classmethod
    def list_sessions(cls) -> list[dict]:
        if not SESSIONS_DIR.exists():
            return []
        sessions = []
        for p in SESSIONS_DIR.glob("*.json"):
            try:
                data = json.loads(p.read_text())
                sessions.append({
                    "id": data["session_id"],
                    "round": data["round"],
                    "doc_type": data["doc_type"],
                    "updated_at": data.get("updated_at", ""),
                })
            except Exception:
                pass
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)


def save_checkpoint(spec: str, round_num: int, session_id: Optional[str] = None):
    """Save spec checkpoint for this round."""
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{session_id}-" if session_id else ""
    path = CHECKPOINTS_DIR / f"{prefix}round-{round_num}.md"
    path.write_text(spec)
    print(f"Checkpoint saved: {path}", file=sys.stderr)


def get_system_prompt(doc_type: str, persona: Optional[str] = None) -> str:
    if persona:
        persona_key = persona.lower().replace(" ", "-").replace("_", "-")
        if persona_key in PERSONAS:
            return PERSONAS[persona_key]
        else:
            return f"You are a {persona} participating in adversarial spec development. Review the document from your professional perspective and critique any issues you find."

    if doc_type == "prd":
        return SYSTEM_PROMPT_PRD
    elif doc_type == "tech":
        return SYSTEM_PROMPT_TECH
    else:
        return SYSTEM_PROMPT_GENERIC


def get_doc_type_name(doc_type: str) -> str:
    if doc_type == "prd":
        return "Product Requirements Document"
    elif doc_type == "tech":
        return "Technical Specification"
    else:
        return "specification"


def load_context_files(context_paths: list[str]) -> str:
    if not context_paths:
        return ""

    sections = []
    for path in context_paths:
        try:
            content = Path(path).read_text()
            sections.append(f"### Context: {path}\n```\n{content}\n```")
        except Exception as e:
            sections.append(f"### Context: {path}\n[Error loading file: {e}]")

    return "## Additional Context\nThe following documents are provided as context:\n\n" + "\n\n".join(sections)


def load_profile(profile_name: str) -> dict:
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    if not profile_path.exists():
        print(f"Error: Profile '{profile_name}' not found at {profile_path}", file=sys.stderr)
        sys.exit(2)

    try:
        return json.loads(profile_path.read_text())
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in profile '{profile_name}': {e}", file=sys.stderr)
        sys.exit(2)


def save_profile(profile_name: str, config: dict):
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{profile_name}.json"
    profile_path.write_text(json.dumps(config, indent=2))
    print(f"Profile saved to {profile_path}")


def list_profiles():
    print("Saved Profiles:\n")
    if not PROFILES_DIR.exists():
        print("  No profiles found.")
        print(f"\n  Profiles are stored in: {PROFILES_DIR}")
        print("\n  Create a profile with: python3 debate.py save-profile <name> --models ... --focus ...")
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


def call_codex_model(
    system_prompt: str,
    user_message: str,
    model: str,
    reasoning_effort: str = DEFAULT_CODEX_REASONING,
    timeout: int = 300,
    search: bool = False
) -> tuple[str, int, int]:
    """
    Call Codex CLI in headless mode using ChatGPT subscription.

    Args:
        system_prompt: System instructions for the model
        user_message: User prompt to send
        model: Model name (e.g., "codex/gpt-5.2-codex" -> uses "gpt-5.2-codex")
        reasoning_effort: Thinking level (minimal, low, medium, high, xhigh). Default: xhigh
        timeout: Timeout in seconds (default 5 minutes)
        search: Enable web search capability for Codex

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)

    Raises:
        RuntimeError: If Codex CLI is not available or fails
    """
    if not CODEX_AVAILABLE:
        raise RuntimeError("Codex CLI not found. Install with: npm install -g @openai/codex")

    # Extract actual model name from "codex/model" format
    actual_model = model.split("/", 1)[1] if "/" in model else model

    # Combine system prompt and user message for Codex
    # Codex exec doesn't have separate system/user roles, so we combine them
    full_prompt = f"""SYSTEM INSTRUCTIONS:
{system_prompt}

USER REQUEST:
{user_message}"""

    try:
        # Run codex exec with JSON output and reasoning effort
        cmd = [
            "codex", "exec",
            "--json",
            "--full-auto",
            "--model", actual_model,
            "-c", f'model_reasoning_effort="{reasoning_effort}"',
        ]
        if search:
            cmd.append("--search")
        cmd.append(full_prompt)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or f"Codex exited with code {result.returncode}"
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

                # Extract agent message content
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        response_text = item.get("text", "")

                # Extract token usage from turn.completed
                if event.get("type") == "turn.completed":
                    usage = event.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

            except json.JSONDecodeError:
                continue  # Skip malformed lines

        if not response_text:
            raise RuntimeError("No agent message found in Codex output")

        return response_text, input_tokens, output_tokens

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Codex CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Codex CLI not found in PATH")


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
    codex_search: bool = False
) -> ModelResponse:
    """Send spec to a single model and return response with retry on failure."""
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
        context_section=context_section
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
                    search=codex_search
                )
                agreed = "[AGREE]" in content
                extracted = extract_spec(content)

                if not agreed and not extracted:
                    print(f"Warning: {model} provided critique but no [SPEC] tags found. Response may be malformed.", file=sys.stderr)

                cost = cost_tracker.add(model, input_tokens, output_tokens)

                return ModelResponse(
                    model=model,
                    response=content,
                    agreed=agreed,
                    spec=extracted,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=cost
                )
            except Exception as e:
                last_error = str(e)
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"Warning: {model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...", file=sys.stderr)
                    time.sleep(delay)
                else:
                    print(f"Error: {model} failed after {MAX_RETRIES} attempts: {last_error}", file=sys.stderr)

        return ModelResponse(model=model, response="", agreed=False, spec=None, error=last_error)

    # Standard litellm path for all other providers
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=8000
            )
            content = response.choices[0].message.content
            agreed = "[AGREE]" in content
            extracted = extract_spec(content)

            # Validation warning if model critiqued but didn't provide revised spec
            if not agreed and not extracted:
                print(f"Warning: {model} provided critique but no [SPEC] tags found. Response may be malformed.", file=sys.stderr)

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            cost = cost_tracker.add(model, input_tokens, output_tokens)

            return ModelResponse(
                model=model,
                response=content,
                agreed=agreed,
                spec=extracted,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            )
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)  # exponential backoff
                print(f"Warning: {model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...", file=sys.stderr)
                time.sleep(delay)
            else:
                print(f"Error: {model} failed after {MAX_RETRIES} attempts: {last_error}", file=sys.stderr)

    return ModelResponse(model=model, response="", agreed=False, spec=None, error=last_error)


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
    codex_search: bool = False
) -> list[ModelResponse]:
    """Call multiple models in parallel and collect responses."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {
            executor.submit(
                call_single_model, model, spec, round_num, doc_type, press, focus, persona, context, preserve_intent, codex_reasoning, codex_search
            ): model
            for model in models
        }
        for future in concurrent.futures.as_completed(future_to_model):
            results.append(future.result())
    return results


def detect_agreement(response: str) -> bool:
    return "[AGREE]" in response


def extract_spec(response: str) -> Optional[str]:
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
        task = {}
        current_key = None
        current_value = []

        for line in task_text.split("\n"):
            line = line.strip()
            if line.startswith("title:"):
                if current_key:
                    task[current_key] = "\n".join(current_value).strip() if len(current_value) > 1 else current_value[0] if current_value else ""
                current_key = "title"
                current_value = [line[6:].strip()]
            elif line.startswith("type:"):
                if current_key:
                    task[current_key] = "\n".join(current_value).strip() if len(current_value) > 1 else current_value[0] if current_value else ""
                current_key = "type"
                current_value = [line[5:].strip()]
            elif line.startswith("priority:"):
                if current_key:
                    task[current_key] = "\n".join(current_value).strip() if len(current_value) > 1 else current_value[0] if current_value else ""
                current_key = "priority"
                current_value = [line[9:].strip()]
            elif line.startswith("description:"):
                if current_key:
                    task[current_key] = "\n".join(current_value).strip() if len(current_value) > 1 else current_value[0] if current_value else ""
                current_key = "description"
                current_value = [line[12:].strip()]
            elif line.startswith("acceptance_criteria:"):
                if current_key:
                    task[current_key] = "\n".join(current_value).strip() if len(current_value) > 1 else current_value[0] if current_value else ""
                current_key = "acceptance_criteria"
                current_value = []
            elif line.startswith("- ") and current_key == "acceptance_criteria":
                current_value.append(line[2:])
            elif current_key:
                current_value.append(line)

        if current_key:
            task[current_key] = current_value if current_key == "acceptance_criteria" else "\n".join(current_value).strip()

        if task.get("title"):
            tasks.append(task)

    return tasks


def get_critique_summary(response: str, max_length: int = 300) -> str:
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
        prev_lines,
        curr_lines,
        fromfile="previous",
        tofile="current",
        lineterm=""
    )
    return "".join(diff)


def list_providers():
    providers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-4o, gpt-4-turbo, o1"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-20250514, claude-opus-4-20250514"),
        ("Google", "GEMINI_API_KEY", "gemini/gemini-2.0-flash, gemini/gemini-pro"),
        ("xAI", "XAI_API_KEY", "xai/grok-3, xai/grok-beta"),
        ("Mistral", "MISTRAL_API_KEY", "mistral/mistral-large, mistral/codestral"),
        ("Groq", "GROQ_API_KEY", "groq/llama-3.3-70b-versatile"),
        ("Together", "TOGETHER_API_KEY", "together_ai/meta-llama/Llama-3-70b"),
        ("Deepseek", "DEEPSEEK_API_KEY", "deepseek/deepseek-chat"),
    ]
    print("Supported providers:\n")
    for name, key, models in providers:
        status = "[set]" if os.environ.get(key) else "[not set]"
        print(f"  {name:12} {key:24} {status}")
        print(f"             Example models: {models}")
        print()

    # Codex CLI (uses ChatGPT subscription, not API key)
    codex_status = "[installed]" if CODEX_AVAILABLE else "[not installed]"
    print(f"  {'Codex CLI':12} {'(ChatGPT subscription)':24} {codex_status}")
    print(f"             Example models: codex/gpt-5.2-codex, codex/gpt-5.1-codex-max")
    print(f"             Reasoning: --codex-reasoning (minimal, low, medium, high, xhigh)")
    print(f"             Install: npm install -g @openai/codex && codex login")
    print()


def list_focus_areas():
    print("Available focus areas (--focus):\n")
    for name, description in FOCUS_AREAS.items():
        first_line = description.strip().split("\n")[1] if "\n" in description else description[:60]
        print(f"  {name:15} {first_line.strip()[:60]}")
    print()


def list_personas():
    print("Available personas (--persona):\n")
    for name, description in PERSONAS.items():
        print(f"  {name}")
        print(f"    {description[:80]}...")
        print()


def send_telegram_notification(models: list[str], round_num: int, results: list[ModelResponse], poll_timeout: int) -> Optional[str]:
    """Send Telegram notification with all model responses and poll for feedback."""
    try:
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        import telegram_bot

        token, chat_id = telegram_bot.get_config()
        if not token or not chat_id:
            print("Warning: Telegram not configured. Skipping notification.", file=sys.stderr)
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

        full_notification = notification + f"\n\n_Reply within {poll_timeout}s to add feedback, or wait to continue._"
        if not telegram_bot.send_long_message(token, chat_id, full_notification):
            print("Warning: Failed to send Telegram notification.", file=sys.stderr)
            return None

        feedback = telegram_bot.poll_for_reply(token, chat_id, poll_timeout, last_update)
        return feedback

    except ImportError:
        print("Warning: telegram_bot.py not found. Skipping notification.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Telegram error: {e}", file=sys.stderr)
        return None


def send_final_spec_to_telegram(spec: str, rounds: int, models: list[str], doc_type: str) -> bool:
    """Send the final converged spec to Telegram."""
    try:
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        import telegram_bot

        token, chat_id = telegram_bot.get_config()
        if not token or not chat_id:
            print("Warning: Telegram not configured. Skipping final spec notification.", file=sys.stderr)
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


def main():
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

Document types:
  prd   - Product Requirements Document (business/product focus)
  tech  - Technical Specification / Architecture Document (engineering focus)
        """
    )
    parser.add_argument("action", choices=["critique", "providers", "send-final", "diff", "export-tasks", "focus-areas", "personas", "profiles", "save-profile", "sessions"],
                        help="Action to perform")
    parser.add_argument("profile_name", nargs="?", help="Profile name (for save-profile action)")
    parser.add_argument("--models", "-m", default="gpt-4o",
                        help="Comma-separated list of models (e.g., gpt-4o,gemini/gemini-2.0-flash,xai/grok-3)")
    parser.add_argument("--doc-type", "-d", choices=["prd", "tech"], default="tech",
                        help="Document type: prd or tech (default: tech)")
    parser.add_argument("--round", "-r", type=int, default=1,
                        help="Current round number")
    parser.add_argument("--json", "-j", action="store_true",
                        help="Output as JSON")
    parser.add_argument("--telegram", "-t", action="store_true",
                        help="Send Telegram notifications and poll for feedback")
    parser.add_argument("--poll-timeout", type=int, default=60,
                        help="Seconds to wait for Telegram reply (default: 60)")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Total rounds completed (used with send-final)")
    parser.add_argument("--press", "-p", action="store_true",
                        help="Press models to confirm they read the full document (anti-laziness check)")
    parser.add_argument("--focus", "-f",
                        help="Focus area for critique (security, scalability, performance, ux, reliability, cost)")
    parser.add_argument("--persona",
                        help="Persona for critique (security-engineer, oncall-engineer, junior-developer, etc.)")
    parser.add_argument("--context", "-c", action="append", default=[],
                        help="Additional context file(s) to include (can be used multiple times)")
    parser.add_argument("--profile",
                        help="Load settings from a saved profile")
    parser.add_argument("--previous",
                        help="Previous spec file (for diff action)")
    parser.add_argument("--current",
                        help="Current spec file (for diff action)")
    parser.add_argument("--show-cost", action="store_true",
                        help="Show cost summary after critique")
    parser.add_argument("--preserve-intent", action="store_true",
                        help="Require explicit justification for any removal or substantial modification")
    parser.add_argument("--codex-reasoning", default=DEFAULT_CODEX_REASONING,
                        choices=["low", "medium", "high", "xhigh"],
                        help=f"Reasoning effort for Codex CLI models (default: {DEFAULT_CODEX_REASONING})")
    parser.add_argument("--session", "-s",
                        help="Session ID for state persistence (enables checkpointing and resume)")
    parser.add_argument("--resume",
                        help="Resume a previous session by ID")
    parser.add_argument("--codex-search", action="store_true",
                        help="Enable web search for Codex CLI models")
    args = parser.parse_args()

    # Handle simple info commands
    if args.action == "providers":
        list_providers()
        return

    if args.action == "focus-areas":
        list_focus_areas()
        return

    if args.action == "personas":
        list_personas()
        return

    if args.action == "profiles":
        list_profiles()
        return

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
                print(f"    updated: {s['updated_at'][:19] if s['updated_at'] else 'unknown'}")
                print()
        return

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
        return

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
        except Exception as e:
            print(f"Error reading files: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Load profile if specified
    if args.profile:
        profile = load_profile(args.profile)
        if "models" in profile and args.models == "gpt-4o":
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

    # Parse models list
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("Error: No models specified", file=sys.stderr)
        sys.exit(1)

    # Load context files
    context = load_context_files(args.context) if args.context else None

    if args.action == "send-final":
        spec = sys.stdin.read().strip()
        if not spec:
            print("Error: No spec provided via stdin", file=sys.stderr)
            sys.exit(1)
        if send_final_spec_to_telegram(spec, args.rounds, models, args.doc_type):
            print("Final document sent to Telegram.")
        else:
            print("Failed to send final document to Telegram.", file=sys.stderr)
            sys.exit(1)
        return

    if args.action == "export-tasks":
        spec = sys.stdin.read().strip()
        if not spec:
            print("Error: No spec provided via stdin", file=sys.stderr)
            sys.exit(1)

        doc_type_name = get_doc_type_name(args.doc_type)
        prompt = EXPORT_TASKS_PROMPT.format(doc_type_name=doc_type_name, spec=spec)

        try:
            response = completion(
                model=models[0],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8000
            )
            content = response.choices[0].message.content
            tasks = extract_tasks(content)

            if args.json:
                print(json.dumps({"tasks": tasks}, indent=2))
            else:
                print(f"\n=== Extracted {len(tasks)} Tasks ===\n")
                for i, task in enumerate(tasks, 1):
                    print(f"{i}. [{task.get('type', 'task')}] [{task.get('priority', 'medium')}] {task.get('title', 'Untitled')}")
                    if task.get('description'):
                        print(f"   {task['description'][:100]}...")
                    if task.get('acceptance_criteria'):
                        print(f"   Acceptance criteria: {len(task['acceptance_criteria'])} items")
                    print()
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # Handle resume
    session_state = None
    if args.resume:
        try:
            session_state = SessionState.load(args.resume)
            print(f"Resuming session '{args.resume}' at round {session_state.round}", file=sys.stderr)
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
            # Re-parse models
            models = session_state.models
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        # Main critique action
        spec = sys.stdin.read().strip()
        if not spec:
            print("Error: No spec provided via stdin", file=sys.stderr)
            sys.exit(1)

    # Initialize session if --session provided
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

    mode = "pressing for confirmation" if args.press else "critiquing"
    focus_info = f" (focus: {args.focus})" if args.focus else ""
    persona_info = f" (persona: {args.persona})" if args.persona else ""
    preserve_info = " (preserve-intent)" if args.preserve_intent else ""
    search_info = " (search)" if args.codex_search else ""
    print(f"Calling {len(models)} model(s) ({mode}){focus_info}{persona_info}{preserve_info}{search_info}: {', '.join(models)}...", file=sys.stderr)

    results = call_models_parallel(
        models, spec, args.round, args.doc_type, args.press,
        args.focus, args.persona, context, args.preserve_intent,
        args.codex_reasoning, args.codex_search
    )

    errors = [r for r in results if r.error]
    for e in errors:
        print(f"Warning: {e.model} returned error: {e.error}", file=sys.stderr)

    successful = [r for r in results if not r.error]
    all_agreed = all(r.agreed for r in successful) if successful else False

    # Save checkpoint after each round
    session_id = session_state.session_id if session_state else args.session
    if session_id or args.session:
        save_checkpoint(spec, args.round, session_id)

    # Get the latest spec from results (first non-agreed response with a spec)
    latest_spec = spec
    for r in successful:
        if r.spec:
            latest_spec = r.spec
            break

    # Update session state
    if session_state:
        session_state.spec = latest_spec
        session_state.round = args.round + 1
        session_state.history.append({
            "round": args.round,
            "all_agreed": all_agreed,
            "models": [{"model": r.model, "agreed": r.agreed, "error": r.error} for r in results],
        })
        session_state.save()

    user_feedback = None
    if args.telegram:
        user_feedback = send_telegram_notification(models, args.round, results, args.poll_timeout)
        if user_feedback:
            print(f"Received feedback: {user_feedback}", file=sys.stderr)

    if args.json:
        output = {
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
                    "cost": r.cost
                }
                for r in results
            ],
            "cost": {
                "total": cost_tracker.total_cost,
                "input_tokens": cost_tracker.total_input_tokens,
                "output_tokens": cost_tracker.total_output_tokens,
                "by_model": cost_tracker.by_model
            }
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
            agreed_models = [r.model for r in successful if r.agreed]
            disagreed_models = [r.model for r in successful if not r.agreed]
            if agreed_models:
                print(f"Agreed: {', '.join(agreed_models)}")
            if disagreed_models:
                print(f"Critiqued: {', '.join(disagreed_models)}")

        if user_feedback:
            print()
            print(f"=== User Feedback ===")
            print(user_feedback)

        if args.show_cost or True:  # Always show cost
            print(cost_tracker.summary())


if __name__ == "__main__":
    main()
