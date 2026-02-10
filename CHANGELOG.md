# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- New `codex/gpt-5.3-codex` model support for Codex CLI (ChatGPT subscription)
- Updated default Codex CLI model from `gpt-5.2-codex` to `gpt-5.3-codex`

### Fixed

- Added `--skip-git-repo-check` flag to Codex CLI calls for non-git directory support
- Script paths now use dynamic lookup to work with both manual and marketplace installations
- Replaced hardcoded `gpt-4o` default model with dynamic detection based on available API keys
- Added pre-flight validation to check that models have required API keys before running critique
- Added clear error messages when API keys are missing, showing which key is needed for each model
- Fixed model selection to prioritize available providers in order: Bedrock, OpenAI, Anthropic, Google, xAI, etc.
- Added documentation for resolving Claude Code auth conflicts (claude.ai token vs ANTHROPIC_API_KEY)
- Updated documentation to include Anthropic provider in supported models table

### Changed

- `--models` argument now auto-detects default model from available API keys instead of assuming `gpt-4o`
- Script now fails fast with helpful error messages if no API keys are configured
- Profile loading now correctly handles the new dynamic default model selection

### Added

- New `get_available_providers()` function to detect configured API keys
- New `get_default_model()` function to select appropriate default based on available keys
- New `validate_model_credentials()` function to pre-validate model API key requirements
- Comprehensive test coverage for new validation functions (11 new tests, 297 total)

## [1.0.0] - 2025-01-11

### Added

- Multi-model adversarial debate for spec refinement
- Support for multiple LLM providers via LiteLLM:
  - OpenAI (gpt-4o, gpt-4-turbo, o1)
  - Google (gemini/gemini-2.0-flash, gemini/gemini-pro)
  - xAI (xai/grok-3, xai/grok-beta)
  - Mistral (mistral/mistral-large, mistral/codestral)
  - Groq (groq/llama-3.3-70b-versatile)
  - Deepseek (deepseek/deepseek-chat)
  - Zhipu (zhipu/glm-4, zhipu/glm-4-plus)
- Codex CLI integration for ChatGPT subscription models
- AWS Bedrock integration for enterprise users
- OpenAI-compatible endpoint support for local/self-hosted models
- Document types: PRD (product) and Tech Spec (engineering)
- Interview mode for in-depth requirements gathering
- Claude's active participation in debates alongside opponent models
- Early agreement verification to prevent rubber-stamping
- User review period with change request workflow
- PRD to Tech Spec flow for complete documentation
- Critique focus modes: security, scalability, performance, ux, reliability, cost
- Professional personas: security-engineer, oncall-engineer, junior-developer, qa-engineer, etc.
- Context injection for existing documents
- Session persistence and resume functionality
- Auto-checkpointing for rollback capability
- Preserve intent mode requiring justification for removals
- Cost tracking with per-model breakdown
- Saved profiles for frequently used configurations
- Diff between spec versions
- Export to task list (with JSON output option)
- Telegram integration for async notifications and human-in-the-loop feedback
- Retry with exponential backoff for API resilience
- Response validation warnings for malformed outputs

### Technical

- Modular codebase: debate.py, models.py, providers.py, session.py, prompts.py, telegram_bot.py
- Full type hints with py.typed marker
- Google-style docstrings
- Input validation for security-sensitive operations
- Structured logging for exception handling
- Unit tests with pytest (194 tests, 91% coverage)
- CI workflow with linting (ruff), type checking (mypy), tests with coverage threshold
- Pre-commit hooks for code quality
