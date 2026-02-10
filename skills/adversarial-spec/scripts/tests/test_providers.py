"""Tests for providers module."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers import (
    BEDROCK_MODEL_MAP,
    DEFAULT_COST,
    MODEL_COSTS,
    is_bedrock_enabled,
    load_global_config,
    load_profile,
    resolve_bedrock_model,
    save_global_config,
    save_profile,
    validate_bedrock_models,
)


class TestModelCosts:
    def test_model_costs_has_expected_models(self):
        expected = [
            "gpt-4o",
            "gemini/gemini-2.0-flash",
            "xai/grok-3",
            "mistral/mistral-large",
            "deepseek/deepseek-chat",
            "zhipu/glm-4",
            "codex/gpt-5.3-codex",
        ]
        for model in expected:
            assert model in MODEL_COSTS

    def test_costs_have_input_and_output(self):
        for model, costs in MODEL_COSTS.items():
            assert "input" in costs
            assert "output" in costs
            assert isinstance(costs["input"], (int, float))
            assert isinstance(costs["output"], (int, float))

    def test_default_cost_exists(self):
        assert "input" in DEFAULT_COST
        assert "output" in DEFAULT_COST


class TestBedrockModelMap:
    def test_has_claude_models(self):
        assert "claude-3-sonnet" in BEDROCK_MODEL_MAP
        assert "claude-3-haiku" in BEDROCK_MODEL_MAP
        assert "claude-3-opus" in BEDROCK_MODEL_MAP

    def test_has_llama_models(self):
        assert "llama-3-8b" in BEDROCK_MODEL_MAP
        assert "llama-3-70b" in BEDROCK_MODEL_MAP

    def test_maps_to_full_bedrock_ids(self):
        for name, bedrock_id in BEDROCK_MODEL_MAP.items():
            assert "." in bedrock_id or ":" in bedrock_id


class TestGlobalConfig:
    def test_load_nonexistent_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                config = load_global_config()
                assert config == {}

    def test_save_and_load_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                save_global_config(
                    {"bedrock": {"enabled": True, "region": "us-east-1"}}
                )

                assert config_path.exists()

                loaded = load_global_config()
                assert loaded["bedrock"]["enabled"] is True
                assert loaded["bedrock"]["region"] == "us-east-1"


class TestBedrockEnabled:
    def test_returns_false_when_not_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                assert is_bedrock_enabled() is False

    def test_returns_true_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"bedrock": {"enabled": True}}))

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                assert is_bedrock_enabled() is True


class TestResolveBrockModel:
    def test_resolves_friendly_name(self):
        result = resolve_bedrock_model("claude-3-sonnet")
        assert result == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_returns_full_id_as_is(self):
        full_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        result = resolve_bedrock_model(full_id)
        assert result == full_id

    def test_returns_none_for_unknown(self):
        result = resolve_bedrock_model("unknown-model")
        assert result is None

    def test_uses_custom_aliases(self):
        config = {"custom_aliases": {"my-model": "custom.model-id"}}
        result = resolve_bedrock_model("my-model", config)
        assert result == "custom.model-id"


class TestValidateBedrockModels:
    def test_validates_available_models(self):
        config = {
            "available_models": ["claude-3-sonnet", "claude-3-haiku"],
        }
        valid, invalid = validate_bedrock_models(["claude-3-sonnet"], config)
        assert len(valid) == 1
        assert len(invalid) == 0

    def test_rejects_unavailable_models(self):
        config = {
            "available_models": ["claude-3-sonnet"],
        }
        valid, invalid = validate_bedrock_models(["claude-3-opus"], config)
        assert len(valid) == 0
        assert len(invalid) == 1
        assert "claude-3-opus" in invalid


class TestProfiles:
    def test_save_and_load_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir) / "profiles"

            with patch("providers.PROFILES_DIR", profiles_dir):
                config = {
                    "models": "gpt-4o,gemini/gemini-2.0-flash",
                    "focus": "security",
                    "persona": "security-engineer",
                }
                save_profile("test-profile", config)

                assert (profiles_dir / "test-profile.json").exists()

                loaded = load_profile("test-profile")
                assert loaded["models"] == "gpt-4o,gemini/gemini-2.0-flash"
                assert loaded["focus"] == "security"


class TestLoadGlobalConfigInvalidJson:
    """Tests for load_global_config with invalid JSON.

    Mutation target: exception handling for json.JSONDecodeError
    """

    def test_returns_empty_dict_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{ invalid json }")

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                config = load_global_config()
                # Mutation: removing return {} would cause crash or wrong value
                assert config == {}


class TestGetBedrockConfig:
    """Tests for get_bedrock_config function.

    Mutation target: .get("bedrock", {}) default value
    """

    def test_returns_empty_dict_when_no_bedrock_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text("{}")

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                from providers import get_bedrock_config

                config = get_bedrock_config()
                # Mutation: changing default {} to None would break code
                assert config == {}

    def test_returns_bedrock_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"bedrock": {"enabled": True}}))

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                from providers import get_bedrock_config

                config = get_bedrock_config()
                assert config["enabled"] is True


class TestResolveBrockModelBoundaries:
    """Additional tests for resolve_bedrock_model edge cases.

    Mutation targets:
    - "." in friendly_name check
    - .startswith("bedrock/") check
    - config is None check
    """

    def test_bedrock_prefix_not_returned_as_is(self):
        # Mutation: removing .startswith("bedrock/") check would return wrong value
        result = resolve_bedrock_model("bedrock/claude-3-sonnet")
        # Should NOT be returned as-is since it starts with bedrock/
        assert result is None or "bedrock/" not in result

    def test_dot_in_name_with_bedrock_prefix_goes_through_lookup(self):
        # Mutation: wrong AND logic would skip lookup
        result = resolve_bedrock_model("bedrock/anthropic.claude")
        assert result is None  # Not in map and not returned as-is

    def test_config_none_loads_from_global(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"bedrock": {"custom_aliases": {"mymodel": "custom.id"}}})
            )

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                # Mutation: not loading from global when config is None
                result = resolve_bedrock_model("mymodel", None)
                assert result == "custom.id"


class TestValidateBedrockModelsBoundaries:
    """Additional tests for validate_bedrock_models edge cases.

    Mutation targets:
    - model in available check
    - resolved model matching
    - for/else construct
    """

    def test_model_resolved_but_not_in_available_is_invalid(self):
        config = {
            "available_models": ["llama-3-8b"],  # Only llama in available
        }
        # claude-3-sonnet resolves but is not in available
        valid, invalid = validate_bedrock_models(["claude-3-sonnet"], config)
        # Mutation: wrong logic would mark as valid
        assert len(invalid) == 1
        assert "claude-3-sonnet" in invalid

    def test_resolved_model_matches_available_resolved(self):
        config = {
            "available_models": ["claude-3-sonnet"],  # Uses friendly name
        }
        # Pass the full bedrock ID - should match available model
        valid, invalid = validate_bedrock_models(
            ["anthropic.claude-3-sonnet-20240229-v1:0"], config
        )
        # Mutation: for/else would fail to add to valid
        assert len(valid) == 1
        assert len(invalid) == 0

    def test_config_none_loads_from_global(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps({"bedrock": {"available_models": ["claude-3-sonnet"]}})
            )

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                valid, invalid = validate_bedrock_models(["claude-3-sonnet"], None)
                assert len(valid) == 1


class TestLoadProfileErrors:
    """Tests for load_profile error handling.

    Mutation targets:
    - sys.exit(2) calls
    - profile_path.exists() check
    """

    def test_exits_when_profile_not_found(self):
        import pytest

        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir) / "profiles"

            with patch("providers.PROFILES_DIR", profiles_dir):
                with pytest.raises(SystemExit) as exc_info:
                    load_profile("nonexistent")
                # Mutation: changing exit code would fail
                assert exc_info.value.code == 2

    def test_exits_on_invalid_json(self):
        import pytest

        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir) / "profiles"
            profiles_dir.mkdir(parents=True)
            (profiles_dir / "bad.json").write_text("{ invalid }")

            with patch("providers.PROFILES_DIR", profiles_dir):
                with pytest.raises(SystemExit) as exc_info:
                    load_profile("bad")
                # Mutation: changing exit code would fail
                assert exc_info.value.code == 2


class TestListProfilesBranches:
    """Tests for list_profiles edge cases.

    Mutation targets:
    - PROFILES_DIR.exists() check
    - empty profiles list check
    - preserve_intent ternary
    """

    def test_no_profiles_dir(self):
        from io import StringIO

        from providers import list_profiles

        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir) / "nonexistent"

            with patch("providers.PROFILES_DIR", profiles_dir):
                with patch("sys.stdout", new_callable=StringIO) as mock_out:
                    list_profiles()
                    output = mock_out.getvalue()
                    # Mutation: not checking exists() would crash
                    assert "No profiles found" in output
                    assert str(profiles_dir) in output

    def test_empty_profiles_dir(self):
        from io import StringIO

        from providers import list_profiles

        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir) / "profiles"
            profiles_dir.mkdir()

            with patch("providers.PROFILES_DIR", profiles_dir):
                with patch("sys.stdout", new_callable=StringIO) as mock_out:
                    list_profiles()
                    output = mock_out.getvalue()
                    # Mutation: not checking empty list would skip message
                    assert "No profiles found" in output

    def test_preserve_intent_true(self):
        from io import StringIO

        from providers import list_profiles

        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir) / "profiles"
            profiles_dir.mkdir()
            (profiles_dir / "test.json").write_text(
                json.dumps({"models": "gpt-4o", "preserve_intent": True})
            )

            with patch("providers.PROFILES_DIR", profiles_dir):
                with patch("sys.stdout", new_callable=StringIO) as mock_out:
                    list_profiles()
                    output = mock_out.getvalue()
                    # Mutation: wrong ternary would show "no"
                    assert "preserve-intent: yes" in output

    def test_preserve_intent_false(self):
        from io import StringIO

        from providers import list_profiles

        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir) / "profiles"
            profiles_dir.mkdir()
            (profiles_dir / "test.json").write_text(
                json.dumps({"models": "gpt-4o", "preserve_intent": False})
            )

            with patch("providers.PROFILES_DIR", profiles_dir):
                with patch("sys.stdout", new_callable=StringIO) as mock_out:
                    list_profiles()
                    output = mock_out.getvalue()
                    # Mutation: wrong ternary would show "yes"
                    assert "preserve-intent: no" in output


class TestListProviders:
    """Tests for list_providers function.

    Mutation targets:
    - bedrock_config.get("enabled") check
    - os.environ.get() checks
    - CODEX_AVAILABLE check
    """

    def test_shows_bedrock_when_enabled(self):
        from io import StringIO

        from providers import list_providers

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "bedrock": {
                            "enabled": True,
                            "region": "us-east-1",
                            "available_models": ["claude-3-sonnet"],
                        }
                    }
                )
            )

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with patch("sys.stdout", new_callable=StringIO) as mock_out:
                    list_providers()
                    output = mock_out.getvalue()
                    # Mutation: not checking enabled would skip section
                    assert "AWS Bedrock (Active)" in output
                    assert "us-east-1" in output
                    assert "claude-3-sonnet" in output

    def test_shows_api_key_status(self):
        from io import StringIO

        from providers import list_providers

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with patch.dict(
                    "os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False
                ):
                    with patch("sys.stdout", new_callable=StringIO) as mock_out:
                        list_providers()
                        output = mock_out.getvalue()
                        # Mutation: wrong check would show wrong status
                        assert "OpenAI" in output
                        assert "[set]" in output

    def test_shows_codex_status(self):
        from io import StringIO

        from providers import list_providers

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with patch("providers.CODEX_AVAILABLE", True):
                    with patch("sys.stdout", new_callable=StringIO) as mock_out:
                        list_providers()
                        output = mock_out.getvalue()
                        # Mutation: wrong check would show wrong status
                        assert "Codex CLI" in output
                        assert "[installed]" in output


class TestListFocusAreas:
    """Tests for list_focus_areas function.

    Mutation target: newline split logic
    """

    def test_lists_all_focus_areas(self):
        from io import StringIO

        from providers import list_focus_areas

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            list_focus_areas()
            output = mock_out.getvalue()
            # Mutation: wrong split would show garbled text
            assert "security" in output
            assert "scalability" in output
            assert "performance" in output


class TestListPersonas:
    """Tests for list_personas function."""

    def test_lists_all_personas(self):
        from io import StringIO

        from providers import list_personas

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            list_personas()
            output = mock_out.getvalue()
            assert "security-engineer" in output
            assert "oncall-engineer" in output


class TestHandleBedrockCommand:
    """Tests for handle_bedrock_command function.

    Mutation targets:
    - subcommand dispatching
    - status output logic
    - enable/disable logic
    - add-model/remove-model logic
    """

    def test_status_not_configured(self):
        from io import StringIO

        from providers import handle_bedrock_command

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with patch("sys.stdout", new_callable=StringIO) as mock_out:
                    handle_bedrock_command("status", None, None)
                    output = mock_out.getvalue()
                    # Mutation: wrong check would show wrong status
                    assert "Not configured" in output

    def test_status_with_models(self):
        from io import StringIO

        from providers import handle_bedrock_command

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(
                    {
                        "bedrock": {
                            "enabled": True,
                            "region": "us-west-2",
                            "available_models": ["claude-3-sonnet"],
                        }
                    }
                )
            )

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with patch("sys.stdout", new_callable=StringIO) as mock_out:
                    handle_bedrock_command("status", None, None)
                    output = mock_out.getvalue()
                    assert "Enabled" in output
                    assert "us-west-2" in output
                    assert "claude-3-sonnet" in output

    def test_enable_requires_region(self):
        import pytest
        from providers import handle_bedrock_command

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with pytest.raises(SystemExit) as exc_info:
                    handle_bedrock_command("enable", None, None)
                # Mutation: wrong exit code
                assert exc_info.value.code == 1

    def test_disable_command(self):
        from providers import handle_bedrock_command, load_global_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"bedrock": {"enabled": True}}))

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                handle_bedrock_command("disable", None, None)
                config = load_global_config()
                # Mutation: not setting enabled = False
                assert config["bedrock"]["enabled"] is False

    def test_add_model_requires_arg(self):
        import pytest
        from providers import handle_bedrock_command

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with pytest.raises(SystemExit) as exc_info:
                    handle_bedrock_command("add-model", None, None)
                assert exc_info.value.code == 1

    def test_add_model_already_exists(self):
        from io import StringIO

        from providers import handle_bedrock_command

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps({"bedrock": {"available_models": ["claude-3-sonnet"]}})
            )

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with patch("sys.stdout", new_callable=StringIO) as mock_out:
                    handle_bedrock_command("add-model", "claude-3-sonnet", None)
                    output = mock_out.getvalue()
                    # Mutation: not checking existence would add duplicate
                    assert "already in the available list" in output

    def test_remove_model_requires_arg(self):
        import pytest
        from providers import handle_bedrock_command

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with pytest.raises(SystemExit) as exc_info:
                    handle_bedrock_command("remove-model", None, None)
                assert exc_info.value.code == 1

    def test_remove_model_not_in_list(self):
        import pytest
        from providers import handle_bedrock_command

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps({"bedrock": {"available_models": []}}))

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with pytest.raises(SystemExit) as exc_info:
                    handle_bedrock_command("remove-model", "nonexistent", None)
                assert exc_info.value.code == 1

    def test_alias_requires_arg(self):
        import pytest
        from providers import handle_bedrock_command

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with pytest.raises(SystemExit) as exc_info:
                    handle_bedrock_command("alias", None, None)
                assert exc_info.value.code == 1

    def test_alias_requires_two_args(self):
        import pytest
        from providers import handle_bedrock_command

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                with pytest.raises(SystemExit) as exc_info:
                    handle_bedrock_command("alias", "mymodel", None)
                assert exc_info.value.code == 1

    def test_list_models_command(self):
        from io import StringIO

        from providers import handle_bedrock_command

        with patch("sys.stdout", new_callable=StringIO) as mock_out:
            handle_bedrock_command("list-models", None, None)
            output = mock_out.getvalue()
            assert "claude-3-sonnet" in output
            assert "llama-3-8b" in output

    def test_unknown_subcommand(self):
        import pytest
        from providers import handle_bedrock_command

        with pytest.raises(SystemExit) as exc_info:
            handle_bedrock_command("unknown", None, None)
        assert exc_info.value.code == 1


class TestGetAvailableProviders:
    def test_returns_providers_with_keys_set(self):
        from providers import get_available_providers

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "ANTHROPIC_API_KEY": "test-key",
            },
            clear=False,
        ):
            available = get_available_providers()
            provider_names = [name for name, _, _ in available]
            assert "OpenAI" in provider_names
            assert "Anthropic" in provider_names

    def test_excludes_providers_without_keys(self):
        from providers import get_available_providers

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
            },
            clear=True,
        ):
            available = get_available_providers()
            provider_names = [name for name, _, _ in available]
            assert "OpenAI" in provider_names
            assert "Anthropic" not in provider_names

    def test_returns_default_models(self):
        from providers import get_available_providers

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            available = get_available_providers()
            for name, key, model in available:
                if name == "OpenAI":
                    assert model == "gpt-4o"

    def test_includes_codex_cli_when_available(self):
        from providers import get_available_providers

        with patch.dict("os.environ", {}, clear=True):
            with patch("providers.CODEX_AVAILABLE", True):
                with patch("providers.GEMINI_CLI_AVAILABLE", False):
                    available = get_available_providers()
                    provider_names = [name for name, _, _ in available]
                    assert "Codex CLI" in provider_names
                    # Verify the default model is gpt-5.3-codex
                    for name, key, model in available:
                        if name == "Codex CLI":
                            assert model == "codex/gpt-5.3-codex"
                            assert key is None

    def test_includes_gemini_cli_when_available(self):
        from providers import get_available_providers

        with patch.dict("os.environ", {}, clear=True):
            with patch("providers.CODEX_AVAILABLE", False):
                with patch("providers.GEMINI_CLI_AVAILABLE", True):
                    available = get_available_providers()
                    provider_names = [name for name, _, _ in available]
                    assert "Gemini CLI" in provider_names
                    # Verify the default model for Gemini CLI
                    for name, key, model in available:
                        if name == "Gemini CLI":
                            assert model == "gemini-cli/gemini-3-pro-preview"
                            assert key is None  # No API key required


class TestGetDefaultModel:
    def test_returns_first_available_model(self):
        from providers import get_default_model

        with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True):
            default = get_default_model()
            assert default == "gemini/gemini-2.0-flash"

    def test_returns_none_when_no_keys(self):
        from providers import get_default_model

        with patch.dict("os.environ", {}, clear=True):
            with patch("providers.CODEX_AVAILABLE", False):
                with patch("providers.GEMINI_CLI_AVAILABLE", False):
                    default = get_default_model()
                    assert default is None

    def test_prefers_bedrock_when_enabled(self):
        from providers import get_default_model

        with patch("providers.get_bedrock_config") as mock_config:
            mock_config.return_value = {
                "enabled": True,
                "available_models": ["claude-3-sonnet", "claude-3-haiku"],
            }
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
                default = get_default_model()
                assert default == "claude-3-sonnet"


class TestValidateModelCredentials:
    def test_validates_openai_models(self):
        from providers import validate_model_credentials

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            valid, invalid = validate_model_credentials(["gpt-4o", "gpt-4-turbo"])
            assert valid == ["gpt-4o", "gpt-4-turbo"]
            assert invalid == []

    def test_detects_missing_keys(self):
        from providers import validate_model_credentials

        with patch.dict("os.environ", {}, clear=True):
            valid, invalid = validate_model_credentials(["gpt-4o"])
            assert valid == []
            assert invalid == ["gpt-4o"]

    def test_validates_mixed_providers(self):
        from providers import validate_model_credentials

        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "XAI_API_KEY": "test-key",
            },
            clear=True,
        ):
            valid, invalid = validate_model_credentials(
                ["gpt-4o", "xai/grok-3", "gemini/gemini-2.0-flash"]
            )
            assert "gpt-4o" in valid
            assert "xai/grok-3" in valid
            assert "gemini/gemini-2.0-flash" in invalid

    def test_validates_codex_availability(self):
        from providers import validate_model_credentials

        with patch("providers.CODEX_AVAILABLE", True):
            valid, invalid = validate_model_credentials(["codex/gpt-5.2-codex"])
            assert valid == ["codex/gpt-5.2-codex"]
            assert invalid == []

        with patch("providers.CODEX_AVAILABLE", False):
            valid, invalid = validate_model_credentials(["codex/gpt-5.2-codex"])
            assert valid == []
            assert invalid == ["codex/gpt-5.2-codex"]

    def test_validates_codex_gpt53_availability(self):
        from providers import validate_model_credentials

        with patch("providers.CODEX_AVAILABLE", True):
            valid, invalid = validate_model_credentials(["codex/gpt-5.3-codex"])
            assert valid == ["codex/gpt-5.3-codex"]
            assert invalid == []

        with patch("providers.CODEX_AVAILABLE", False):
            valid, invalid = validate_model_credentials(["codex/gpt-5.3-codex"])
            assert valid == []
            assert invalid == ["codex/gpt-5.3-codex"]

    def test_validates_gemini_cli_availability(self):
        from providers import validate_model_credentials

        with patch("providers.GEMINI_CLI_AVAILABLE", True):
            valid, invalid = validate_model_credentials(
                ["gemini-cli/gemini-3-pro-preview"]
            )
            assert valid == ["gemini-cli/gemini-3-pro-preview"]
            assert invalid == []

        with patch("providers.GEMINI_CLI_AVAILABLE", False):
            valid, invalid = validate_model_credentials(
                ["gemini-cli/gemini-3-pro-preview"]
            )
            assert valid == []
            assert invalid == ["gemini-cli/gemini-3-pro-preview"]

    def test_defers_to_bedrock_validation_when_enabled(self):
        from providers import validate_model_credentials

        with patch("providers.get_bedrock_config") as mock_config:
            mock_config.return_value = {"enabled": True, "available_models": []}
            with patch("providers.validate_bedrock_models") as mock_validate:
                mock_validate.return_value = (["model1"], ["model2"])
                valid, invalid = validate_model_credentials(["model1", "model2"])
                assert valid == ["model1"]
                assert invalid == ["model2"]
                mock_validate.assert_called_once()
