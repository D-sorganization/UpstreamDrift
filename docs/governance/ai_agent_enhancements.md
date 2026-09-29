# AI Agent Enhancements for UpstreamDrift

This document describes the AI agent enhancements implemented for the UpstreamDrift application, including Ollama model discovery, Claude Code/Codex CLI integration, subscription management, and agent control capabilities.

## Overview

The enhancements provide:

1. **Dynamic Ollama Model Discovery** - Refresh available models from local Ollama instance
2. **Claude Code CLI Integration** - Use Anthropic's Claude Code CLI for code analysis
3. **Codex CLI Integration** - Use OpenAI's Codex CLI for code generation
4. **Subscription Management** - Tier-based feature access (Free, Pro, Enterprise)
5. **Agent Control** - AI-powered control of all app aspects (engines, models, simulations)

## Files Modified/Created

### Modified Files

#### `src/shared/python/ai/gui/settings_dialog.py`

- Added `models_refreshed` signal to `ProviderConfigWidget`
- Added `_refresh_ollama_models()` method to fetch models from Ollama
- Added `_update_ollama_models()` method to update model dropdown
- Added "Refresh Available Models" button for Ollama provider
- Model count display showing discovered models

#### `src/shared/python/ai/sample_tools.py`

- Added `_register_agent_control_tools()` function
- Added `_register_cli_tools()` function
- Both functions register new tools with the existing tool registry

### New Files

#### `src/shared/python/ai/tools/cli_tools.py`

CLI tool adapters for external AI tools:

- `ClaudeCodeTool` - Interface with Claude Code CLI
- `CodexCLITool` - Interface with Codex CLI
- `ShellTool` - Secure shell command execution with allowlist
- `CLIToolManager` - Unified manager for all CLI tools

#### `src/shared/python/ai/tools/agent_control.py`

Agent control for app management:

- `AgentController` - Main controller class
- Engine control (start/stop/configure)
- Model management (load/unload/compare)
- File operations (import/export)
- Simulation control (start/stop/pause/resume)
- Settings management
- System status reporting

#### `src/shared/python/ai/auth/authentication.py`

Authentication and subscription management:

- `SubscriptionTier` enum (FREE, PRO, ENTERPRISE)
- `UserProfile` dataclass
- `AuthManager` class for credential management
- `FeatureGate` decorator for feature access control
- API key and OAuth login support

#### `src/shared/python/ai/auth/__init__.py`

Package initialization with exports.

#### `src/shared/python/ai/tools/__init__.py`

Package initialization with exports.

## Usage

### Ollama Model Refresh

In the AI Settings dialog (Provider tab):

1. Select "Ollama (Local - FREE)" as provider
2. Click "Test Connection" to verify Ollama is running
3. Click "🔄 Refresh Available Models" to fetch installed models
4. The model dropdown will update with discovered models

### Claude Code CLI Integration

```python
from src.shared.python.ai.tools.cli_tools import ClaudeCodeTool

tool = ClaudeCodeTool()
if tool.is_available():
    result = tool.ask("Explain this code")
    print(result.output)
```

**Installation:** `npm install -g @anthropic-ai/claude-code`

### Codex CLI Integration

```python
from src.shared.python.ai.tools.cli_tools import CodexCLITool

tool = CodexCLITool()
if tool.is_available():
    result = tool.generate("Create a function to sort models", language="python")
    print(result.output)
```

**Installation:** `pip install openai-cli` or `npm install -g @openai/codex-cli`

### Agent Control

```python
from src.shared.python.ai.tools.agent_control import AgentController

controller = AgentController()

# Start an engine
result = controller.start_engine("mujoco")
if result.success:
    # Load a model
    controller.load_model("humanoid_golf")
    # Run simulation
    controller.start_simulation("humanoid_golf", duration=10.0)
```

### Subscription Management

```python
from src.shared.python.ai.auth.authentication import AuthManager, SubscriptionTier

auth = AuthManager()

# Login with API key
auth.login_with_api_key("your-api-key")

# Check feature access
if auth.has_feature("claude_code"):
    # Enable Claude Code integration
    pass

# Check subscription tier
if auth.subscription_tier == SubscriptionTier.PRO:
    # Enable pro features
    pass
```

### Feature Gating

```python
from src.shared.python.ai.auth.authentication import FeatureGate

@FeatureGate.require("claude_code")
def use_claude_code():
    # Only accessible to users with claude_code feature
    pass

@FeatureGate.require_tier(SubscriptionTier.PRO)
def access_pro_features():
    # Only accessible to PRO and ENTERPRISE users
    pass
```

## Subscription Tiers

| Feature             | FREE | PRO | ENTERPRISE |
| ------------------- | ---- | --- | ---------- |
| Ollama Chat         | ✓    | ✓   | ✓          |
| Basic Tools         | ✓    | ✓   | ✓          |
| Local Models        | ✓    | ✓   | ✓          |
| Claude Code         | -    | ✓   | ✓          |
| Codex CLI           | -    | ✓   | ✓          |
| Cloud Models        | -    | ✓   | ✓          |
| Priority Support    | -    | ✓   | ✓          |
| Advanced Tools      | -    | ✓   | ✓          |
| Custom Integrations | -    | -   | ✓          |
| Dedicated Support   | -    | -   | ✓          |
| SSO Auth            | -    | -   | ✓          |
| Audit Logs          | -    | -   | ✓          |

## Security Considerations

1. **API Key Storage**: Keys stored in system keyring (Windows Credential Manager, macOS Keychain)
2. **Credential File**: `~/.golf_modeling_suite/auth_credentials.json` with 0600 permissions
3. **Shell Tool**: Implements allowlist-based command filtering, dangerous commands blocked
4. **Feature Gates**: Server-side validation recommended for production use

## Future Enhancements

1. **OAuth Integration**: Complete OAuth flow for Google, GitHub providers
2. **Subscription API**: Integration with payment provider for subscription management
3. **Advanced Agent Tools**: More granular control over simulation parameters
4. **Multi-Engine Orchestration**: Coordinate multiple engines simultaneously
5. **Conversation Persistence**: Cross-session conversation history sync
6. **RAG Enhancement**: Codebase-aware responses with improved indexing

## Troubleshooting

### Ollama Models Not Refreshing

- Ensure Ollama is running: `ollama serve`
- Check connection in settings dialog
- Verify OLLAMA_HOST environment variable

### Claude Code Not Available

- Install: `npm install -g @anthropic-ai/claude-code`
- Verify: `claude --version`
- Check PATH environment variable

### Codex CLI Not Available

- Install: `pip install openai-cli`
- Set API key: `export OPENAI_API_KEY=your-key`
- Verify: `codex --version`

### Authentication Issues

- Check credentials file: `~/.golf_modeling_suite/auth_credentials.json`
- Verify file permissions: `chmod 600 ~/.golf_modeling_suite/auth_credentials.json`
- Try logout and re-login
