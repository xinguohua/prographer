"""Local secrets example.

Copy to ``local_settings.py`` and fill in your real values.
``local_settings.py`` is gitignored.
"""

# Required: an OpenAI-compatible API key (ChatAnywhere or another provider).
CHATANYWHERE_API_KEY = "PASTE_YOUR_KEY_HERE"

# Optional: your OpenAI-compatible chat/completions endpoint.
# Examples:
#   https://api.openai.com/v1/chat/completions
#   https://api.chatanywhere.tech/v1/chat/completions
CHATANYWHERE_ENDPOINT = "https://api.openai.com/v1/chat/completions"
