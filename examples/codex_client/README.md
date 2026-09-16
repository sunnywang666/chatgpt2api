# Isolated Codex CLI acceptance fixture

Use `run-codex.sh` on macOS/Linux or `run-codex.ps1` on Windows. Both launchers:

1. call `GET <base-url>/models`; the `--list-models` mode needs no model, while a launch requires an exact returned ID;
2. create a task-local `CODEX_HOME` containing only generated provider routing;
3. use the ordinary bearer key for discovery, then pass it as `OPENAI_API_KEY` only to the final Codex child process;
4. ask Codex to fix the intentionally broken `acceptance-fixture/calculator.py` and run its standard-library test.

The generated config has no credential. Neither launcher copies normal Codex login/configuration. Do not use Codex debug logging for this acceptance flow.

Use [docs/codex-client.md](../../docs/codex-client.md) for complete commands, resume instructions, and the difference between this fixture and a real provider acceptance.
