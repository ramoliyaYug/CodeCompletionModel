// Inline completion provider that shows AI-generated ghost text.
// Ghost text appears in a lighter color; pressing Tab accepts the suggestion.

const vscode = require('vscode');

/**
 * Maximum number of characters to send as prefix context.
 * Keeps requests within the backend's 1024-token context window.
 * (~4 chars per token × 960 available tokens ≈ 3 840 chars)
 */
const MAX_PREFIX_CHARS = 3000;

/**
 * Maximum number of characters to send as FIM suffix context.
 */
const MAX_SUFFIX_CHARS = 500;

class CodeCompletionProvider {
    /**
     * @param {import('./apiClient').ApiClient} apiClient
     * @param {import('./statusBar').StatusBarManager} statusBar
     */
    constructor(apiClient, statusBar) {
        this._apiClient = apiClient;
        this._statusBar = statusBar;
    }

    /**
     * Called by VS Code whenever inline completions are needed.
     * Implements debounce, cancellation, context extraction, and API call.
     *
     * @param {vscode.TextDocument} document
     * @param {vscode.Position} position
     * @param {vscode.InlineCompletionContext} context
     * @param {vscode.CancellationToken} token
     * @returns {Promise<vscode.InlineCompletionItem[]>}
     */
    async provideInlineCompletionItems(document, position, context, token) {
        const config = vscode.workspace.getConfiguration('codecopilot');

        // ── Guard: extension disabled ──────────────────────────────
        if (!config.get('enabled', true)) {
            return [];
        }

        // ── Debounce ───────────────────────────────────────────────
        // Wait a short period so we don't fire on every keystroke.
        // If the user keeps typing, VS Code cancels the old token and
        // calls us again — the Promise below resolves with `true`
        // and we bail out.
        const debounceMs = config.get('debounceMs', 300);
        const wasCancelled = await new Promise(resolve => {
            const timer = setTimeout(() => {
                cancelDisposable.dispose();
                resolve(false);
            }, debounceMs);

            const cancelDisposable = token.onCancellationRequested(() => {
                clearTimeout(timer);
                resolve(true);
            });
        });

        if (wasCancelled || token.isCancellationRequested) {
            return [];
        }

        // ── Extract code prefix (everything before the cursor) ────
        let prefix = document.getText(
            new vscode.Range(new vscode.Position(0, 0), position)
        );

        // Don't trigger on very short / empty content
        if (prefix.trim().length < 3) {
            return [];
        }

        // Limit prefix length to stay within context window
        if (prefix.length > MAX_PREFIX_CHARS) {
            prefix = prefix.substring(prefix.length - MAX_PREFIX_CHARS);
        }

        // ── Extract FIM suffix (everything after the cursor) ──────
        let fimSuffix = null;
        if (config.get('enableFim', true)) {
            const lastLine = document.lineCount - 1;
            const lastChar = document.lineAt(lastLine).text.length;
            const suffix = document.getText(
                new vscode.Range(position, new vscode.Position(lastLine, lastChar))
            );
            if (suffix.trim().length > 0) {
                fimSuffix = suffix.substring(0, MAX_SUFFIX_CHARS);
            }
        }

        // ── Prepare cancellable fetch ─────────────────────────────
        const abortController = new AbortController();
        const cancelDisposable = token.onCancellationRequested(() => {
            abortController.abort();
        });

        try {
            this._statusBar.setGenerating();

            const completion = await this._apiClient.getCompletion(
                prefix,
                { fimSuffix },
                abortController.signal
            );

            this._statusBar.setReady();

            if (!completion || completion.trim().length === 0) {
                return [];
            }

            // Create the inline completion item.
            // VS Code renders this as ghost text (lighter color) at the
            // cursor position. Pressing Tab accepts the suggestion.
            const item = new vscode.InlineCompletionItem(
                completion,
                new vscode.Range(position, position)
            );

            return [item];
        } catch (err) {
            // AbortError is expected when the user keeps typing
            if (err.name === 'AbortError') {
                return [];
            }

            console.error('[CodeCopilot] Completion error:', err.message);
            this._statusBar.setError();
            return [];
        } finally {
            cancelDisposable.dispose();
        }
    }
}

module.exports = { CodeCompletionProvider };
