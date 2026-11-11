// API client for communicating with the CodeCopilot backend server.
// Uses native Node.js fetch (available in Node 18+ / VS Code 1.145+).

class ApiClient {
    /**
     * Read current configuration values from VS Code settings.
     * Called on every request so changes take effect immediately.
     * @returns {{ serverUrl: string, maxNewTokens: number, temperature: number, topP: number }}
     */
    _getConfig() {
        const vscode = require('vscode');
        const config = vscode.workspace.getConfiguration('codecopilot');
        return {
            serverUrl: config.get('serverUrl', 'http://localhost:8000'),
            maxNewTokens: config.get('maxNewTokens', 64),
            temperature: config.get('temperature', 0.2),
            topP: config.get('topP', 0.95),
        };
    }

    /**
     * Request a single code completion from the backend.
     *
     * @param {string} code        - The code prefix (text before the cursor).
     * @param {object} [options]   - Optional overrides.
     * @param {string|null} [options.fimSuffix]   - Code after cursor for Fill-In-The-Middle.
     * @param {number}      [options.maxNewTokens] - Override max tokens.
     * @param {number}      [options.temperature]  - Override temperature.
     * @param {number}      [options.topP]         - Override top-p.
     * @param {AbortSignal} [abortSignal]          - Signal to cancel the request.
     * @returns {Promise<string>} The generated completion text.
     */
    async getCompletion(code, options = {}, abortSignal) {
        const cfg = this._getConfig();

        const payload = {
            code,
            max_new_tokens: options.maxNewTokens ?? cfg.maxNewTokens,
            temperature: options.temperature ?? cfg.temperature,
            top_p: options.topP ?? cfg.topP,
            stop_tokens: ['\n\n', '\ndef ', '\nclass '],
            include_syntax_check: false,
        };

        if (options.fimSuffix) {
            payload.fim_suffix = options.fimSuffix;
        }

        const url = `${cfg.serverUrl}/api/v1/completion`;

        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: abortSignal,
        });

        if (!response.ok) {
            const body = await response.text().catch(() => '');
            throw new Error(`API error ${response.status}: ${body}`);
        }

        const data = await response.json();
        return data.completion || '';
    }

    /**
     * Ping the backend health endpoint.
     * @returns {Promise<object|null>} Health payload, or null if unreachable.
     */
    async checkHealth() {
        const cfg = this._getConfig();
        const url = `${cfg.serverUrl}/api/v1/health`;

        try {
            const response = await fetch(url, {
                method: 'GET',
                signal: AbortSignal.timeout(5000),
            });
            if (!response.ok) return null;
            return await response.json();
        } catch {
            return null;
        }
    }
}

module.exports = { ApiClient };
