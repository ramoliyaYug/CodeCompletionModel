// CodeCopilot — AI-powered inline code completion extension.
//
// Provides ghost-text suggestions (shown in a lighter color) as you type.
// Press Tab to accept a suggestion. Powered by a fine-tuned StarCoder2
// model running on a local FastAPI backend.

const vscode = require('vscode');
const { ApiClient } = require('./src/apiClient');
const { CodeCompletionProvider } = require('./src/completionProvider');
const { StatusBarManager } = require('./src/statusBar');

/**
 * Called by VS Code when the extension is activated (on first Python file open).
 * @param {vscode.ExtensionContext} context
 */
function activate(context) {
    console.log('[CodeCopilot] Extension activated');

    // ── Core services ──────────────────────────────────────────
    const apiClient = new ApiClient();
    const statusBar = new StatusBarManager();
    const completionProvider = new CodeCompletionProvider(apiClient, statusBar);

    // ── Register inline completion provider ────────────────────
    // Build a document selector from the configured language list.
    const config = vscode.workspace.getConfiguration('codecopilot');
    const languages = config.get('languages', ['python']);

    const selector = languages.map(lang => ({ language: lang }));

    const providerDisposable = vscode.languages.registerInlineCompletionItemProvider(
        selector,
        completionProvider
    );

    // ── Commands ───────────────────────────────────────────────
    const toggleCmd = vscode.commands.registerCommand(
        'codecopilot.toggle',
        () => {
            const cfg = vscode.workspace.getConfiguration('codecopilot');
            const isEnabled = cfg.get('enabled', true);
            cfg.update('enabled', !isEnabled, true); // global setting

            if (isEnabled) {
                statusBar.setDisabled();
                vscode.window.showInformationMessage('CodeCopilot: Disabled');
            } else {
                statusBar.setReady();
                vscode.window.showInformationMessage('CodeCopilot: Enabled');
            }
        }
    );

    const healthCmd = vscode.commands.registerCommand(
        'codecopilot.checkHealth',
        async () => {
            const health = await apiClient.checkHealth();

            if (health && health.status === 'ok') {
                statusBar.setReady();

                const gpu = health.gpuMemory
                    ? ` | GPU: ${health.gpuMemory.gpuName} (${health.gpuMemory.allocatedGB}GB / ${health.gpuMemory.totalGB}GB)`
                    : '';

                vscode.window.showInformationMessage(
                    `CodeCopilot: Server OK | Model: ${health.modelLoaded ? 'Loaded' : 'Not loaded'} | Device: ${health.device}${gpu}`
                );
            } else {
                statusBar.setError();
                const serverUrl = vscode.workspace
                    .getConfiguration('codecopilot')
                    .get('serverUrl', 'http://localhost:8000');

                vscode.window.showErrorMessage(
                    `CodeCopilot: Cannot reach server. Is the backend running at ${serverUrl}?`
                );
            }
        }
    );

    // ── React to configuration changes ─────────────────────────
    const configWatcher = vscode.workspace.onDidChangeConfiguration(e => {
        if (e.affectsConfiguration('codecopilot.enabled')) {
            const enabled = vscode.workspace
                .getConfiguration('codecopilot')
                .get('enabled', true);

            enabled ? statusBar.setReady() : statusBar.setDisabled();
        }
    });

    // ── Initial health check (non-blocking) ────────────────────
    apiClient.checkHealth().then(health => {
        const enabled = vscode.workspace
            .getConfiguration('codecopilot')
            .get('enabled', true);

        if (!enabled) {
            statusBar.setDisabled();
        } else if (health && health.status === 'ok' && health.modelLoaded) {
            statusBar.setReady();
        } else {
            statusBar.setError();
        }
    });

    // ── Register disposables ───────────────────────────────────
    context.subscriptions.push(
        providerDisposable,
        toggleCmd,
        healthCmd,
        configWatcher,
        statusBar
    );
}

/**
 * Called by VS Code when the extension is deactivated.
 */
function deactivate() {
    console.log('[CodeCopilot] Extension deactivated');
}

module.exports = { activate, deactivate };
