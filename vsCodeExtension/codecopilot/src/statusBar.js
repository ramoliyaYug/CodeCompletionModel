// Status bar item that shows CodeCopilot state in the bottom bar.
// Click it to run the health-check command.

const vscode = require('vscode');

class StatusBarManager {
    constructor() {
        this._item = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        this._item.command = 'codecopilot.checkHealth';
        this.setReady();
        this._item.show();
    }

    /** Server reachable and model loaded */
    setReady() {
        this._item.text = '$(sparkle) CodeCopilot';
        this._item.tooltip = 'CodeCopilot: Ready — click to check health';
        this._item.backgroundColor = undefined;
    }

    /** A completion request is in flight */
    setGenerating() {
        this._item.text = '$(sync~spin) CodeCopilot';
        this._item.tooltip = 'CodeCopilot: Generating completion…';
        this._item.backgroundColor = undefined;
    }

    /** Server unreachable or an error occurred */
    setError() {
        this._item.text = '$(error) CodeCopilot';
        this._item.tooltip = 'CodeCopilot: Server offline — click to retry';
        this._item.backgroundColor = new vscode.ThemeColor(
            'statusBarItem.errorBackground'
        );
    }

    /** Extension disabled via settings */
    setDisabled() {
        this._item.text = '$(circle-slash) CodeCopilot';
        this._item.tooltip = 'CodeCopilot: Disabled';
        this._item.backgroundColor = undefined;
    }

    /** Required by VS Code Disposable interface */
    dispose() {
        this._item.dispose();
    }
}

module.exports = { StatusBarManager };
