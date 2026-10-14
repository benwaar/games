import * as vscode from 'vscode';
import { ScrEditorProvider } from './scrEditor';
import { TapEditorProvider } from './tapEditor';

export function activate(context: vscode.ExtensionContext): void {
  context.subscriptions.push(ScrEditorProvider.register(context));
  context.subscriptions.push(TapEditorProvider.register(context));
}

export function deactivate(): void {}
