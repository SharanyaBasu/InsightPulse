import Dialog from "../Terminal/Dialog";
export default function KeyboardHelp({ onClose }) {
  return <Dialog title="Keyboard shortcuts" onClose={onClose}>
    <dl className="shortcut-list">{[["1–5", "Navigate pages"], ["r", "Refresh market data"], ["?", "Open this help"], ["Esc", "Close dialog"]].map(([key, text]) => <div key={key}><dt><kbd className="t-kbd">{key}</kbd></dt><dd>{text}</dd></div>)}</dl>
    <p className="muted">Shortcuts pause while you use form controls or a dialog.</p>
  </Dialog>;
}
