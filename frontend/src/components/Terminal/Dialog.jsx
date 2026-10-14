import { useEffect, useRef, useId } from "react";

export default function Dialog({ title, onClose, children }) {
  const ref = useRef(null);
  const titleId = useId();
  useEffect(() => {
    const opener = document.activeElement;
    const dialog = ref.current;
    dialog.showModal();
    return () => { dialog.close(); opener?.focus(); };
  }, []);
  function keepFocusInside(event) {
    if (event.key !== "Tab") return;
    // Native modal dialogs can tab into browser chrome on some platforms.
    // Resolve controls on each keypress because chart/retry content changes.
    const controls = [...event.currentTarget.querySelectorAll(
      'button, a[href], input, select, textarea, summary, [tabindex]'
    )].filter(element => element.tabIndex >= 0 && !element.matches(':disabled, [inert], [inert] *') && element.getClientRects().length > 0);
    const first = controls[0];
    const last = controls.at(-1);
    if (!first) {
      event.preventDefault();
      return;
    }
    const active = document.activeElement;
    if (event.shiftKey && (active === first || active === event.currentTarget)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (active === last || active === event.currentTarget)) {
      event.preventDefault();
      first.focus();
    }
  }
  return <dialog ref={ref} className="terminal-dialog" aria-labelledby={titleId}
    onKeyDown={keepFocusInside}
    onCancel={event => { event.preventDefault(); onClose(); }}
    onClick={event => { if (event.target === event.currentTarget) {
      const box = event.currentTarget.getBoundingClientRect();
      if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) onClose();
    } }}>
    <header className="panel-heading"><h2 id={titleId}>{title}</h2><button className="ui-button" onClick={onClose} autoFocus>Close <span aria-hidden="true">↗</span></button></header>
    <div className="panel-body">{children}</div>
  </dialog>;
}
