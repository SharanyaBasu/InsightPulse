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
  return <dialog ref={ref} className="terminal-dialog" aria-labelledby={titleId}
    onCancel={event => { event.preventDefault(); onClose(); }}
    onClick={event => { if (event.target === event.currentTarget) {
      const box = event.currentTarget.getBoundingClientRect();
      if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) onClose();
    } }}>
    <header className="panel-heading"><h2 id={titleId}>{title}</h2><button className="ui-button" onClick={onClose} autoFocus>Close <span aria-hidden="true">↗</span></button></header>
    <div className="panel-body">{children}</div>
  </dialog>;
}
