import { useId } from "react";

export default function TerminalPanel({ title, headerRight, children, style, className = "" }) {
  const id = useId();
  return (
    <section className={`terminal-panel ${className}`} style={style} aria-labelledby={title ? id : undefined}>
      {title && <header className="panel-heading"><h2 id={id}>{title}</h2>{headerRight}</header>}
      <div className="panel-body">{children}</div>
    </section>
  );
}
