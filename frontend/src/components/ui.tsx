"use client";

import { Loader2, X } from "lucide-react";
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import styles from "./ui.module.css";

type Variant = "default" | "primary" | "strong" | "ghost" | "dashed";

const cx = (...names: (string | false | null | undefined)[]) => names.filter(Boolean).join(" ");

export function Button({
  variant = "default",
  size = "medium",
  block,
  danger,
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  size?: "small" | "medium" | "large";
  block?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      className={cx(
        styles.button,
        variant !== "default" && styles[variant],
        size === "small" && styles.small,
        size === "large" && styles.large,
        block && styles.block,
        danger && styles.danger,
        className,
      )}
      {...props}
    />
  );
}

export function IconButton({
  label,
  size = "medium",
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; size?: "medium" | "small" | "tiny" }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cx(
        styles.iconButton,
        size === "small" && styles.iconSmall,
        size === "tiny" && styles.iconTiny,
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
  hint?: string;
  title?: string;
  disabled?: boolean;
}

/** A row of buttons, one pressed. */
export function Segmented<T extends string>({
  label,
  options,
  value,
  onChange,
  disabled,
}: {
  label: string;
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
}) {
  return (
    <div role="group" aria-label={label} className={styles.segmented}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={styles.segment}
          aria-pressed={option.value === value}
          title={option.title}
          disabled={disabled || option.disabled}
          onClick={() => onChange(option.value)}
        >
          <span className={styles.segmentLabel}>{option.label}</span>
          {option.hint && <span className={styles.segmentHint}>{option.hint}</span>}
        </button>
      ))}
    </div>
  );
}

export function Switch({
  checked,
  onChange,
  title,
  hint,
  disabled,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  title: string;
  hint?: ReactNode;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      className={styles.switchRow}
      disabled={disabled}
      onClick={() => onChange(!checked)}
    >
      <span className={styles.switchText}>
        <span className={styles.switchTitle}>{title}</span>
        {hint && <span className={styles.switchHint}>{hint}</span>}
      </span>
      <span className={styles.switchTrack} aria-hidden="true" />
    </button>
  );
}

export function ProgressBar({ value, label }: { value: number; label: string }) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div
      className={styles.progress}
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={percent}
    >
      <div className={styles.progressFill} style={{ width: `${percent}%` }} />
    </div>
  );
}

export function Spinner({ size = 16 }: { size?: number }) {
  return <Loader2 size={size} className={styles.spinner} aria-hidden="true" />;
}

export function ErrorText({ children }: { children: ReactNode }) {
  return (
    <div role="alert" className={styles.error}>
      {children}
    </div>
  );
}

export function SectionLabel({ children, as: Tag = "h3" }: { children: ReactNode; as?: "h2" | "h3" | "span" }) {
  return <Tag className={styles.sectionLabel}>{children}</Tag>;
}

/** Native modal <dialog>: focus trap, Esc and the backdrop come for free. */
export function Dialog({
  open,
  onClose,
  title,
  subtitle,
  children,
  footer,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  children: ReactNode;
  footer?: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      className={styles.dialog}
      aria-labelledby="dialog-title"
      onClose={onClose}
      onClick={(event) => {
        if (event.target === ref.current) onClose();
      }}
    >
      {open && (
        <>
          <div className={styles.dialogHeader}>
            <h2 id="dialog-title" className={styles.dialogTitle}>
              {title}
            </h2>
            {subtitle && <span className={styles.dialogSubtitle}>{subtitle}</span>}
            <IconButton label="Close" size="small" className={styles.dialogClose} onClick={onClose}>
              <X size={16} aria-hidden="true" />
            </IconButton>
          </div>
          <div className={styles.dialogBody}>{children}</div>
          {footer && <div className={styles.dialogFooter}>{footer}</div>}
        </>
      )}
    </dialog>
  );
}

export interface MenuItem {
  label: string;
  hint?: string;
  icon?: ReactNode;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
}

/**
 * A button that opens a small menu. The menu is drawn on top of the whole page
 * (a portal, positioned against the window), so neither the scrolling panes it
 * opens from nor the sticky track heads can clip or cover it. ``null`` in
 * ``items`` is a divider.
 */
export function Menu({
  label,
  items,
  children,
  className,
  align = "end",
  disabled,
}: {
  label: string;
  items: (MenuItem | null)[];
  children: ReactNode;
  className?: string;
  align?: "start" | "end";
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState<{ left: number; top: number } | null>(null);
  const button = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    if (!open || !button.current || !list.current) return;
    const b = button.current.getBoundingClientRect();
    const m = list.current.getBoundingClientRect();
    const left = Math.max(8, Math.min(window.innerWidth - m.width - 8, align === "end" ? b.right - m.width : b.left));
    const below = b.bottom + 4;
    const top = below + m.height > window.innerHeight - 8 ? Math.max(8, b.top - m.height - 4) : below;
    setPlace({ left, top });
    list.current.querySelector<HTMLButtonElement>("button:not(:disabled)")?.focus();
  }, [open, align]);

  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    const outside = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!list.current?.contains(target) && !button.current?.contains(target)) close();
    };
    const key = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        close();
        button.current?.focus();
      }
    };
    window.addEventListener("pointerdown", outside, true);
    window.addEventListener("keydown", key, true);
    window.addEventListener("resize", close);
    window.addEventListener("scroll", close, true);
    return () => {
      window.removeEventListener("pointerdown", outside, true);
      window.removeEventListener("keydown", key, true);
      window.removeEventListener("resize", close);
      window.removeEventListener("scroll", close, true);
    };
  }, [open]);

  const move = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const buttons = [...(list.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? [])];
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const next = (index + (event.key === "ArrowDown" ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next]?.focus();
  };

  const menu = (
    <div
      ref={list}
      role="menu"
      aria-label={label}
      className={styles.menu}
      style={{ left: place?.left ?? 0, top: place?.top ?? 0, visibility: place ? "visible" : "hidden" }}
      onKeyDown={move}
    >
      {items.map((item, index) =>
        item === null ? (
          <div key={`divider-${index}`} className={styles.menuDivider} role="separator" />
        ) : (
          <button
            key={item.label}
            type="button"
            role="menuitem"
            className={cx(styles.menuItem, item.danger && styles.menuDanger)}
            disabled={item.disabled}
            onClick={() => {
              setOpen(false);
              button.current?.focus();
              item.onSelect();
            }}
          >
            {item.icon && <span className={styles.menuIcon}>{item.icon}</span>}
            <span className={styles.menuText}>
              <span>{item.label}</span>
              {item.hint && <span className={styles.menuHint}>{item.hint}</span>}
            </span>
          </button>
        ),
      )}
    </div>
  );

  return (
    <>
      <button
        ref={button}
        type="button"
        className={className}
        aria-label={label}
        title={label}
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={disabled}
        onClick={() => {
          setPlace(null);
          setOpen((value) => !value);
        }}
      >
        {children}
      </button>
      {open && createPortal(menu, document.body)}
    </>
  );
}

/**
 * A bar between two panes that is dragged to resize one of them. ``value`` is
 * that pane's size in pixels. ``direction``: the way the size grows when the
 * pointer moves right or down (-1 for a pane on the right or at the bottom).
 */
export function Splitter({
  orientation,
  label,
  value,
  min,
  max,
  direction = 1,
  onChange,
  onReset,
}: {
  /** "vertical": a vertical bar between a left and a right pane. */
  orientation: "vertical" | "horizontal";
  label: string;
  value: number;
  min: number;
  max: number;
  direction?: 1 | -1;
  onChange: (value: number) => void;
  onReset?: () => void;
}) {
  const drag = useRef<{ from: number; value: number } | null>(null);
  const clamp = (next: number) => Math.round(Math.max(min, Math.min(max, next)));
  const position = (event: ReactPointerEvent) => (orientation === "vertical" ? event.clientX : event.clientY);
  return (
    <div
      role="separator"
      tabIndex={0}
      aria-label={label}
      aria-orientation={orientation}
      aria-valuenow={Math.round(value)}
      aria-valuemin={min}
      aria-valuemax={Math.round(max)}
      title={`${label} (drag, or double-click to reset)`}
      className={cx(styles.splitter, orientation === "vertical" ? styles.splitterVertical : styles.splitterHorizontal)}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        drag.current = { from: position(event), value };
      }}
      onPointerMove={(event) => {
        if (drag.current) onChange(clamp(drag.current.value + direction * (position(event) - drag.current.from)));
      }}
      onPointerUp={() => {
        drag.current = null;
      }}
      onPointerCancel={() => {
        drag.current = null;
      }}
      onDoubleClick={onReset}
      onKeyDown={(event) => {
        const grow = orientation === "vertical" ? "ArrowRight" : "ArrowDown";
        const shrink = orientation === "vertical" ? "ArrowLeft" : "ArrowUp";
        if (event.key === grow || event.key === shrink) {
          event.preventDefault();
          onChange(clamp(value + direction * (event.key === grow ? 24 : -24)));
        } else if (event.key === "Enter" && onReset) {
          event.preventDefault();
          onReset();
        }
      }}
    />
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className={styles.kbd}>{children}</kbd>;
}

export { cx };
