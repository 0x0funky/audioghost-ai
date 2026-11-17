"use client";

import { Loader2, X } from "lucide-react";
import { useEffect, useRef, type ButtonHTMLAttributes, type ReactNode } from "react";
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
          disabled={disabled}
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

export { cx };
