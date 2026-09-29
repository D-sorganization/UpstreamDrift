import { forwardRef, useId } from 'react';
import type { ButtonHTMLAttributes } from 'react';

export type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'success' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  /**
   * #7443: when a button is disabled for a knowable reason, pass it here. The
   * reason is wired to both `title` (pointer tooltip) and a visually-hidden
   * `aria-describedby` node so assistive tech announces *why* the control is
   * dead instead of leaving the user guessing. Ignored when not disabled.
   */
  disabledReason?: string;
}

const BASE =
  'inline-flex items-center justify-center font-medium transition-colors ' +
  'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ' +
  'disabled:bg-gray-700 disabled:text-gray-400 disabled:cursor-not-allowed';

const VARIANTS: Record<ButtonVariant, string> = {
  // Canonical primary accent is blue (#7421).
  primary: 'bg-blue-600 hover:bg-blue-700 text-white',
  secondary: 'bg-gray-700 hover:bg-gray-600 text-gray-100',
  danger: 'bg-red-600 hover:bg-red-700 text-white',
  success: 'bg-green-600 hover:bg-green-700 text-white',
  ghost: 'bg-transparent hover:bg-gray-700 text-gray-200',
};

const SIZES: Record<ButtonSize, string> = {
  sm: 'px-2 py-1 text-xs rounded',
  md: 'px-3 py-1.5 text-sm rounded-md',
  lg: 'px-4 py-2 text-base rounded-md',
};

/**
 * Shared button primitive (UI/UX #7420). All new buttons must use this instead
 * of hand-rolled Tailwind strings so paddings, radii, focus rings, and disabled
 * states stay consistent app-wide.
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = 'primary',
    size = 'md',
    className = '',
    type = 'button',
    disabledReason,
    disabled,
    title,
    'aria-describedby': ariaDescribedBy,
    ...rest
  },
  ref
) {
  const classes = `${BASE} ${VARIANTS[variant]} ${SIZES[size]} ${className}`.trim();
  const reasonId = useId();
  // Only surface the reason while actually disabled (#7443).
  const showReason = Boolean(disabled && disabledReason);
  const describedBy =
    [ariaDescribedBy, showReason ? reasonId : null].filter(Boolean).join(' ') ||
    undefined;
  return (
    <>
      <button
        ref={ref}
        type={type}
        className={classes}
        disabled={disabled}
        title={showReason ? disabledReason : title}
        aria-describedby={describedBy}
        {...rest}
      />
      {showReason && (
        <span id={reasonId} className="sr-only">
          {disabledReason}
        </span>
      )}
    </>
  );
});
