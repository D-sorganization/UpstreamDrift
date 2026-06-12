import { forwardRef } from 'react';
import type { ButtonHTMLAttributes } from 'react';

export type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'success' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
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
  { variant = 'primary', size = 'md', className = '', type = 'button', ...rest },
  ref
) {
  const classes = `${BASE} ${VARIANTS[variant]} ${SIZES[size]} ${className}`.trim();
  return <button ref={ref} type={type} className={classes} {...rest} />;
});
