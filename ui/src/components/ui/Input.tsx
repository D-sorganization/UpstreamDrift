import { forwardRef } from 'react';
import type { InputHTMLAttributes } from 'react';

export type InputSize = 'sm' | 'md';

export interface InputProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  inputSize?: InputSize;
}

const BASE =
  'bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-500 ' +
  'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ' +
  'disabled:opacity-50 disabled:cursor-not-allowed';

const SIZES: Record<InputSize, string> = {
  md: 'px-3 py-2 text-sm',
  sm: 'px-2 py-1 text-xs',
};

/**
 * Shared text-input primitive (UI/UX #7420). Use `inputSize="sm"` for dense
 * panels.
 */
export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { inputSize = 'md', className = '', ...rest },
  ref
) {
  const classes = `${BASE} ${SIZES[inputSize]} ${className}`.trim();
  return <input ref={ref} className={classes} {...rest} />;
});
