import { forwardRef } from 'react';
import type { SelectHTMLAttributes } from 'react';
import type { InputSize } from './Input';

export interface SelectProps
  extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  inputSize?: InputSize;
}

const BASE =
  'bg-gray-700 border border-gray-600 rounded-md text-white ' +
  'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 ' +
  'disabled:opacity-50 disabled:cursor-not-allowed';

const SIZES: Record<InputSize, string> = {
  md: 'px-3 py-2 text-sm',
  sm: 'px-2 py-1 text-xs',
};

/**
 * Shared select primitive (UI/UX #7420), matching Input styling.
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { inputSize = 'md', className = '', children, ...rest },
  ref
) {
  const classes = `${BASE} ${SIZES[inputSize]} ${className}`.trim();
  return (
    <select ref={ref} className={classes} {...rest}>
      {children}
    </select>
  );
});
