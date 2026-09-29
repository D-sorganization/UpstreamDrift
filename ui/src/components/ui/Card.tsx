import type { HTMLAttributes } from 'react';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Use `dense` for inline control groups (p-2 instead of p-4). */
  dense?: boolean;
}

/**
 * Shared panel/card wrapper (UI/UX #7420): `bg-gray-800 rounded-md`, `p-4`
 * default (`p-2` when dense).
 */
export function Card({ dense = false, className = '', ...rest }: CardProps) {
  const padding = dense ? 'p-2' : 'p-4';
  const classes = `bg-gray-800 rounded-md ${padding} ${className}`.trim();
  return <div className={classes} {...rest} />;
}
