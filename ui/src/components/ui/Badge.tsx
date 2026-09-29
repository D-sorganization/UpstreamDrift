import type { HTMLAttributes } from 'react';

export type BadgeTone = 'neutral' | 'primary' | 'success' | 'warning' | 'danger';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
}

const TONES: Record<BadgeTone, string> = {
  neutral: 'bg-gray-700 text-gray-200',
  primary: 'bg-blue-900/80 text-blue-100',
  success: 'bg-green-900/80 text-green-100',
  warning: 'bg-amber-900/80 text-amber-100',
  danger: 'bg-red-900/80 text-red-100',
};

/**
 * Shared pill/badge primitive (UI/UX #7420). Semantic tones follow the #7421
 * palette (warning = amber, not yellow).
 */
export function Badge({ tone = 'neutral', className = '', ...rest }: BadgeProps) {
  const classes =
    `inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${TONES[tone]} ${className}`.trim();
  return <span className={classes} {...rest} />;
}
