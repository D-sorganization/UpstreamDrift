import { Wifi, WifiOff, Loader2, AlertTriangle } from 'lucide-react';
import type { ConnectionStatus as ConnectionStatusType } from '@/api/client';

interface Props {
  status: ConnectionStatusType;
  className?: string;
  /**
   * #7435: when the connection is `lost`, render an inline "Reconnect"
   * affordance instead of a dead-end message. Wire this to the page's start()
   * so the stale banner clears the moment the user acts.
   */
  onReconnect?: () => void;
}

const STATUS_CONFIG = {
  disconnected: {
    icon: WifiOff,
    text: 'Disconnected',
    color: 'text-gray-400',
    bgColor: 'bg-gray-700',
    pulseColor: '',
  },
  connecting: {
    icon: Loader2,
    text: 'Connecting...',
    color: 'text-blue-400',
    bgColor: 'bg-blue-900/50',
    pulseColor: '',
    animate: true,
  },
  connected: {
    icon: Wifi,
    text: 'Connected',
    color: 'text-green-400',
    bgColor: 'bg-green-900/50',
    pulseColor: 'bg-green-500',
  },
  reconnecting: {
    icon: Loader2,
    text: 'Reconnecting...',
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-900/50',
    pulseColor: '',
    animate: true,
  },
  failed: {
    icon: AlertTriangle,
    text: 'Connection Failed',
    color: 'text-red-400',
    bgColor: 'bg-red-900/50',
    pulseColor: '',
  },
  // #6896: connection dropped mid-run and cannot be resumed — the user must
  // explicitly restart. We do NOT imply continuity ("Reconnecting…") here.
  lost: {
    icon: AlertTriangle,
    text: 'Connection lost — restart required',
    color: 'text-amber-400',
    bgColor: 'bg-amber-900/50',
    pulseColor: '',
  },
};

export function ConnectionStatus({
  status,
  className = '',
  onReconnect,
}: Props) {
  const config = STATUS_CONFIG[status];
  const Icon = config.icon;
  const showReconnect = status === 'lost' && onReconnect != null;

  return (
    <div
      className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${config.bgColor} ${className}`}
      role="status"
      aria-live="polite"
      aria-label={`Connection status: ${config.text}`}
    >
      {/* Status indicator dot */}
      <span className="relative flex h-2 w-2">
        {config.pulseColor && (
          <span
            className={`animate-ping absolute inline-flex h-full w-full rounded-full ${config.pulseColor} opacity-75`}
          />
        )}
        <span
          className={`relative inline-flex rounded-full h-2 w-2 ${
            config.pulseColor || config.color.replace('text-', 'bg-')
          }`}
        />
      </span>

      {/* Icon */}
      <Icon
        className={`w-4 h-4 ${config.color} ${
          'animate' in config && config.animate ? 'animate-spin' : ''
        }`}
        aria-hidden="true"
      />

      {/* Text */}
      <span className={`text-xs font-medium ${config.color}`}>{config.text}</span>

      {/* #7435: actionable reconnect instead of a dead-end "lost" banner. */}
      {showReconnect && (
        <button
          type="button"
          onClick={onReconnect}
          className="ml-1 rounded-full px-2 py-0.5 text-xs font-semibold text-amber-200 bg-amber-800/60 hover:bg-amber-700/70 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-400"
        >
          Reconnect
        </button>
      )}
    </div>
  );
}
