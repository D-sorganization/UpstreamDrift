import { useState, useEffect, createContext, useContext, useCallback, ReactNode } from 'react';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';
import { readCachedSettings } from '@/api/settingsClient';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

interface Toast {
  id: string;
  message: string;
  type: ToastType;
  duration?: number;
  /** Number of coalesced occurrences of an identical (message, type) toast. */
  count: number;
}

/** Maximum simultaneously-visible toasts; oldest are dropped beyond this. */
const MAX_TOASTS = 5;

interface ToastContextType {
  showToast: (message: string, type?: ToastType, duration?: number) => void;
  showSuccess: (message: string) => void;
  showError: (message: string) => void;
  showWarning: (message: string) => void;
  showInfo: (message: string) => void;
}

const ToastContext = createContext<ToastContextType | null>(null);

const ICONS = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};

const STYLES = {
  success: 'bg-green-900/90 border-green-500 text-green-100',
  error: 'bg-red-900/90 border-red-500 text-red-100',
  warning: 'bg-amber-900/90 border-amber-500 text-amber-100',
  info: 'bg-blue-900/90 border-blue-500 text-blue-100',
};

const ICON_STYLES = {
  success: 'text-green-400',
  error: 'text-red-400',
  warning: 'text-amber-400',
  info: 'text-blue-400',
};

function ToastItem({ toast, onClose }: { toast: Toast; onClose: () => void }) {
  const Icon = ICONS[toast.type];
  const assertive = toast.type === 'error';

  useEffect(() => {
    if (toast.duration) {
      const timer = setTimeout(onClose, toast.duration);
      return () => clearTimeout(timer);
    }
    // `toast.count` is a dep so a coalesced duplicate restarts the dismiss timer.
  }, [toast.duration, toast.count, onClose]);

  return (
    <div
      role={assertive ? 'alert' : 'status'}
      aria-live={assertive ? 'assertive' : 'polite'}
      aria-atomic="true"
      className={`flex items-center gap-3 px-4 py-3 rounded-lg border shadow-lg
                  backdrop-blur-sm animate-in slide-in-from-right-5 duration-200
                  ${STYLES[toast.type]}`}
    >
      <Icon className={`w-5 h-5 flex-shrink-0 ${ICON_STYLES[toast.type]}`} aria-hidden="true" />
      <p className="text-sm font-medium flex-1">{toast.message}</p>
      {toast.count > 1 && (
        <span
          className="text-xs font-semibold px-1.5 py-0.5 rounded-full bg-white/15"
          aria-label={`${toast.count} occurrences`}
        >
          ×{toast.count}
        </span>
      )}
      <button
        onClick={onClose}
        className="p-1 rounded hover:bg-white/10 transition-colors focus:outline-none focus:ring-2 focus:ring-white/20"
        aria-label="Dismiss notification"
      >
        <X className="w-4 h-4" aria-hidden="true" />
      </button>
    </div>
  );
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showToast = useCallback((message: string, type: ToastType = 'info', duration?: number) => {
    // Notification preferences (#7457): the cached server settings control
    // default duration and verbosity. An explicit duration argument wins.
    const prefs = readCachedSettings()?.notifications;
    const verbosity = prefs?.verbosity ?? 'all';
    if (verbosity === 'silent') return;
    if (verbosity === 'errors' && type !== 'error' && type !== 'warning') return;
    const resolvedDuration = duration ?? prefs?.toast_duration_ms ?? 4000;
    const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    setToasts((prev) => {
      // Deduplicate: an identical (message, type) toast already on screen is
      // coalesced — bump its count and restart its timer instead of stacking.
      const existing = prev.find((t) => t.message === message && t.type === type);
      if (existing) {
        return prev.map((t) =>
          t === existing
            ? { ...t, count: t.count + 1, duration: resolvedDuration }
            : t,
        );
      }
      // Cap the stack: drop the oldest beyond MAX_TOASTS.
      return [
        ...prev,
        { id, message, type, duration: resolvedDuration, count: 1 },
      ].slice(-MAX_TOASTS);
    });
  }, []);

  const showSuccess = useCallback((message: string) => showToast(message, 'success'), [showToast]);
  const showError = useCallback((message: string) => showToast(message, 'error', 6000), [showToast]);
  const showWarning = useCallback((message: string) => showToast(message, 'warning'), [showToast]);
  const showInfo = useCallback((message: string) => showToast(message, 'info'), [showToast]);

  return (
    <ToastContext.Provider value={{ showToast, showSuccess, showError, showWarning, showInfo }}>
      {children}
      {/* Toast container */}
      <div
        className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm"
        aria-label="Notifications"
      >
        {toasts.map((toast) => (
          <ToastItem
            key={toast.id}
            toast={toast}
            onClose={() => removeToast(toast.id)}
          />
        ))}
      </div>
    </ToastContext.Provider>
  );
}

// Hook is tightly coupled to ToastProvider - standard React pattern
// eslint-disable-next-line react-refresh/only-export-components
export function useToast(): ToastContextType {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}
