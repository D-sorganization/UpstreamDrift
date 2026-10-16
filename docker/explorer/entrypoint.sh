#!/bin/sh
# Model-explorer container entrypoint (UpstreamDrift #12878, workstream D).
#
# Usage (first argument selects the mode):
#   engines                         JSON report of every engine + smoke test
#   probe-gl                        check MUJOCO_GL can render offscreen
#   export [--model M --out F ...]  open a model, run/replay it, write an MP4
#   app-screenshot --out F          construct the real Qt window, save a PNG
#   gui [APP_ARGS...]               run the visible PyQt6 Model Explorer
#                                   (Xvfb+noVNC, or the host display; see below)
#   <anything else>                 exec'd verbatim (e.g. `bash`)
#
# Design by Contract: every precondition is checked up front and fails with
# `ERROR: <what is wrong and how to fix it>` and exit status 64, instead of a
# segfault or a blank window later.
#
# Environment:
#   QT_QPA_PLATFORM            offscreen (headless) | xcb (GUI)        [required]
#   MUJOCO_GL                  egl | osmesa                            [required]
#   DISPLAY                    X display; :99 for the built-in Xvfb    [GUI]
#   UD_EXPLORER_DISPLAY_MODE   vnc (default) | host (WSLg / host X11)
#   VNC_GEOMETRY               Xvfb screen, default 1600x900x24
#   NOVNC_PORT / VNC_PORT      default 6080 / 5900
#   UD_GUI_APP                 python module to run, default
#                              src.tools.model_explorer.launch_model_explorer
set -eu

CLI="python /workspace/scripts/docker/explorer_cli.py"

die() {
    echo "ERROR: $*" >&2
    exit 64
}

require_env() {
    [ -n "${QT_QPA_PLATFORM:-}" ] || die "QT_QPA_PLATFORM is not set (offscreen for headless, xcb for the GUI)"
    [ -n "${MUJOCO_GL:-}" ] || die "MUJOCO_GL is not set (use egl, or osmesa for pure software rendering)"
    case "$MUJOCO_GL" in
        egl | osmesa) ;;
        *) die "MUJOCO_GL=$MUJOCO_GL cannot render inside a container; use egl or osmesa" ;;
    esac
}

require_display() {
    [ -n "${DISPLAY:-}" ] || die "no display: DISPLAY is not set (the explorer-gui service sets :99; for WSLg use the explorer-gui-wslg service)"
}

wait_for_x_socket() {
    # $1 = display number (e.g. 99). Wait up to ~10 s for /tmp/.X11-unix/X<n>.
    tries=0
    while [ ! -S "/tmp/.X11-unix/X$1" ]; do
        tries=$((tries + 1))
        [ "$tries" -le 50 ] || return 1
        sleep 0.2
    done
}

start_vnc_display() {
    display_number="${DISPLAY#:}"
    display_number="${display_number%%.*}"
    Xvfb "$DISPLAY" -screen 0 "${VNC_GEOMETRY:-1600x900x24}" -nolisten tcp >/tmp/xvfb.log 2>&1 &
    wait_for_x_socket "$display_number" || {
        cat /tmp/xvfb.log >&2
        die "Xvfb did not create /tmp/.X11-unix/X$display_number within 10 s"
    }
    openbox >/tmp/openbox.log 2>&1 &
    # x11vnc listens on loopback inside the container; only noVNC is published.
    x11vnc -display "$DISPLAY" -forever -shared -nopw -localhost \
        -rfbport "${VNC_PORT:-5900}" -quiet >/tmp/x11vnc.log 2>&1 &
    websockify --web /usr/share/novnc "${NOVNC_PORT:-6080}" "localhost:${VNC_PORT:-5900}" \
        >/tmp/websockify.log 2>&1 &
    echo "[explorer] noVNC ready: http://localhost:${NOVNC_PORT:-6080}/vnc.html?autoconnect=1&resize=scale" >&2
}

use_host_display() {
    display_number="${DISPLAY#:}"
    display_number="${display_number%%.*}"
    [ -S "/tmp/.X11-unix/X$display_number" ] ||
        die "host X11 socket /tmp/.X11-unix/X$display_number is not mounted (WSLg: run from WSL with /tmp/.X11-unix and /mnt/wslg available, or use the noVNC service explorer-gui)"
}

run_gui() {
    require_env
    [ "$QT_QPA_PLATFORM" = "xcb" ] || die "the GUI needs QT_QPA_PLATFORM=xcb (got $QT_QPA_PLATFORM)"
    require_display
    case "${UD_EXPLORER_DISPLAY_MODE:-vnc}" in
        vnc) start_vnc_display ;;
        host) use_host_display ;;
        *) die "UD_EXPLORER_DISPLAY_MODE must be vnc or host (got ${UD_EXPLORER_DISPLAY_MODE})" ;;
    esac
    $CLI probe-gl >&2 || exit $?
    app="${UD_GUI_APP:-src.tools.model_explorer.launch_model_explorer}"
    exec python -m "$app" "$@"
}

mode="${1:-engines}"
[ "$#" -eq 0 ] || shift

case "$mode" in
    engines)
        exec $CLI engines
        ;;
    probe-gl)
        require_env
        exec $CLI probe-gl
        ;;
    export)
        require_env
        exec $CLI export "$@"
        ;;
    app-screenshot)
        require_env
        case "$QT_QPA_PLATFORM" in
            offscreen) ;;
            *)
                require_display
                if [ "${UD_EXPLORER_DISPLAY_MODE:-vnc}" = "vnc" ]; then start_vnc_display; fi
                ;;
        esac
        exec $CLI app-screenshot "$@"
        ;;
    gui)
        run_gui "$@"
        ;;
    *)
        exec "$mode" "$@"
        ;;
esac
