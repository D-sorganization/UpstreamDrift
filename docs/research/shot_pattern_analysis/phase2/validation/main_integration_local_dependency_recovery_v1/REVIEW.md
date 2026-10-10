# Main Integration Local Dependency Recovery

The original owned Gemini worker attempted `git submodule update --no-fetch` on an uninitialized submodule. Git still started an initial network clone. Root observed the owned process tree, stopped its clone descendants, then deliberately terminated the worker when Git retried. Session `43385` is terminal; no restart was inferred from an observation timeout. The uncommitted merge was preserved.

Root cloned from the existing local Tools checkout using `--shared --no-checkout` with `GIT_NO_LAZY_FETCH=1`, then checked out exact pin `86d0f28b1cc5acf61185e07e320c816c2d005512`. Checkout and clean-status checks passed. This changes only the isolated integration dependency, not the frozen producer or shared validation environment. It is dependency preparation, not integration acceptance.

Gemini resumed in session `46287`, PID `711803`, with explicit instructions to preserve the existing merge, use the prepared dependency, avoid initialization/network/installs and leave all results uncommitted for review. Start and exact prompt evidence are preserved here. Check this live process and `/tmp/p2-main-integration-agy-resume1` before starting another worker.
