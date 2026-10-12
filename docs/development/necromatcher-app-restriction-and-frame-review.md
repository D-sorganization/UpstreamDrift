# Necromatcher Restricted Seed and Frame Review Procedure

## Scope and Ownership

This app boundary progresses #11450 and #11414 under epic #11232 and PR #11359.
Tiger matching uses reviewed original source indices `[0,191)`; the uncertain
right-hand release and subsequent follow-through are excluded. The cutoff is
conservative and does not establish a measured physical contact time. Hogan's
reviewed interval remains `[0,750)`.

## Create a Lossless Restricted Seed

1. Recall the source fit and review its saved curve, source window, full recipe,
   contacts and shaft evidence.
2. Import or inherit an authenticated reviewed window. Select **Lossless
   Restricted Seed** in either host; this explicitly sends
   `restrict_initialization` with `restricted_spline` and strict initialization.
3. Enter both retained endpoint source indices, the training sample indices and
   the retained knot count. The original prior and knot clock remain authoritative.
   Editable frame/knot choices survive mode switching and saved-spline roundtrips.
4. Submit a new version. Canonical queue admission, worker verification and
   storage rederive the restriction and reject incompatible bounds, endpoints,
   contacts or shaft rows. The app does not trim those hypotheses automatically.
5. Inspect the seed's explicit unoptimized caption and source lineage before an
   independently requested optimization. Seed creation calls the existing
   initializer and does not run the optimizer.

## Review the Exact Saved Domain

The React page authenticates the existing fit-summary endpoint before requesting
any fitted projection. Its ordered `frame_indices` are the overlay domain, rather
than the complete capture length. Native review obtains the same domain from
canonical `load_fit` in the existing background load.

Both sliders map ordinal positions to exact original source indices. For a sparse
fit `[10,75,190]`, slider positions `[0,1,2]` request those three source images.
An excluded recalled React index selects the first retained frame. An unavailable
or malformed summary prevents fitted projection. Source-only browsing retains
the full original capture, including Tiger frames 191–209, without a fitted overlay.

## Validation and Current Limits

Root validation passed 45 Python and 45 React cases across the final combined
controls/domain cohort. Meaningful baseline failures reproduced HTTP and dialog
mode rejection, edited-domain reset and excluded/sparse frame projection. Agent
checks passed TypeScript, ESLint, configured mypy, Ruff and formatting. Temporary
Qt fixtures and mocked HTTP/UI projections establish software behavior; they
are not new historical native runs or reconstructed-motion validation.

The three independent app freezes are retained in external Temp receipts:
`restriction-app-boundary-freeze-v1.json`,
`restriction-app-boundary-freeze-v2.json` and
`necromatcher-frame-domain-freeze-v1/freeze.json`.
Root logs are `necromatcher-app-controls-root-green-v1.log` and
`necromatcher-app-controls-root-ui-green-v1.log`.

The accepted earlier 21-page Methods V8 report and historical V22 artifacts retain
their original producer and qualification records. This software change neither
updates those numerical results nor establishes physical timing, anatomy, contact,
continuous ROM, dynamics or a successful match. GitHub has registered no checks
for published head `3bd9aea`; CI remains unverified and the goal remains active.
