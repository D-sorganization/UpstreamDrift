---
issue: 12055
summary: "Native prediction verifies complete live/source/model/handle/callback identity after bundle preparation before stepping, closing a late handle-mutation gap, and the execution validator reuses that verification, cutting repeated library-byte reads by 25 percent in controlled short native fixtures; full matching and parity acceptance remain open"
---

Native prediction verifies complete live/source/model/handle/callback identity
after bundle preparation before stepping, closing a late handle-mutation gap.
The existing execution validator reuses that provider verification, reducing
repeated library-byte reads by 25 percent in controlled short native fixtures.
Full matching and parity acceptance remain open; see chapter 38 and F09p turnover.
