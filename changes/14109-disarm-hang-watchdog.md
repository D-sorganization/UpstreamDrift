---
issue: 14109
summary: "Disarm the root conftest hang watchdog in pytest_unconfigure (ignore SIGUSR1, close the dump pipe, and terminate/reap the watchdog subprocess) and exit the watchdog immediately on stdin EOF so late SIGUSR1 signals during interpreter shutdown cannot kill pytest with exit code 138."
title: "Disarm Unit-Gate Hang Watchdog on Pytest_Unconfigure"
branch: "agy-c/14109-disarm-hang-watchdog"
---
