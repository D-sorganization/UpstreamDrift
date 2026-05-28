with open(".github/workflows/anti-phantom-merge.yml") as f:
    content = f.read()

# Ah! The failure from the CI log:
# anti-phantom-merge.yml::guard: runs-on 'ubuntu-latest' is a hosted runner; route to d-sorg-fleet
# My previous patch changed it to ubuntu-latest, but the internal policy requires d-sorg-fleet, and my patch actually caused the policy violation.
# BUT wait, the first CI failure said:
# The self-hosted runner lost communication with the server. Verify the machine is running...
# That is an infrastructure problem on their side.
# I shouldn't have changed `d-sorg-fleet` to `ubuntu-latest`. I'll revert my commit.
