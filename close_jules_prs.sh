#!/bin/bash
# Close all Jules-generated PRs as they're superseded by consolidated PR #624

# Find repository root dynamically
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
    echo "Error: Must be run from within the repository." >&2
    exit 1
fi
cd "$REPO_ROOT"

echo "Closing Jules PRs in Golf Suite (superseded by #624)..."

# Get all Jules PRs and close them
count=0
for pr in 717 716 715 714 713 712 711 710 709 708 707 706 705 704 703 702 701 700 699 698 697 696 695 694 693 692 691 690 689 688 687 686 685 684 683 682 681 680 679 678 677 676 675 674 673 672 671 670 669 668; do
    echo "Closing PR #$pr..."
    gh pr close $pr --comment "Superseded by consolidated workflow standardization PR #624. All fixes have been incorporated into the comprehensive PR." --delete-branch 2>/dev/null
    count=$((count + 1))
    
    # Add delay to avoid rate limiting
    if [ $((count % 10)) -eq 0 ]; then
        echo "  Pausing to avoid rate limits..."
        sleep 2
    fi
done

echo "Closed $count Jules PRs in Golf Suite"
