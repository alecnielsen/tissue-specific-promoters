#!/usr/bin/env bash
#
# Ralph Wiggum Loop: Iterative Code Review for Ctrl-DNA Scripts
#
# This script implements iterative code review where each iteration starts
# with fresh context. Claude reviews and fixes issues, then we loop until
# either all issues are resolved or max iterations reached.
#
# Usage:
#   ./ralph_review.sh                    # Run review loop
#   ./ralph_review.sh --status           # Show current tracking status
#   ./ralph_review.sh --reset            # Reset iteration counter
#
# Environment variables:
#   MAX_ITERATIONS  - Maximum iterations (default: 5)
#   VERBOSE         - Set to 1 for verbose output

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PROMPTS_DIR="$SCRIPT_DIR/prompts"
LOGS_DIR="$SCRIPT_DIR/logs"
TRACKING_FILE="$SCRIPT_DIR/tracking.yaml"
HISTORY_FILE="$LOGS_DIR/scripts_history.md"

MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
VERBOSE="${VERBOSE:-0}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    if ! command -v claude &> /dev/null; then
        log_error "claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
        exit 1
    fi

    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found"
        exit 1
    fi

    # Check for PyYAML
    if ! python3 -c "import yaml" 2>/dev/null; then
        log_error "PyYAML not installed. Install with: pip install pyyaml"
        exit 1
    fi
}

# Initialize tracking file if it doesn't exist
init_tracking() {
    if [[ ! -f "$TRACKING_FILE" ]]; then
        cat > "$TRACKING_FILE" << 'EOF'
# Ralph Wiggum Review Tracking
# Auto-generated - do not edit manually

module:
  name: scripts
  status: pending
  iterations: 0
  last_result: null
  last_reviewed: null

summary:
  total_iterations: 0
  last_run: null
EOF
        log_info "Created tracking file: $TRACKING_FILE"
    fi
}

# Read tracking YAML value
read_tracking() {
    local key="$1"
    python3 -c "
import yaml
with open('$TRACKING_FILE') as f:
    data = yaml.safe_load(f)
keys = '$key'.split('.')
val = data
for k in keys:
    val = val.get(k, '') if val else ''
print(val if val else '')
"
}

# Update tracking YAML
update_tracking() {
    local field="$1"
    local value="$2"

    python3 << EOF
import yaml
from datetime import datetime

with open('$TRACKING_FILE', 'r') as f:
    data = yaml.safe_load(f)

# Update field
if '$field' == 'iterations':
    data['module']['$field'] = int('$value')
    data['summary']['total_iterations'] = int('$value')
else:
    data['module']['$field'] = '$value'

# Update timestamp
if '$field' in ['status', 'last_result']:
    data['module']['last_reviewed'] = datetime.now().isoformat()
    data['summary']['last_run'] = datetime.now().isoformat()

with open('$TRACKING_FILE', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, sort_keys=False)
EOF
}

# Collect source code from scripts/ and Ctrl-DNA optimizer
collect_source_code() {
    local output=""
    local scripts_dir="$REPO_ROOT/scripts"

    # Collect scripts/*.py
    if [[ -d "$scripts_dir" ]]; then
        while IFS= read -r -d '' pyfile; do
            local rel_path="${pyfile#$REPO_ROOT/}"
            output+="
=== FILE: $rel_path ===
$(cat "$pyfile")

"
        done < <(find "$scripts_dir" -name "*.py" -type f -print0 | sort -z)
    fi

    # Collect Ctrl-DNA optimizer files (EDITABLE - not just context)
    local optimizer_dir="$REPO_ROOT/Ctrl-DNA/ctrl_dna/dna_optimizers_multi"
    local optimizer_files=(
        "base_optimizer.py"
        "lagrange_optimizer.py"
    )

    for file in "${optimizer_files[@]}"; do
        local filepath="$optimizer_dir/$file"
        if [[ -f "$filepath" ]]; then
            local rel_path="${filepath#$REPO_ROOT/}"
            output+="
=== FILE: $rel_path ===
$(cat "$filepath")

"
        fi
    done

    echo "$output"
}

# Run a single review iteration
run_review_iteration() {
    local iteration="$1"

    log_info "Running review iteration $iteration"

    # Get the prompt
    local prompt_file="$PROMPTS_DIR/scripts.md"
    if [[ ! -f "$prompt_file" ]]; then
        log_error "Prompt file not found: $prompt_file"
        return 1
    fi

    # Get NOTES.md context
    local notes_context=""
    if [[ -f "$REPO_ROOT/NOTES.md" ]]; then
        notes_context=$(cat "$REPO_ROOT/NOTES.md")
    fi

    # Collect source code
    local source_code
    source_code=$(collect_source_code)

    # Read the prompt
    local prompt
    prompt=$(cat "$prompt_file")

    # Check for iteration history
    local history_context=""
    if [[ -f "$HISTORY_FILE" ]]; then
        history_context=$(cat "$HISTORY_FILE")
    fi

    # Build history section if we have previous iterations
    local history_section=""
    if [[ -n "$history_context" ]]; then
        history_section="
---

# PREVIOUS ITERATION HISTORY

The following shows what previous review iterations found and fixed.
Use this to understand what has already been addressed and avoid repeating the same findings.

$history_context

---
"
    fi

    # Construct the full review request
    local full_request="
$prompt
$history_section
---

# PROJECT CONTEXT (NOTES.md)

$notes_context

---

# SOURCE CODE TO REVIEW

$source_code

---

Please review the source code above against the scientific criteria in the prompt.
Fix any issues directly in the files, then report what you changed.
If no code issues are found (or remain after fixes), respond with exactly NO_ISSUES (and nothing else).
"

    # Create a temporary file for the request
    local temp_file
    temp_file=$(mktemp)
    echo "$full_request" > "$temp_file"

    # Run claude with --print for non-interactive mode
    if [[ "$VERBOSE" == "1" ]]; then
        log_info "Sending request to Claude..."
    fi

    # Use claude --print with full permissions for automated review loop
    local result
    result=$(cd "$REPO_ROOT" && claude --print --dangerously-skip-permissions < "$temp_file" 2>&1) || {
        log_error "Claude review failed"
        rm -f "$temp_file"
        return 1
    }

    rm -f "$temp_file"

    # Log this iteration's result to history file
    mkdir -p "$LOGS_DIR"
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    {
        echo "## Iteration $iteration ($timestamp)"
        echo ""
        echo "$result"
        echo ""
        echo "---"
        echo ""
    } >> "$HISTORY_FILE"

    # Check for NO_ISSUES - robust parsing
    if echo "$result" | grep -qE '^\s*NO_ISSUES\s*$'; then
        log_success "NO_ISSUES found - review complete!"
        update_tracking "status" "clean"
        update_tracking "last_result" "NO_ISSUES"
        echo ""  # Return empty hash to signal success
        return 0
    else
        log_warning "Issues found or fixes made"
        update_tracking "status" "issues"
        update_tracking "last_result" "issues_found"

        # Print the result to stdout
        echo ""
        echo "=== Review output (iteration $iteration) ==="
        echo "$result"
        echo "============================================="
        echo ""

        # Return hash of result for stuck detection
        echo "$result" | shasum -a 256 | cut -d' ' -f1
        return 1
    fi
}

# Main review loop
run_review_loop() {
    log_info "Starting Ralph Wiggum review loop"
    log_info "Max iterations: $MAX_ITERATIONS"

    init_tracking

    # Get current iteration count
    local current_iterations
    current_iterations=$(read_tracking "module.iterations")
    current_iterations=${current_iterations:-0}

    # Check if already at max
    if [[ "$current_iterations" -ge "$MAX_ITERATIONS" ]]; then
        log_warning "Already at max iterations ($MAX_ITERATIONS). Use --reset to start over."
        return 0
    fi

    # Update status to in_progress
    update_tracking "status" "in_progress"

    # Stuck detection
    local prev_issue_hash=""
    local stuck_count=0
    local max_stuck=2

    # Loop until clean or max iterations
    while [[ "$current_iterations" -lt "$MAX_ITERATIONS" ]]; do
        local iteration=$((current_iterations + 1))
        update_tracking "iterations" "$iteration"

        log_info "=== Iteration $iteration/$MAX_ITERATIONS ==="

        # Run review and capture output
        local output
        local exit_code
        output=$(run_review_iteration "$iteration")
        exit_code=$?

        if [[ "$exit_code" -eq 0 ]]; then
            # Clean - we're done
            log_success "Review passed at iteration $iteration"
            return 0
        fi

        # Issues found - check for stuck condition
        local issue_hash
        issue_hash=$(echo "$output" | tail -1)

        if [[ "$issue_hash" == "$prev_issue_hash" ]]; then
            ((stuck_count++))
            log_warning "Same output detected ($stuck_count/$max_stuck)"

            if [[ "$stuck_count" -ge "$max_stuck" ]]; then
                log_error "Review appears stuck - same output $max_stuck times in a row"
                log_error "Human intervention may be needed"
                update_tracking "status" "stuck"
                return 1
            fi
        else
            stuck_count=0
        fi

        prev_issue_hash="$issue_hash"
        current_iterations=$iteration

        # Brief pause between iterations
        sleep 2
    done

    log_warning "Reached max iterations ($MAX_ITERATIONS) with issues remaining"
    return 1
}

# Show current status
show_status() {
    echo ""
    echo "=== Ralph Wiggum Review Status ==="
    echo ""

    if [[ ! -f "$TRACKING_FILE" ]]; then
        echo "No tracking file found. Run review first."
        return
    fi

    python3 - "$TRACKING_FILE" << 'PYEOF'
import sys
import yaml

with open(sys.argv[1], 'r') as f:
    data = yaml.safe_load(f)

module = data.get('module', {})
status = module.get('status', 'unknown')
iterations = module.get('iterations', 0)
last_result = module.get('last_result', 'none')
last_reviewed = module.get('last_reviewed', 'never')

if last_reviewed and last_reviewed != 'null':
    last_reviewed = str(last_reviewed)[:19]
else:
    last_reviewed = 'never'

status_emoji = {
    'pending': '[ ]',
    'in_progress': '[~]',
    'clean': '[+]',
    'issues': '[!]',
    'stuck': '[X]'
}.get(status, '[?]')

print(f"Module: scripts")
print(f"  Status: {status_emoji} {status}")
print(f"  Iterations: {iterations}")
print(f"  Last result: {last_result}")
print(f"  Last reviewed: {last_reviewed}")
PYEOF

    echo ""
    if [[ -f "$HISTORY_FILE" ]]; then
        echo "History log: $HISTORY_FILE"
        echo "  $(wc -l < "$HISTORY_FILE") lines"
    fi
}

# Reset tracking
reset_tracking() {
    log_info "Resetting tracking..."
    rm -f "$TRACKING_FILE"
    rm -f "$HISTORY_FILE"
    init_tracking
    log_success "Tracking reset complete"
}

# Main entry point
main() {
    check_dependencies

    case "${1:-}" in
        --status|-s)
            show_status
            ;;
        --reset|-r)
            reset_tracking
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  (none)     Run the review loop"
            echo "  --status   Show current tracking status"
            echo "  --reset    Reset iteration counter and history"
            echo "  --help     Show this help"
            echo ""
            echo "Environment variables:"
            echo "  MAX_ITERATIONS  Max iterations (default: 5)"
            echo "  VERBOSE         Set to 1 for verbose output"
            ;;
        *)
            run_review_loop
            ;;
    esac
}

main "$@"
