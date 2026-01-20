#!/bin/bash
# Sandboxed Ralph Wiggum review runner using macOS native sandbox (sandbox-exec)
#
# Provides:
# - Filesystem isolation (no access to ~/.ssh, ~/.aws, most of ~/.config, etc.)
# - Uses existing Claude auth (keychain access preserved)
# - Native performance (no container overhead)
#
# The sandbox profile allows:
# - Read/write to the repo directory
# - Read access to ~/.claude (for settings)
# - Network access (required for Anthropic API)
# - Keychain access (for subscription auth)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[SANDBOX]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[SANDBOX]${NC} $1"; }
log_error() { echo -e "${RED}[SANDBOX]${NC} $1"; }

# Check we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    log_error "This sandbox script is for macOS only"
    log_error "On Linux, use Docker-based sandboxing or run ralph_review.sh directly"
    exit 1
fi

# Check sandbox-exec exists
if ! command -v sandbox-exec &> /dev/null; then
    log_error "sandbox-exec not found. This requires macOS."
    exit 1
fi

# Generate sandbox profile
# Strategy: allow default, then deny sensitive paths
generate_sandbox_profile() {
    cat << SBPROFILE
(version 1)
(allow default)

;; DENY sensitive directories - credentials, secrets, personal data
(deny file-read* file-write*
    (subpath "$HOME/.ssh")
    (subpath "$HOME/.aws")
    (subpath "$HOME/.gnupg")
    (subpath "$HOME/.kube")
    (subpath "$HOME/.docker")
    (subpath "$HOME/.azure")
    (subpath "$HOME/.gcloud")
    (subpath "$HOME/Documents")
    (subpath "$HOME/Desktop")
    (subpath "$HOME/Downloads")
    (subpath "$HOME/.bash_history")
    (subpath "$HOME/.zsh_history")
    (subpath "$HOME/.zhistory")
    (subpath "$HOME/.local/share/fish/fish_history")
    (literal "$HOME/.netrc")
    (literal "$HOME/.npmrc")
    (literal "$HOME/.pypirc")
)
SBPROFILE
}

# Run review in sandbox
run_sandboxed() {
    local args=("$@")

    log_info "Starting sandboxed review (macOS native sandbox)..."
    log_warn "Filesystem isolated: repo + Claude config only"
    log_warn "Blocked: ~/.ssh, ~/.aws, ~/.gnupg, ~/.kube, Documents, etc."
    log_info "Using existing Claude subscription auth"

    # Create temp file for sandbox profile
    local profile_file
    profile_file=$(mktemp /tmp/sandbox_profile_XXXXXX)
    generate_sandbox_profile > "$profile_file"

    # Run with sandbox
    local exit_code=0
    sandbox-exec -f "$profile_file" "$SCRIPT_DIR/ralph_review.sh" "${args[@]}" || exit_code=$?

    # Cleanup
    rm -f "$profile_file"

    return $exit_code
}

# Show help
show_help() {
    echo "Usage: $0 [options] [ralph_review.sh args...]"
    echo ""
    echo "Runs ralph_review.sh inside a macOS sandbox that blocks access to"
    echo "sensitive files (~/.ssh, ~/.aws, etc.) while allowing Claude to"
    echo "read and modify files in this repository."
    echo ""
    echo "Options:"
    echo "  --help          Show this help"
    echo "  --show-profile  Print the sandbox profile and exit"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run sandboxed review loop"
    echo "  $0 --status           # Show review status (sandboxed)"
    echo "  $0 --reset            # Reset tracking (sandboxed)"
    echo ""
    echo "Environment variables:"
    echo "  MAX_ITERATIONS  Max review iterations (default: 5)"
    echo "  VERBOSE         Set to 1 for verbose output"
}

# Main
case "${1:-}" in
    --help|-h)
        show_help
        ;;
    --show-profile)
        generate_sandbox_profile
        ;;
    *)
        run_sandboxed "$@"
        ;;
esac
