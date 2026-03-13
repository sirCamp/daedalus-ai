#!/usr/bin/env bash
# Daedalus Sentinel — remote experiment monitor
#
# Deployed automatically by SSHRunner. Runs inside screen/tmux.
# Monitors a training process, tails logs, writes structured events.
#
# Usage: sentinel.sh <work_dir> <pid> [poll_interval] [log_interval]
#
# Writes to: <work_dir>/events.jsonl
# Reads from: <work_dir>/logs/stdout.log, stderr.log (BOTH are parsed)

set -euo pipefail

WORK_DIR="${1:?Usage: sentinel.sh <work_dir> <pid> [poll_interval] [log_interval]}"
TRAIN_PID="${2:?Usage: sentinel.sh <work_dir> <pid> [poll_interval] [log_interval]}"
POLL_INTERVAL="${3:-30}"
LOG_INTERVAL="${4:-10}"

EVENTS_FILE="$WORK_DIR/events.jsonl"
STDOUT_LOG="$WORK_DIR/logs/stdout.log"
STDERR_LOG="$WORK_DIR/logs/stderr.log"
SENTINEL_PID_FILE="$WORK_DIR/sentinel.pid"

# Track log file positions (both stdout and stderr)
STDOUT_POS=0
STDERR_POS=0

# Track last loss values for spike detection
LOSS_VALUES=()
LAST_EPOCH=""
LAST_STEP=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

emit_event() {
    local event="$1"
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    # Build JSON manually (no jq dependency)
    echo "{\"event\":\"$event\",\"timestamp\":\"$ts\"${2:+,$2}}" >> "$EVENTS_FILE"
}

is_alive() {
    kill -0 "$TRAIN_PID" 2>/dev/null
}

# Get exit code of a PID via /proc or wait.
# Returns 0 for success, non-zero for failure, 127 if unknown.
get_exit_code() {
    # Try /proc (Linux)
    if [ -f "/proc/$TRAIN_PID/status" ]; then
        return 127  # still alive or can't determine
    fi
    # Try wait (works if we're parent or process is zombie)
    wait "$TRAIN_PID" 2>/dev/null
    return $?
}

# ---------------------------------------------------------------------------
# Log parsing (same patterns as Python LogMonitor)
# Parses BOTH stdout and stderr — HF Trainer, tqdm, and most ML
# frameworks write progress/metrics to stderr.
# ---------------------------------------------------------------------------

parse_log_lines() {
    local lines="$1"
    [ -z "$lines" ] && return

    while IFS= read -r line; do
        [ -z "$line" ] && continue

        # HF Trainer dict: {'loss': 0.345, ..., 'epoch': 1.5}
        if [[ "$line" =~ [\{\']loss[\'\"]:[[:space:]]*([0-9.e+-]+).*[\'\"]epoch[\'\"]:[[:space:]]*([0-9.]+) ]]; then
            local loss="${BASH_REMATCH[1]}"
            local epoch="${BASH_REMATCH[2]}"
            emit_event "METRIC" "\"metric\":\"loss\",\"value\":$loss"
            emit_event "PROGRESS" "\"key\":\"epoch\",\"value\":\"$epoch\""
            track_loss "$loss"
            continue
        fi

        # eval_loss (must check before generic loss)
        if [[ "$line" =~ eval[_/]loss[[:space:]=:]+([0-9.]+[eE]?[+-]?[0-9]*) ]]; then
            emit_event "METRIC" "\"metric\":\"eval_loss\",\"value\":${BASH_REMATCH[1]}"
            continue
        fi

        # Generic loss
        if [[ "$line" =~ loss[[:space:]=:]+([0-9.]+[eE]?[+-]?[0-9]*) ]]; then
            local loss="${BASH_REMATCH[1]}"
            emit_event "METRIC" "\"metric\":\"loss\",\"value\":$loss"
            track_loss "$loss"
            continue
        fi

        # Accuracy / token_accuracy
        if [[ "$line" =~ [Aa]cc(uracy)?[[:space:]=:]+([0-9.]+[eE]?[+-]?[0-9]*) ]]; then
            emit_event "METRIC" "\"metric\":\"accuracy\",\"value\":${BASH_REMATCH[2]}"
            continue
        fi

        # Learning rate
        if [[ "$line" =~ (learning_rate|lr)[[:space:]=:]+([0-9.]+[eE]?[+-]?[0-9]*) ]]; then
            emit_event "METRIC" "\"metric\":\"learning_rate\",\"value\":${BASH_REMATCH[2]}"
            continue
        fi

        # tqdm progress bar: 42%|████▏     | 52/125 [00:49<01:09,  1.05it/s]
        if [[ "$line" =~ ([0-9]+)%\|.*\|[[:space:]]*([0-9]+)/([0-9]+) ]]; then
            local pct="${BASH_REMATCH[1]}"
            local current="${BASH_REMATCH[2]}"
            local total="${BASH_REMATCH[3]}"
            emit_event "PROGRESS" "\"key\":\"step\",\"value\":\"$current/$total ($pct%)\""
            LAST_STEP="$current/$total"
            continue
        fi

        # Epoch progress: Epoch 3/10
        if [[ "$line" =~ [Ee]poch[[:space:]:]+([0-9]+)(/([0-9]+))? ]]; then
            local ep="${BASH_REMATCH[1]}/${BASH_REMATCH[3]:-?}"
            if [ "$ep" != "$LAST_EPOCH" ]; then
                LAST_EPOCH="$ep"
                emit_event "PROGRESS" "\"key\":\"epoch\",\"value\":\"$ep\""
            fi
            continue
        fi

        # Step progress: Step 500/5000
        if [[ "$line" =~ [Ss]tep[[:space:]:]+([0-9]+)(/([0-9]+))? ]]; then
            local st="${BASH_REMATCH[1]}/${BASH_REMATCH[3]:-?}"
            if [ "$st" != "$LAST_STEP" ]; then
                LAST_STEP="$st"
                emit_event "PROGRESS" "\"key\":\"step\",\"value\":\"$st\""
            fi
            continue
        fi

        # NaN detection
        if [[ "$line" =~ [Nn][Aa][Nn] ]]; then
            emit_event "ALERT" "\"alert_type\":\"nan\",\"detail\":\"NaN detected in output\""
            continue
        fi

        # OOM detection
        if [[ "$line" =~ (CUDA[[:space:]]out[[:space:]]of[[:space:]]memory|OutOfMemoryError|OOM) ]]; then
            emit_event "ALERT" "\"alert_type\":\"oom\",\"detail\":\"Out of memory\""
            continue
        fi

        # Generic errors
        if [[ "$line" =~ (RuntimeError|ValueError|KeyError|FileNotFoundError|ModuleNotFoundError):[[:space:]]*(.*) ]]; then
            local err_type="${BASH_REMATCH[1]}"
            local err_msg="${BASH_REMATCH[2]}"
            # Escape quotes in error message
            err_msg="${err_msg//\"/\\\"}"
            emit_event "ALERT" "\"alert_type\":\"error\",\"detail\":\"$err_type: $err_msg\""
            continue
        fi

    done <<< "$lines"
}

track_loss() {
    local loss="$1"
    LOSS_VALUES+=("$loss")

    # Keep last 10
    if [ ${#LOSS_VALUES[@]} -gt 10 ]; then
        LOSS_VALUES=("${LOSS_VALUES[@]:(-10)}")
    fi

    # Spike detection (need at least 4 values)
    if [ ${#LOSS_VALUES[@]} -ge 4 ]; then
        # Compare last value to average of previous values using awk
        local spike
        spike=$(printf '%s\n' "${LOSS_VALUES[@]}" | awk '
        {
            values[NR] = $1
            n = NR
        }
        END {
            if (n < 4) exit
            sum = 0
            for (i = 1; i < n; i++) sum += values[i]
            avg = sum / (n - 1)
            if (avg > 0 && values[n] / avg > 5.0) {
                printf "Loss jumped %.1fx (avg %.4f -> %.4f)", values[n] / avg, avg, values[n]
            }
        }')
        if [ -n "$spike" ]; then
            emit_event "ALERT" "\"alert_type\":\"loss_spike\",\"detail\":\"$spike\""
        fi
    fi
}

read_new_stdout() {
    [ ! -f "$STDOUT_LOG" ] && return
    local file_size
    file_size=$(wc -c < "$STDOUT_LOG" 2>/dev/null || echo 0)
    file_size=$((file_size + 0))  # ensure numeric

    if [ "$file_size" -gt "$STDOUT_POS" ]; then
        local new_content
        new_content=$(dd if="$STDOUT_LOG" bs=1 skip="$STDOUT_POS" 2>/dev/null || true)
        STDOUT_POS=$file_size
        if [ -n "$new_content" ]; then
            parse_log_lines "$new_content"
        fi
    fi
}

read_new_stderr() {
    [ ! -f "$STDERR_LOG" ] && return
    local file_size
    file_size=$(wc -c < "$STDERR_LOG" 2>/dev/null || echo 0)
    file_size=$((file_size + 0))

    if [ "$file_size" -gt "$STDERR_POS" ]; then
        local new_content
        new_content=$(dd if="$STDERR_LOG" bs=1 skip="$STDERR_POS" 2>/dev/null || true)
        STDERR_POS=$file_size
        if [ -n "$new_content" ]; then
            parse_log_lines "$new_content"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Completion detection
# ---------------------------------------------------------------------------

check_completion() {
    # 1. Check for explicit result files
    if [ -f "$WORK_DIR/results.json" ] || \
       [ -f "$WORK_DIR/eval_results.json" ] || \
       [ -n "$(find "$WORK_DIR" -name 'eval_results.json' -print -quit 2>/dev/null)" ]; then
        return 0  # completed with results
    fi

    # 2. Check for HF Trainer checkpoints (trainer_state.json)
    if [ -n "$(find "$WORK_DIR" -name 'trainer_state.json' -print -quit 2>/dev/null)" ]; then
        return 0
    fi

    # 3. Check for saved model artifacts (common patterns)
    if [ -f "$WORK_DIR/pytorch_model.bin" ] || \
       [ -f "$WORK_DIR/model.safetensors" ] || \
       [ -n "$(find "$WORK_DIR" -maxdepth 3 -name 'model.safetensors' -print -quit 2>/dev/null)" ] || \
       [ -n "$(find "$WORK_DIR" -maxdepth 3 -name 'adapter_model.safetensors' -print -quit 2>/dev/null)" ]; then
        return 0
    fi

    # 4. Check process exit code (if available)
    # wait works if the sentinel is in the same session
    if wait "$TRAIN_PID" 2>/dev/null; then
        return 0  # exit code 0 = success
    fi

    # 5. Heuristic: check stdout/stderr for completion signals
    if [ -f "$STDOUT_LOG" ]; then
        local tail_out
        tail_out=$(tail -10 "$STDOUT_LOG" 2>/dev/null || true)
        if [[ "$tail_out" =~ (Training complete|Saving model|Model saved|Best model) ]]; then
            return 0
        fi
    fi
    if [ -f "$STDERR_LOG" ]; then
        local tail_err
        tail_err=$(tail -10 "$STDERR_LOG" 2>/dev/null || true)
        if [[ "$tail_err" =~ (Training complete|Saving model|Model saved|100%\|) ]]; then
            return 0
        fi
    fi

    return 1  # not completed
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

main() {
    # Write sentinel PID
    echo $$ > "$SENTINEL_PID_FILE"

    # Initialize events file
    emit_event "SENTINEL_START" "\"pid\":$TRAIN_PID,\"poll_interval\":$POLL_INTERVAL,\"log_interval\":$LOG_INTERVAL"

    local last_poll_time=0
    local last_heartbeat_time=0
    local stall_count=0
    local prev_log_activity=0

    while true; do
        local now
        now=$(date +%s)

        # --- Tail logs (every LOG_INTERVAL seconds) ---
        # Parse BOTH stdout and stderr
        local prev_stdout=$STDOUT_POS
        local prev_stderr=$STDERR_POS
        read_new_stdout
        read_new_stderr

        # Track combined log activity for stall detection
        local current_activity=$((STDOUT_POS + STDERR_POS))
        if [ "$current_activity" -eq "$prev_log_activity" ]; then
            stall_count=$((stall_count + 1))
            local stall_seconds=$((stall_count * LOG_INTERVAL))
            if [ $((stall_seconds % 1800)) -eq 0 ] && [ "$stall_seconds" -gt 0 ]; then
                emit_event "ALERT" "\"alert_type\":\"stall\",\"detail\":\"No log output for $((stall_seconds / 60)) minutes\""
            fi
        else
            stall_count=0
            prev_log_activity=$current_activity
        fi

        # --- Poll process (every POLL_INTERVAL seconds) ---
        if [ $((now - last_poll_time)) -ge "$POLL_INTERVAL" ]; then
            last_poll_time=$now

            if ! is_alive; then
                # Process ended — read any remaining log output
                read_new_stdout
                read_new_stderr

                # Determine if completed or failed
                if check_completion; then
                    emit_event "COMPLETED" "\"pid\":$TRAIN_PID"
                    echo '{"state":"completed","pid":'"$TRAIN_PID"'}' > "$WORK_DIR/status.json"
                else
                    local last_err=""
                    if [ -f "$STDERR_LOG" ]; then
                        last_err=$(tail -3 "$STDERR_LOG" 2>/dev/null | tr '\n' ' ' | sed 's/"/\\"/g' || true)
                    fi
                    emit_event "FAILED" "\"pid\":$TRAIN_PID,\"error\":\"$last_err\""
                    echo '{"state":"failed","pid":'"$TRAIN_PID"'}' > "$WORK_DIR/status.json"
                fi

                emit_event "SENTINEL_STOP" "\"reason\":\"process_exited\""
                exit 0
            fi

            # Process alive — emit heartbeat every 10 polls
            if [ $((now - last_heartbeat_time)) -ge $((POLL_INTERVAL * 10)) ]; then
                last_heartbeat_time=$now
                emit_event "HEARTBEAT" "\"pid\":$TRAIN_PID"
            fi
        fi

        sleep "$LOG_INTERVAL"
    done
}

# Cleanup on signal
cleanup() {
    emit_event "SENTINEL_STOP" "\"reason\":\"signal\""
    exit 0
}
trap cleanup SIGTERM SIGINT

main
