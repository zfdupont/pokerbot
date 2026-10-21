#!/usr/bin/env bash
# scripts/pi_run.sh SESSION_ID PROMPT_FILE OUT_PREFIX [extra pi args...]
# Runs pi as a non-interactive coding subagent (SDD implementer role; see
# .mex/patterns/sdd-pi-implementer.md). Streams events live:
#   OUT_PREFIX-events.jsonl (raw), OUT_PREFIX-progress.txt (one line per tool
#   call + assistant text), OUT_PREFIX-stdout.txt (final reply).
# Reusing SESSION_ID resumes that conversation (fix rounds).
# PI_MODEL overrides the model (default deepseek/deepseek-flash).
# </dev/null is load-bearing: pi waits for stdin EOF, and a Claude Code Bash
# tool's stdin never closes.
set -o pipefail
sid=$1; prompt=$2; out=$3; shift 3
pi -p --approve --mode json --session-id "$sid" --model "${PI_MODEL:-deepseek/deepseek-flash}" --thinking high "$@" \
  @"$prompt" "Follow the instructions in the attached file." </dev/null 2>"$out-stderr.txt" \
| tee "$out-events.jsonl" \
| jq --unbuffered -r '
    if .type == "tool_execution_start" then
      "\(now|localtime|strftime("%H:%M:%S")) \(.toolName): \((.args.command // .args.path // (.args|tostring))|gsub("\n";" ")|.[0:160])"
    elif .type == "tool_execution_end" and .isError then "  ^ tool error"
    elif .type == "message_end" and .message.role == "assistant" then
      (.message.content[]? | select(.type == "text") | .text)
    else empty end' > "$out-progress.txt"
rc=$?
# Last assistant text = the report contract reply.
jq -r 'select(.type=="message_end" and .message.role=="assistant") | .message.content[]? | select(.type=="text") | .text' \
  "$out-events.jsonl" | tail -40 > "$out-stdout.txt"
exit $rc
