#!/bin/zsh
# usage: arm.sh <run|judge> <tree: eval|r4> <label> [extra args]
# Exports the staging env file's keys and model settings (never printed) and
# runs the eval from the chosen checkout: eval = tickeragent-eval (r7), r4 = tickeragent-r4 (r8).
set -e
unset OPENAI_API_KEY GOOGLE_API_KEY GEMINI_API_KEY MOONSHOT_API_KEY ANTHROPIC_API_KEY
what=$1; tree=$2; label=$3; shift 3
case $tree in
  eval) root=/Users/akinfemiakinaluko/machbit/tickeragent-eval ;;
  r4) root=/Users/akinfemiakinaluko/machbit/tickeragent-r4 ;;
esac
cd $root/TradingAgents/eval
ENVF=/Users/akinfemiakinaluko/machbit/tickeragent.ai/.env.staging.host
while IFS= read -r line; do
  k=${line%%=*}; v=${line#*=}; v=${v#[\"\']}; v=${v%[\"\']}
  [[ -n $v ]] && export "$k=$v"
done < <(grep -E '^(TIINGO_API_KEY|FRED_API_KEY|ALPHA_VANTAGE_API_KEY|SEC_EDGAR_USER_AGENT|OPENROUTER_API_KEY)=' $ENVF)
unset QUALITY_FORCE_HOLD_TICKER
# OpenRouter caps new accounts at 20 requests a minute per model: retry with backoff.
export TRADINGAGENTS_LLM_MAX_RETRIES=8
export TYPESAFE_API_KEY="$OPENROUTER_API_KEY"
export TYPESAFE_BASE_URL=https://openrouter.ai/api TYPESAFE_DEFAULT_MODEL=jev-1.13 TYPESAFE_DEADLINE_SECONDS=20
if [[ $what == run ]]; then
  exec $root/server/.venv/bin/python /Users/akinfemiakinaluko/machbit/tickeragent-r4/TradingAgents/eval/tools/run_extra.py --label $label --env $ENVF "$@"
else
  exec $root/server/.venv/bin/python /Users/akinfemiakinaluko/machbit/tickeragent-r4/TradingAgents/eval/tools/judge_extra.py --label $label --env $ENVF "$@"
fi
