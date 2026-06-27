#!/usr/bin/env bash
# Pull and verify the recommended Ollama model for PrivateRAG.
set -euo pipefail

MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"
BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"

echo "==> Checking Ollama..."
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Get it from https://ollama.com/download"
  exit 1
fi

if ! curl -sf "${BASE_URL}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is not running. Start the Ollama app (macOS menu bar) or run: ollama serve"
  exit 1
fi

echo "==> Pulling model: ${MODEL}"
ollama pull "${MODEL}"

echo "==> Smoke test..."
curl -sf "${BASE_URL}/api/chat" -d "{
  \"model\": \"${MODEL}\",
  \"stream\": false,
  \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: OK\"}]
}" | python3 -c "import sys,json; print('Response:', json.load(sys.stdin)['message']['content'][:80])"

echo ""
echo "Done. Set in .env:"
echo "  OLLAMA_MODEL=${MODEL}"
echo "  LLM_PROVIDER=ollama   # or keep auto (uses Ollama when cloud keys fail)"
echo ""
echo "Run eval locally:"
echo "  python scripts/run_eval.py --dataset eval/dataset.ec2.json --mode ollama --no-rewrite-llm --sleep-seconds 2"
