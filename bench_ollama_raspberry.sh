#!/usr/bin/env bash
set -euo pipefail

# Force CPU-only unless you explicitly want GPU
export OLLAMA_NUM_GPU="${OLLAMA_NUM_GPU:-0}"
HOST="${HOST:-http://127.0.0.1:11434}"

# ----------------- Config -----------------
MODELS=("nano-125m-q5" "nano-125m-f16" "nano-260m-q5" "nano-260m-f16")
THREADS=(2 4)
CTX=(512 1024 2048 4096)
PRED=(32 64 128)
REPS=${REPS:-3}
COOLDOWN=${COOLDOWN:-2}

SOC_LABEL="${SOC_LABEL:-raspberry-pi}"   # tag device
OS_REL="$(grep -m1 PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '"' || true)"
HOSTNAME="$(hostname)"

# Long Romanian prompt (sliced by context)
LONG_PROMPT="Intr-o seara ma intorceam de la munca si am vazut pe strada un catel ud care se adapostea sub un chiosc inchis. Ploua marunt, iar luminile din vitrine se reflectau in balti, facand orasul sa para un film vechi. M-am oprit, am scos din rucsac o punga goala si am rupt un colt de paine pe care il pastrasem de la pranz. Catelul a ezitat, apoi s-a apropiat cu coada intre picioare, dar ochii ii scanteiau de curiozitate. In departare se auzea tramvaiul, iar pe trotuar treceau oameni cu pas grabit, neobservand micul spectacol al increderii care incerca sa se nasca intre doi straini. Mi-am pus haina pe umeri ca pe o manta improvizata si m-am aplecat. In clipa aceea, cineva a strigat din spatele meu ca ar fi mai bine sa nu ating animalul, ca poate musca. Dar vocea aceea parea mai mult o teama veche decat un avertisment. Catelul a luat firimiturile si s-a asezat la un pas, urmarindu-ma cu atentie. I-am intins palma goala, iar el, dupa cateva clipe, a mirosit-o si a tresarit, ca si cum si-ar fi amintit ca lumea poate fi blanda. Am zambit singur si mi-am dat seama ca nu eram grabit nicaieri, desi ceasul trecuse de noua. Am pornit incet spre casa, iar el m-a urmat la distanta, oprindu-se la fiecare colt ca sa se asigure ca nu il chem intr-o capcana. Cand am ajuns la bloc, s-a asezat la scari, privind spre geamurile intunecate ca si cum ar fi citit povesti nespuse. I-am lasat un castron improvizat din capacul cutiei mele de pranz si am turnat apa din sticla. In camera mea modesta, am aprins o veioza si m-am gandit la drumul pe care il parcurge uneori increderea: incepe cu o ezitare, continua cu o firimitura, se leaga de un pas comun si se odihneste langa o usa care poate candva se va deschide."

prompt_for_ctx () {
  local ctx="$1" n_chars
  case "$ctx" in
    512)  n_chars=220  ;;
    1024) n_chars=520  ;;
    2048) n_chars=1200 ;;
    4096) n_chars=2600 ;;
    *)    n_chars=400  ;;
  esac
  printf "%s" "${LONG_PROMPT:0:n_chars}"
}

# ----------------- CSV output -----------------
OUT="${HOME}/ollama_pi_results.csv"
if [[ ! -f "$OUT" ]]; then
  echo "ts,host,soc,os,model,rep,num_thread,num_ctx,num_predict,prompt_tokens,gen_tokens,prompt_sec,gen_sec,ttft_sec,total_sec,prompt_tps,gen_tps,proc_rss_mb,proc_hwm_mb,sys_mem_avail_mb,avg_cpu_temp_c" > "$OUT"
fi

# ----------------- Dependencies -----------------
command -v jq >/dev/null || { echo "jq missing: sudo apt-get install jq"; exit 1; }
pgrep -f "ollama serve" >/dev/null || { echo "Starting ollama serve..."; nohup ollama serve > "${HOME}/ollama.log" 2>&1 & sleep 2; }

has_vcgencmd=0
if command -v vcgencmd >/dev/null; then has_vcgencmd=1; fi

# ----------------- Helpers -----------------
get_mem_stats () {
  local pid
  pid=$(pgrep -f "ollama serve" | head -n1 || true)
  if [[ -n "${pid}" && -r "/proc/$pid/status" ]]; then
    local rss hwm
    rss=$(awk '/VmRSS/{print int($2/1024)}' "/proc/$pid/status" 2>/dev/null || echo 0)
    hwm=$(awk '/VmHWM/{print int($2/1024)}' "/proc/$pid/status" 2>/dev/null || echo 0)
    echo "${rss},${hwm}"
  else
    echo "NA,NA"
  fi
}

get_sys_mem_avail_mb () {
  awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo "NA"
}

run_tempmon_bg () {
  # Poll CPU temp every second into a logfile; echo PID
  local logfile="$1"
  : > "$logfile"
  if [[ $has_vcgencmd -eq 1 ]]; then
    ( while true; do
        vcgencmd measure_temp 2>/dev/null | awk -F"=|'" '{print $2}' >> "$logfile"
        sleep 1
      done ) &
  else
    # Fallback: /sys thermal
    local path=""
    for p in /sys/class/thermal/thermal_zone*/temp; do
      if [[ -r "$p" ]]; then path="$p"; break; fi
    done
    ( while true; do
        if [[ -n "$path" ]]; then
          v=$(cat "$path" 2>/dev/null || echo "")
          if [[ -n "$v" ]]; then
            if [[ "$v" =~ ^[0-9]+$ ]]; then awk -v t="$v" 'BEGIN{printf("%.1f\n", t/1000.0)}' >> "$logfile"
            else echo "$v" >> "$logfile"
            fi
          fi
        fi
        sleep 1
      done ) &
  fi
  echo $!
}

avg_cpu_temp_from_log () {
  local file="$1"
  awk '
    $0 ~ /^[0-9.]+$/ {sum+=$1; n++}
    END { if(n>0) printf("%.1f\n", sum/n); else print "" }
  ' "$file"
}

# ----------------- Warmup -----------------
for model in "${MODELS[@]}"; do
  curl -s "$HOST/api/generate" -X POST -H "Content-Type: application/json" \
    -d '{"model":"'"$model"'","prompt":"warmup","stream":false,"options":{"num_ctx":512,"num_predict":8,"num_thread":2}}' >/dev/null || true
  curl -s "$HOST/api/generate" -X POST -H "Content-Type: application/json" \
    -d '{"model":"'"$model"'","prompt":"warmup lung","stream":false,"options":{"num_ctx":4096,"num_predict":8,"num_thread":2}}' >/dev/null || true
done

# ----------------- Main loop -----------------
for model in "${MODELS[@]}"; do
  for th in "${THREADS[@]}"; do
    for ctx in "${CTX[@]}"; do
      for pred in "${PRED[@]}"; do
        P="$(prompt_for_ctx "$ctx")"

        for rep in $(seq 1 "$REPS"); do
          # Start CPU temp monitor (best effort)
          TMPLOG=$(mktemp /tmp/pi_temps_XXXXXX.txt)
          TMPPID=$(run_tempmon_bg "$TMPLOG")

          # STREAMED run for accurate TTFT
          t0=$(date +%s.%N)
          first_ts=""
          RESP_TMP=$(mktemp /tmp/ollama_resp_XXXXXX.jsonl)

          curl -sN "$HOST/api/generate" -X POST -H "Content-Type: application/json" \
            -d '{
              "model":"'"$model"'",
              "prompt":"'"$P"'",
              "stream":true,
              "options":{"num_ctx":'"$ctx"',"num_predict":'"$pred"',"num_thread":'"$th"',"temperature":0.7,"top_p":0.9}
            }' | while IFS= read -r line; do
                  [[ -z "$line" ]] && continue
                  echo "$line" >> "$RESP_TMP"
                  if [[ -z "$first_ts" ]]; then
                    first_ts=$(date +%s.%N)
                  fi
               done

          # Stop temp monitor
          kill "$TMPPID" >/dev/null 2>&1 || true
          sleep 0.1

          # Parse final aggregate
          FINAL_JSON=$(tac "$RESP_TMP" | grep -m1 '"done":true' || true)
          if [[ -z "$FINAL_JSON" ]]; then
            FINAL_JSON=$(tail -n1 "$RESP_TMP" || echo "{}")
          fi

          prompt_count=$(echo "$FINAL_JSON" | jq -r '.prompt_eval_count // 0')
          eval_count=$(echo "$FINAL_JSON"   | jq -r '.eval_count // 0')
          p_dur=$(echo "$FINAL_JSON"        | jq -r '.prompt_eval_duration // 0')
          e_dur=$(echo "$FINAL_JSON"        | jq -r '.eval_duration // 0')
          t_dur=$(echo "$FINAL_JSON"        | jq -r '.total_duration // 0')

          prompt_sec=$(awk -v n=$p_dur 'BEGIN{printf("%.4f", n/1e9)}')
          gen_sec=$(awk -v n=$e_dur 'BEGIN{printf("%.4f", n/1e9)}')
          total_sec=$(awk -v n=$t_dur 'BEGIN{printf("%.4f", n/1e9)}')

          if [[ -n "$first_ts" ]]; then
            ttft_sec=$(awk -v a="$t0" -v b="$first_ts" 'BEGIN{printf("%.4f", b-a)}')
          else
            ttft_sec="$prompt_sec"
          fi

          prompt_tps=$(awk -v c=$prompt_count -v s=$prompt_sec 'BEGIN{if(s>0) printf("%.2f", c/s); else print 0}')
          gen_tps=$(awk -v c=$eval_count   -v s=$gen_sec    'BEGIN{if(s>0) printf("%.2f", c/s); else print 0}')

          # Memory (process + system)
          read -r mem_rss_mb mem_hwm_mb <<< "$(get_mem_stats | tr ',' ' ')"
          sys_mem_avail_mb="$(get_sys_mem_avail_mb)"

          avg_cpu_temp_c="$(avg_cpu_temp_from_log "$TMPLOG")"

          echo "$(date +%F_%T),$HOSTNAME,$SOC_LABEL,$OS_REL,$model,$rep,$th,$ctx,$pred,$prompt_count,$eval_count,$prompt_sec,$gen_sec,$ttft_sec,$total_sec,$prompt_tps,$gen_tps,$mem_rss_mb,$mem_hwm_mb,$sys_mem_avail_mb,$avg_cpu_temp_c" >> "$OUT"

          rm -f "$RESP_TMP" "$TMPLOG"
          sleep "${COOLDOWN}"
        done
      done
    done
  done
done

echo "Results saved to $OUT"
