#!/usr/bin/env bash
set -euo pipefail

# ---------- Config ----------
HOST="http://127.0.0.1:11434"   # hard-wire to avoid empty $HOST
MODELS=("nano-125m-q5" "nano-125m-f16" "nano-260m-q5" "nano-260m-f16")
THREADS=(2 4)
CTX=(512 1024 2048 4096)
PRED=(32 64 128)
REPS=${REPS:-3}
COOLDOWN=${COOLDOWN:-2}
CSV="${HOME}/ollama_pi_results.csv"

SOC_LABEL="${SOC_LABEL:-raspberry-pi}"
OS_REL="$(grep -m1 PRETTY_NAME /etc/os-release | cut -d= -f2- | tr -d '"' || true)"
HOSTNAME="$(hostname)"

export OLLAMA_NUM_GPU="${OLLAMA_NUM_GPU:-0}"
export LC_ALL=C

# ---------- Dependencies ----------
command -v jq >/dev/null   || { echo "jq missing: sudo apt-get install -y jq"; exit 1; }
command -v curl >/dev/null || { echo "curl missing"; exit 1; }
command -v ollama >/dev/null || { echo "ollama missing in PATH"; exit 1; }
command -v awk  >/dev/null || { echo "awk missing"; exit 1; }
command -v date >/dev/null || { echo "date missing"; exit 1; }

# ---------- CSV ----------
if [[ ! -f "$CSV" ]]; then
  echo "ts,host,soc,os,model,rep,num_thread,num_ctx,num_predict,prompt_tokens,gen_tokens,prompt_sec,gen_sec,ttft_sec,total_sec,prompt_tps,gen_tps,proc_rss_mb,proc_hwm_mb,sys_mem_avail_mb,cpu_temp_c" > "$CSV"
fi
echo "[INFO] CSV output: $CSV"

# ---------- Ensure server ----------
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  echo "[INFO] Starting ollama serve…"
  nohup ollama serve -a 127.0.0.1:11434 >/dev/null 2>&1 &
fi
echo "[INFO] Waiting for Ollama to become ready…"
for i in $(seq 1 60); do
  if curl -sS --connect-timeout 2 --max-time 2 "$HOST/api/tags" >/dev/null; then
    echo "[INFO] Ollama is ready."
    break
  fi
  sleep 1
  [[ $i -eq 60 ]] && { echo "[ERROR] Ollama did not become ready."; exit 1; }
done

# ---------- Prompt text (we slice below) ----------
BASE_PAR="Intr-o seara ma intorceam de la munca si am vazut pe strada un catel ud care se adapostea sub un chiosc inchis. Ploua marunt, iar luminile din vitrine se reflectau in balti, facand orasul sa para un film vechi. M-am oprit, am scos din rucsac o punga goala si am rupt un colt de paine pe care il pastrasem de la pranz. Catelul a ezitat, apoi s-a apropiat cu coada intre picioare, dar ochii ii scanteiau de curiozitate. In departare se auzea tramvaiul, iar pe trotuar treceau oameni cu pas grabit, neobservand micul spectacol al increderii care incerca sa se nasca intre doi straini. Mi-am pus haina pe umeri ca pe o manta improvizata si m-am aplecat. In clipa aceea, cineva a strigat din spatele meu ca ar fi mai bine sa nu ating animalul, ca poate musca. Dar vocea aceea parea mai mult o teama veche decat un avertisment. Catelul a luat firimiturile si s-a asezat la un pas, urmarindu-ma cu atentie. I-am intins palma goala, iar el, dupa cateva clipe, a mirosit-o si a tresarit, ca si cum si-ar fi amintit ca lumea poate fi blanda. Am zambit singur si mi-am dat seama ca nu eram grabit nicaieri, desi ceasul trecuse de noua. Am pornit incet spre casa, iar el m-a urmat la distanta, oprindu-se la fiecare colt ca sa se asigure ca nu il chem intr-o capcana. Cand am ajuns la bloc, s-a asezat la scari, privind spre geamurile intunecate ca si cum ar fi citit povesti nespuse. I-am lasat un castron improvizat din capacul cutiei mele de pranz si am turnat apa din sticla. In camera mea modesta, am aprins o veioza si m-am gandit la drumul pe care il parcurge uneori increderea: incepe cu o ezitare, continua cu o firimitura, se leaga de un pas comun si se odihneste langa o usa care poate candva se va deschide."

make_prompt () {
  local ctx="$1" n_chars
  case "$ctx" in
    512)  n_chars=220  ;;
    1024) n_chars=520  ;;
    2048) n_chars=1200 ;;
    4096) n_chars=2600 ;;
    *)    n_chars=400  ;;
  esac
  local text=""
  while [[ ${#text} -lt $n_chars ]]; do text+="$BASE_PAR "; done
  echo "${text:0:$n_chars}"
}

# ---------- Helpers ----------
get_mem_stats () {
  local pid rss hwm
  pid=$(pgrep -f "ollama serve" | head -n1 || true)
  if [[ -n "${pid}" && -r "/proc/$pid/status" ]]; then
    rss=$(awk '/VmRSS/{print int($2/1024)}' "/proc/$pid/status" 2>/dev/null || echo 0)
    hwm=$(awk '/VmHWM/{print int($2/1024)}' "/proc/$pid/status" 2>/dev/null || echo 0)
    echo "${rss},${hwm}"
  else
    echo "NA,NA"
  fi
}
get_sys_mem_avail_mb () { awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null || echo "NA"; }
read_cpu_temp_once () {
  if command -v vcgencmd >/dev/null; then
    vcgencmd measure_temp 2>/dev/null | awk -F"=|'" '{print $2}'
  else
    for p in /sys/class/thermal/thermal_zone*/temp; do
      [[ -r "$p" ]] || continue
      v=$(cat "$p" 2>/dev/null || echo "")
      if [[ "$v" =~ ^[0-9]+$ ]]; then awk -v t="$v" 'BEGIN{printf("%.1f\n", t/1000.0)}'
      elif [[ -n "$v" ]]; then echo "$v"; fi
      return
    done
  fi
}

# ---------- Warmup ----------
echo "[INFO] Warming up models…"
for m in "${MODELS[@]}"; do
  curl -sS --connect-timeout 3 --max-time 5 "$HOST/api/generate" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$m"'","prompt":"warmup","stream":false,"options":{"num_ctx":512,"num_predict":4,"num_thread":2}}' >/dev/null || true
done
for m in "${MODELS[@]}"; do
  curl -sS --connect-timeout 5 --max-time 8 "$HOST/api/generate" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$m"'","prompt":"warmup 4k","stream":false,"options":{"num_ctx":4096,"num_predict":8,"num_thread":2}}' >/dev/null || true
done
echo "[INFO] Warmup done."

# ---------- Main ----------
for model in "${MODELS[@]}"; do
  for th in "${THREADS[@]}"; do
    for ctx in "${CTX[@]}"; do
      base_prompt="$(make_prompt "$ctx")"
      for pred in "${PRED[@]}"; do
        for rep in $(seq 1 "$REPS"); do
          # --------- CACHE OFF: unique stamp + unique system string ----------
          STAMP=" [bench ${HOSTNAME} ${model} th=${th} ctx=${ctx} pred=${pred} rep=${rep} @$(date +%s%N)] "
          P_JSON=$(jq -Rs . <<<"$STAMP$base_prompt")
          SYS_JSON=$(jq -Rs . <<<"bench-$HOSTNAME th=$th ctx=$ctx pred=$pred rep=$rep $(date +%s%N)")
          echo "[RUN] model=${model} rep=${rep} threads=${th} ctx=${ctx} pred=${pred} (cache=OFF stamp=$(printf '%s' "$STAMP" | sha1sum | cut -c1-12))"

          RESP_TMP=$(mktemp /tmp/ollama_resp_XXXXXX.jsonl)
          t0=$(date +%s.%N)

          curl -sS --http1.1 --no-buffer --connect-timeout 10 --max-time 240 \
            "$HOST/api/generate" -H "Content-Type: application/json" \
            -d '{"model":"'"$model"'","system":'"$SYS_JSON"',"prompt":'"$P_JSON"',"stream":true,"options":{"num_ctx":'"$ctx"',"num_predict":'"$pred"',"num_thread":'"$th"',"temperature":0.7,"top_p":0.9}}' \
          | awk -v T0="$t0" '
              BEGIN{ gotfirst=0 }
              {
                if(!gotfirst){
                  cmd="date +%s.%N"; cmd | getline now; close(cmd);
                  print "__FIRST__ " now;
                  gotfirst=1;
                }
                print $0;
                if(index($0,"\"done\":true")>0){ fflush(); exit 0 }
              }
            ' > "$RESP_TMP"

          first_ts=$(awk '/^__FIRST__/ {print $2; exit}' "$RESP_TMP")
          if [[ -n "${first_ts:-}" ]]; then
            ttft_sec=$(awk -v a="$t0" -v b="$first_ts" 'BEGIN{printf("%.4f", b-a)}')
          else
            ttft_sec=""
          fi

          FINAL_JSON=$(tac "$RESP_TMP" | grep -m1 '"done":true' || true)
          if [[ -z "$FINAL_JSON" ]]; then
            echo "   → [WARN] missing done:true; skipping row"
            rm -f "$RESP_TMP"
            sleep "${COOLDOWN}"
            continue
          fi

          prompt_count=$(jq -r '.prompt_eval_count // 0' <<<"$FINAL_JSON")
          eval_count=$(jq -r '.eval_count // 0' <<<"$FINAL_JSON")
          p_dur=$(jq -r '.prompt_eval_duration // 0' <<<"$FINAL_JSON")
          e_dur=$(jq -r '.eval_duration // 0' <<<"$FINAL_JSON")
          t_dur=$(jq -r '.total_duration // 0' <<<"$FINAL_JSON")

          prompt_sec=$(awk -v n=$p_dur 'BEGIN{printf("%.4f", n/1e9)}')
          gen_sec=$(awk -v n=$e_dur 'BEGIN{printf("%.4f", n/1e9)}')
          total_sec=$(awk -v n=$t_dur 'BEGIN{printf("%.4f", n/1e9)}')
          prompt_tps=$(awk -v c=$prompt_count -v s=$prompt_sec 'BEGIN{if(s>0) printf("%.2f", c/s); else print 0}')
          gen_tps=$(awk -v c=$eval_count   -v s=$gen_sec    'BEGIN{if(s>0) printf("%.2f", c/s); else print 0}')

          read -r mem_rss_mb mem_hwm_mb <<< "$(get_mem_stats | tr ',' ' ')"
          sys_mem_avail_mb="$(get_sys_mem_avail_mb)"
          cpu_temp="$(read_cpu_temp_once || true)"

          echo "   → tokens: prompt=${prompt_count} gen=${eval_count} | TTFT=${ttft_sec:-NA}s | prompt_tps=${prompt_tps} | gen_tps=${gen_tps} | rss=${mem_rss_mb}MB | cpu≈${cpu_temp}°C"

          echo "$(date +%F_%T),$HOSTNAME,$SOC_LABEL,$OS_REL,$model,$rep,$th,$ctx,$pred,$prompt_count,$eval_count,$prompt_sec,$gen_sec,${ttft_sec:-},$total_sec,$prompt_tps,$gen_tps,$mem_rss_mb,$mem_hwm_mb,$sys_mem_avail_mb,$cpu_temp" >> "$CSV"

          rm -f "$RESP_TMP"
          sleep "${COOLDOWN}"
        done
      done
    done
  done
done

echo "Results saved to $CSV"
