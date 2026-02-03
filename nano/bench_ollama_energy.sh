set -euo pipefail

export OLLAMA_NUM_GPU="${OLLAMA_NUM_GPU:-0}"
HOST="${HOST:-http://127.0.0.1:11434}"

MODELS=("nano-125m-q5" "nano-125m-f16" "nano-260m-q5" "nano-260m-f16")
THREADS=(2 4)
CTX=(512 1024 2048 4096)
PRED=(32 64 128)
REPS=${REPS:-3}
IDLE_SECONDS=${IDLE_SECONDS:-3}
COOLDOWN=${COOLDOWN:-2}

SOC_LABEL="${SOC_LABEL:-jetson-nano-4gb}"
JETPACK="$(dpkg -l | awk '/nvidia-l4t-core/{print $3; exit}' || true)"
HOSTNAME="$(hostname)"
OUT="${HOME}/NANO_GOOD_BENCH.csv"
export LC_ALL=C

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
  local text=""
  while [[ ${#text} -lt $n_chars ]]; do text+="$LONG_PROMPT "; done
  printf "%s" "${text:0:n_chars}"
}

if [[ ! -f "$OUT" ]]; then
  echo "ts,host,soc,jetpack,model,rep,num_thread,num_ctx,num_predict,prompt_tokens,gen_tokens,prompt_sec,gen_sec,ttft_sec,total_sec,prompt_tps,gen_tps,proc_rss_mb,proc_hwm_mb,sys_mem_avail_mb,avg_power_w,max_power_w,idle_power_w,net_power_w,energy_total_j,energy_per_generated_token_j,energy_per_all_tokens_j,avg_gpu_temp_c,avg_cpu_temp_c,throttling" > "$OUT"
fi
echo "[INFO] CSV output: $OUT"

command -v tegrastats >/dev/null || { echo "tegrastats missing"; exit 1; }
command -v jq >/dev/null || { echo "jq missing: sudo apt-get install -y jq"; exit 1; }
pgrep -f "ollama serve" >/dev/null || { echo "Starting ollama serve..."; nohup ollama serve > "${HOME}/ollama.log" 2>&1 & sleep 2; }

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
run_tegrastats_bg () { local f="$1"; : > "$f"; tegrastats --interval 1000 > "$f" 2>/dev/null & echo $!; }

avg_from_tegrastats () {
  local file="$1"
  awk '
    {
      for (i=1;i<=NF;i++) {
        if ($i=="POM_5V_IN") { split($(i+1),a,"/"); gsub(/mW/,"",a[1]); if (a[1]~/^[0-9]+$/){p+=a[1];np++} }
        if ($i ~ /^GPU@/) { gsub(/GPU@/,"",$i); gsub(/C.*/,"",$i); if ($i~/^[0-9]+$/){g+= $i; ng++} }
        if ($i ~ /^CPU@/) { gsub(/CPU@/,"",$i); gsub(/C.*/,"",$i); if ($i~/^[0-9]+$/){c+= $i; nc++} }
        if (tolower($i) ~ /throt/) { thr=1 }
      }
      if ($(NF-1)=="POM_5V_IN") { split($NF,a,"/"); gsub(/mW/,"",a[1]); if (a[1]~/^[0-9]+$/ && a[1]>pmax){pmax=a[1]} }
    }
    END {
      aw = (np>0)? p/np : 0;
      printf("%.3f %.3f %.1f %.1f %s\n", aw/1000.0, (pmax? pmax/1000.0:0), (ng? g/ng:0), (nc? c/nc:0), (thr? "yes":"no"))
    }' "$file"
}

idle_power_now () {
  local log=/tmp/ts_idle_$$.log
  tegrastats --interval 1000 > "$log" 2>/dev/null &
  local tp=$!; sleep "${IDLE_SECONDS}"; kill "$tp" >/dev/null 2>&1 || true
  awk '{
    for(i=1;i<=NF;i++){
      if($i=="POM_5V_IN"){
        split($(i+1),a,"/"); gsub(/mW/,"",a[1]);
        if (a[1] ~ /^[0-9]+$/) {sum+=a[1]; n++}
      }
    }
  } END { if(n>0) printf("%.3f\n", (sum/n)/1000.0); else print "0.000" }' "$log"
  rm -f "$log"
}

for model in "${MODELS[@]}"; do
  curl -sS --connect-timeout 3 --max-time 5  "$HOST/api/generate" -H "Content-Type: application/json" \
    -d '{"model":"'"$model"'","prompt":"warmup","stream":false,"options":{"num_ctx":512,"num_predict":8,"num_thread":2},"keep_alive":0}' >/dev/null || true
  curl -sS --connect-timeout 5 --max-time 8  "$HOST/api/generate" -H "Content-Type: application/json" \
    -d '{"model":"'"$model"'","prompt":"warmup 4k","stream":false,"options":{"num_ctx":4096,"num_predict":8,"num_thread":2},"keep_alive":0}' >/dev/null || true
done

for model in "${MODELS[@]}"; do
  for th in "${THREADS[@]}"; do
    for ctx in "${CTX[@]}"; do
      base_p="$(prompt_for_ctx "$ctx")"
      P_JSON=$(jq -Rs . <<<"$base_p")
      for pred in "${PRED[@]}"; do
        for rep in $(seq 1 "$REPS"); do
          echo "[RUN] model=${model} rep=${rep} threads=${th} ctx=${ctx} pred=${pred}"

          idle_w="$(idle_power_now)"
          RUN_LOG=$(mktemp /tmp/tegrastats_run_XXXXXX.log)
          TEGRA_PID=$(run_tegrastats_bg "$RUN_LOG")

          RESP_TMP=$(mktemp /tmp/ollama_resp_XXXXXX.jsonl)
          t0=$(date +%s.%N)

          curl -sS --http1.1 --no-buffer --connect-timeout 10 --max-time 240 \
            "$HOST/api/generate" -H "Content-Type: application/json" \
            -d '{"model":"'"$model"'","system":"bench-stable v1","prompt":'"$P_JSON"',"stream":true,"keep_alive":0,"options":{"num_ctx":'"$ctx"',"num_predict":'"$pred"',"num_thread":'"$th"',"temperature":0.7,"top_p":0.9,"seed":42}}' \
          | awk -v T0="$t0" '
              BEGIN { gotfirst=0 }
              {
                if (!gotfirst) {
                  cmd="date +%s.%N"; cmd | getline now; close(cmd);
                  print "__FIRST__ " now;     # <-- write a marker line
                  gotfirst=1;
                }
                print $0;
                if (index($0,"\"done\":true")>0) { fflush(); exit 0 }
              }
            ' > "$RESP_TMP"

          kill "$TEGRA_PID" >/dev/null 2>&1 || true
          sleep 0.2

          first_ts=$(awk '/^__FIRST__/ {print $2; exit}' "$RESP_TMP")
          ttft_sec=$(awk -v a="$t0" -v b="$first_ts" 'BEGIN{printf("%.4f", b-a)}')
          FINAL_JSON=$(tac "$RESP_TMP" | grep -m1 '"done":true' || true)
          [[ -z "$FINAL_JSON" ]] && FINAL_JSON=$(tail -n1 "$RESP_TMP" || echo "{}")

          prompt_count=$(echo "$FINAL_JSON" | jq -r '.prompt_eval_count // 0')
          eval_count=$(echo "$FINAL_JSON"   | jq -r '.eval_count // 0')
          p_dur=$(echo "$FINAL_JSON"        | jq -r '.prompt_eval_duration // 0')
          e_dur=$(echo "$FINAL_JSON"        | jq -r '.eval_duration // 0')
          t_dur=$(echo "$FINAL_JSON"        | jq -r '.total_duration // 0')

          prompt_sec=$(awk -v n=$p_dur 'BEGIN{printf("%.4f", n/1e9)}')
          gen_sec=$(awk -v n=$e_dur 'BEGIN{printf("%.4f", n/1e9)}')
          total_sec=$(awk -v n=$t_dur 'BEGIN{printf("%.4f", n/1e9)}')
          prompt_tps=$(awk -v c=$prompt_count -v s=$prompt_sec 'BEGIN{if(s>0) printf("%.2f", c/s); else print 0}')
          gen_tps=$(awk -v c=$eval_count   -v s=$gen_sec    'BEGIN{if(s>0) printf("%.2f", c/s); else print 0}')

          read -r mem_rss_mb mem_hwm_mb <<< "$(get_mem_stats | tr ',' ' ')"
          sys_mem_avail_mb="$(get_sys_mem_avail_mb)"
          read -r avg_w max_w avg_gpu_c avg_cpu_c thr_flag <<< "$(avg_from_tegrastats "$RUN_LOG")"
          net_w=$(awk -v a="$avg_w" -v i="$idle_w" 'BEGIN{v=a-i; if(v<0) v=0; printf("%.3f", v)}')

          total_tokens=$((prompt_count + eval_count))
          energy_total_j=$(awk -v w="$net_w" -v s="$total_sec" 'BEGIN{printf("%.4f", w*s)}')
          ept_gen_j=$(awk -v e="$energy_total_j" -v n="$eval_count" 'BEGIN{if(n>0) printf("%.6f", e/n); else print "0.000000"}')
          ept_all_j=$(awk -v e="$energy_total_j" -v n="$total_tokens" 'BEGIN{if(n>0) printf("%.6f", e/n); else print "0.000000"}')

          echo "   -> tokens: prompt=${prompt_count} gen=${eval_count} | TTFT=${ttft_sec}s | prompt_tps=${prompt_tps} | gen_tps=${gen_tps} | rss=${mem_rss_mb}MB | P(avg)=${avg_w}W"

          echo "$(date +%F_%T),$HOSTNAME,$SOC_LABEL,$JETPACK,$model,$rep,$th,$ctx,$pred,$prompt_count,$eval_count,$prompt_sec,$gen_sec,$ttft_sec,$total_sec,$prompt_tps,$gen_tps,$mem_rss_mb,$mem_hwm_mb,$sys_mem_avail_mb,$avg_w,$max_w,$idle_w,$net_w,$energy_total_j,$ept_gen_j,$ept_all_j,$avg_gpu_c,$avg_cpu_c,$thr_flag" >> "$OUT"

          rm -f "$RUN_LOG" "$RESP_TMP"
          sleep "${COOLDOWN}"
        done
      done
    done
  done
done

echo "Results saved to $OUT"

