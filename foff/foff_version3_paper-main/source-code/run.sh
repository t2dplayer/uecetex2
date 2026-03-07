#!/bin/bash

rm *.json *.pdf

# Array de tempos de espera na fila
queue_waiting_times=(3)
# Array de cenários (use "challenging" e "balanced" conforme definido no argparse)
scenarios=("challenging" "balanced")
# Array de métodos de extração


# Loop para todas as combinações
for qt in "${queue_waiting_times[@]}"; do
  for sc in "${scenarios[@]}"; do    
    echo "Executando com queue_waiting_time=${qt}, scenario=${sc}"
    python main.py --queue-waiting-time="$qt" --scenario="$sc"
  done
done

