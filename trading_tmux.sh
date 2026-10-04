#!/bin/bash
tmux new-window -n "bnb" "source ./mainnet_api.sh && ./trade_bnb.sh; exec $SHELL"
tmux new-window -n "sol" "source ./mainnet_api.sh && ./trade_sol.sh; exec $SHELL"
tmux new-window -n "btc" "source ./mainnet_api.sh && ./trade_btc.sh; exec $SHELL"

tmux new-window -n "trading_tail" "tail -f trader-operations-BNBUSDT-mainnet.log"
tmux split-window -t ":trading_tail" -v "tail -f trader-operations-SOLUSDT-mainnet.log"
tmux split-window -t ":trading_tail" -v "tail -f trader-operations-BTCUSDT-mainnet.log"

