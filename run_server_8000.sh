#!/bin/bash

PORT=8000
GPU_ID=1
OUTPUT_ROOT="/work/facefusion/temp/8000"
LOG_FILE="/work/facefusion/logs/8000.log"
PID_FILE="server_${PORT}.pid"

# 解决 libtinfo 警告
export TERM=xterm

case "$1" in
    start)
        if [ -f $PID_FILE ]; then
            echo "Service is already running (PID: $(cat $PID_FILE))"
        else
            echo "Starting FaceFusion API Service on Port $PORT..."
            mkdir -p "$(dirname "$LOG_FILE")"
            
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            nohup python api_server.py \
                --port $PORT \
                --output_dir "$OUTPUT_ROOT" \
                --log_file "$LOG_FILE" > /dev/null 2>&1 &
            
            echo $! > $PID_FILE
            echo "Service started with PID $!"
        fi
        ;;
    stop)
        if [ -f $PID_FILE ]; then
            PID=$(cat $PID_FILE)
            echo "Stopping service (PID: $PID)..."
            kill $PID 2>/dev/null || echo "Process not found, clearing PID file."
            rm $PID_FILE
            echo "Service stopped."
        else
            echo "Service is not running."
        fi
        ;;
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
    *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
        ;;
esac
