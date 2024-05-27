#! /bin/sh

FILE=pytorch_phi3.py
TS=$(date +%Y-%m-%d_%H-%M-%S)

if [ -z "$VIRTUAL_ENV" ]; then
    . .venv/bin/activate
fi

cat $FILE > log_"$TS".txt
python $FILE >> log_"$TS".txt 2>&1
