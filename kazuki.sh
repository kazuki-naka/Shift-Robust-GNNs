#!bin/bash

python3 main_gnn.py --n-epochs=200 --dataset="citeseer" --n-repeats=100 --n-hidden=32 --dropout=0.5 --weight-decay=0.0005 --biased-sample=True --SR=True