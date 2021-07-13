#!/bin/sh

for var in 0 1 2 3 4

do 
    python train.py --fold $var --model rf
done


python test.py
