#!/bin/bash

# CUIDADO: nao modifique os parametros abaixo
NX=8
NZ=4
FREQ=100
CONFIG=CONFIG_IN_TEST1
SIM_TIME=100

# TODO modifique a compilacao do programa abaixo se precisar
nvcc -D_NX=$NX -D_NZ=$NZ -D_SIM_TIME=$SIM_TIME -D_OUT_FREQ=$FREQ -D_IN_CONFIG=$CONFIG -std=c++11 -O3 -o miniCFD miniCFD_serial.cu 
