#!/bin/bash

python3 simulation.py -t $PWD/topology/microbench/microbenchmark.csv\
 -c $PWD/config/ex_mnk_ws.cfg\
 -i "gemm"

# python3 simulation.py -t $PWD/topology/microbench/microbenchmark.csv\
#  -c $PWD/config/ex_mnk_os.cfg\
#  -i "gemm"

# python3 simulation.py -t $PWD/topology/microbench/microbenchmark.csv\
#  -c $PWD/config/ex_mnk_is.cfg\
#  -i "gemm"

#python3 simulation.py -t $PWD/topology/microbench/microbenchmark_part.csv\
# -c $PWD/config/ex_mnk_is.cfg\
# -i "gemm"

# python3 simulation.py -t $PWD/topology/microbench/microbenchmark_part.csv\
# -c $PWD/config/ex_mnk_ws.cfg\
# -i "gemm"

# python3 simulation.py -t $PWD/topology/microbench/microbenchmark_part.csv\
# -c $PWD/config/ex_mnk_os.cfg\
# -i "gemm"



# python3 simulation.py -t $PWD/topology/microbench/microbenchmark_conv.csv\
#  -c $PWD/config/ex_mnk_is.cfg\
#  -i "conv"

# python3 simulation.py -t $PWD/topology/microbench/microbenchmark_conv.csv\
#  -c $PWD/config/ex_mnk_ws.cfg\
#  -i "conv"

# python3 simulation.py -t $PWD/topology/microbench/microbenchmark_conv.csv\
#  -c $PWD/config/ex_mnk_os.cfg\
#  -i "conv"