export OMP_NUM_THREADS=16
#export KMP_AFFINITY=granularity=fine,compact
#export KMP_AFFINITY=granularity=fine,balanced
unset KMP_AFFINITY
export OMP_PROC_BIND=true