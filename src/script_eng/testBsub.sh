gpu_option="num=1:gtile=1:mode=exclusive_process"
for i in {1..35}
do
 # your-unix-command-here
 if [ ${i} -lt 10 ]
 then
 bsub -gpu ${gpu_option} -0 test_${i}.out -e err_${i}.err "bash ./run0${i}.sh"
 else
 bsub -gpu ${gpu_option} -0 test_${i}.out -e err_${i}.err "bash ./run${i}.sh"
 fi

done