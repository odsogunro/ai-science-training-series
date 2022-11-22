###
# manually
###
# ssh <username>@theta.alcf.anl.gov
# ssh thetagpusn1 
# qsub -A ALCFAITP -n 1 -q training-gpu -t 20 -I 
# qsub -A ALCFAITP -n 1 -q single-gpu -t 60 -I 
# module load conda/2022-07-01
# conda activate
# cd YOUR_GITHUP_REPO
# mpirun -np 1 python pi.py   # 3.141988,   8.029037714004517  s
# mpirun -np 2 python pi.py   # 3.1415096   4.212774038314819  s
# mpirun -np 4 python pi.py   # 3.1425632   2.093632459640503  s
# mpirun -np 8 python pi.py   # 3.1411632   1.0610620975494385 s
