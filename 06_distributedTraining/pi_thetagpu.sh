#!/bin/bash -l
#COBALT -t 20
#COBALT -q training-gpu
#COBALT -n 1
#COBALT -A ALCFAITP
#COBALT --attrs filesystems=home,grand

# Set up software deps:
module load conda/2022-07-01
conda activate

# You have to point this to YOUR local copy of ai-science-training-series

export TF_XLA_FLAGS="--tf_xla_auto_jit=2"
mpirun -np 1 python pi.py   # 3.1427248 Time: 14.03110408782959 s
mpirun -np 2 python pi.py   # 3.1406688 Time: 4.5909953117370605 s
mpirun -np 4 python pi.py   # 3.1413696 Time: 2.3250551223754883 s
mpirun -np 8 python pi.py   # 3.1412944 Time: 1.1389317512512207 s
mpirun -np 16 python pi.py  # 3.1422056 Time: 0.7726776599884033 s
# mpirun -np 32 python pi.py  # ,     s
# mpirun -np 64 python pi.py  # ,     s
# mpirun -np 128 python pi.py # ,     s







###
# -----
###
# --------------------------------------------------------------------------
# There are not enough slots available in the system to satisfy the 32
# slots that were requested by the application:

#   python

# Either request fewer slots for your application, or make more slots
# available for use.

# A "slot" is the Open MPI term for an allocatable unit where we can
# launch a process.  The number of slots available are defined by the
# environment in which Open MPI processes are run:

#   1. Hostfile, via "slots=N" clauses (N defaults to number of
#      processor cores if not provided)
#   2. The --host command line parameter, via a ":N" suffix on the
#      hostname (N defaults to 1 if not provided)
#   3. Resource manager (e.g., SLURM, PBS/Torque, LSF, etc.)
#   4. If none of a hostfile, the --host command line parameter, or an
#      RM is present, Open MPI defaults to the number of processor cores

# In all the above cases, if you want Open MPI to default to the number
# of hardware threads instead of the number of processor cores, use the
# --use-hwthread-cpus option.

# Alternatively, you can use the --oversubscribe option to ignore the
# number of available slots when deciding the number of processes to
# launch.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# There are not enough slots available in the system to satisfy the 64
# slots that were requested by the application:

#   python

# Either request fewer slots for your application, or make more slots
# available for use.

# A "slot" is the Open MPI term for an allocatable unit where we can
# launch a process.  The number of slots available are defined by the
# environment in which Open MPI processes are run:

#   1. Hostfile, via "slots=N" clauses (N defaults to number of
#      processor cores if not provided)
#   2. The --host command line parameter, via a ":N" suffix on the
#      hostname (N defaults to 1 if not provided)
#   3. Resource manager (e.g., SLURM, PBS/Torque, LSF, etc.)
#   4. If none of a hostfile, the --host command line parameter, or an
#      RM is present, Open MPI defaults to the number of processor cores

# In all the above cases, if you want Open MPI to default to the number
# of hardware threads instead of the number of processor cores, use the
# --use-hwthread-cpus option.

# Alternatively, you can use the --oversubscribe option to ignore the
# number of available slots when deciding the number of processes to
# launch.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# There are not enough slots available in the system to satisfy the 128
# slots that were requested by the application:

#   python

# Either request fewer slots for your application, or make more slots
# available for use.

# A "slot" is the Open MPI term for an allocatable unit where we can
# launch a process.  The number of slots available are defined by the
# environment in which Open MPI processes are run:

#   1. Hostfile, via "slots=N" clauses (N defaults to number of
#      processor cores if not provided)
#   2. The --host command line parameter, via a ":N" suffix on the
#      hostname (N defaults to 1 if not provided)
#   3. Resource manager (e.g., SLURM, PBS/Torque, LSF, etc.)
#   4. If none of a hostfile, the --host command line parameter, or an
#      RM is present, Open MPI defaults to the number of processor cores

# In all the above cases, if you want Open MPI to default to the number
# of hardware threads instead of the number of processor cores, use the
# --use-hwthread-cpus option.

# Alternatively, you can use the --oversubscribe option to ignore the
# number of available slots when deciding the number of processes to
# launch.
# --------------------------------------------------------------------------
# osogunro@thetagpu06:~/Projects/ai-science-training-series/06_distributedTraining$