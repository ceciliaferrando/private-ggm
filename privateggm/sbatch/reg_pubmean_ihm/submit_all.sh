#!/bin/bash
# Submit all PACE-GGM regression jobs + IHM regression job
cd /mnt/nfs/work1/sheldon/cferrando/private-ggm/privateggm/sbatch/reg_pubmean_ihm

for DS in adult SeoulBikeData LifeExpectancy ibm bank32nh Ailerons; do
    sbatch sbatch_pace_${DS}.sh
done

sbatch sbatch_ihm_all.sh
