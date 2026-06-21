# sae_statind

## Infrastructure

This project lives on **Vulcan** (Alliance cluster). The local working directory is mounted via SSHFS from the remote path `/project/aip-bahtol/rhyderi1/sae_statind`.

- **SSH alias:** `vulcan`
- **SLURM account:** `aip-bahtol`
- **Remote path:** `/project/aip-bahtol/rhyderi1/sae_statind`

## Running Jobs

Submit a SLURM job:
```bash
ssh vulcan "cd /project/aip-bahtol/rhyderi1/sae_statind && sbatch <script>"
```

Check the queue:
```bash
ssh vulcan "squeue -u rhyderi1"
```
