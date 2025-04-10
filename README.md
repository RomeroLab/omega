# OMEGA

This is the code to run the OMEGA program presented in Freschlin et al. bioRxiv (2024). We provide example files and preliminary documentation on OMEGA options. Please note that the code provided below may contain bugs. 

## Install OMEGA
OMEGA can be run using the minimal conda environment provided with this code. Alternatively, you can open the `omega_colab.ipynb` notebook in Google Colab and run OMEGA there if there is an installation problem with the conda environment.

```
conda env create --file environment.yml
conda activate omega
```

#### Verify installation
We provide a test optimization to verify install. It performs a full library desig protocol using very few optimization steps so it is fast.
```
python ./code/omega.py genes --config configs/test_install.yml
```

#### Test a full run
The following includes commands for a simple library design using 7 subpools. Each subpool uses 50 junctions and is optimized 5 times - the best solution is chosen for fragment design. By default, it uses 5 CPUs to run parallel optimization runs for each subpool. The runtime varies significantly by system. To decrease CPU usage, add `--njobs [N cpus]` after the `--config` argument. 
```
python ./code/omega.py genes --config configs/genes_test.yml
#   --njobs 1
```

## Examples

#### Optimize your own library

We provide a template config you can use to optimize your own library. Please see `configs/template.yml` for default values. You can optimize your sequences by filling in the parameters below. We provide the default test primers as the primer argument. These are highly specific primers designed by Subramanian et al. and are what we use to amplify subpools.

For 50 junctions, running the algorithm for 1000 steps and doing 5 independent optimizations for each subpool is sufficient. However, increasing steps to 3k and number of optimizations will marginally improve fidelity. More complex assemblies may require more optimizations steps or runs.

```
python ./code/omega.py genes --config configs/template.yml \
    --input_seqs [.fasta file] \
    --njunctions [number of GG site junctions per subpool] \
    --upstream_bbsite [upstream backbone GG site] \
    --downstream_bbsite [downstream backbone GG site] \
    --primers ./data/test_primers.csv \
    --nopt_steps 1000 \
    --nopt_runs 5
```

#### Modify arguments with commandline
OMEGA uses a config to set various runtime parameters. Any of these can be passed as commandline arguments that override the default config values. For example, the below code updates the number of GG sites per subpool from 50 to 70 without modifying the config. For a full explanation of OMEGA parameters, please see Options.
```
python ./code/omega.py --config configs/test_install.yml \
    --njunctions 70
```

## Output files

OMEGA includes 3 files as output. These are written to the output directly indicated in the OMEGA program. See below for a brief explanation of files.

`oligo_order.csv` includes just oligo sequences for designed library. These may be submitted directly to order an oligopool. `optimization_results.csv` is the most comprehensive output file. For each gene, it includes the gene name, submitted sequence, oligo sequence, forward and reverse primers, and fidelity. 

`pool_stats.csv` is a pool-level view of the optimization results. It provides fidelities, number of genes per pool, number of sites used in each pool, random seed used to design the pool, Type IIS restriction enzyme, and primer information for each pool.

#### Explanation on fidelity calculations
We report fidelity in 3 ways. The first is `fidelity`, which is the same fidelity calculation reported in Pryor et al. This assumes that all GG sites are being used in a single sequential assembly - it does not fully reflect OMEGA conditions. This metric is used to guide fragment design.

We also calculate the fidelity for each individual gene and report the lowest fidelity for each subpool as `min_gene_fidelity`. This is the more relavent metric for OMEGA. `min_gene_fidelity` takes into account the complex assembly background while limiting the fidelity calculation to the relavent gene length. `min_site_fidelity` reports the the least orthogonal site included in the optimized sites.


## A note on assembly conditions

In our paper, we use the fidelity data for an 18 hr digest at 37C using T4 DNA ligase because these conditions generated the highest fidelity GG sites. We include data for other enzymes/assembly conditions as provided by Potapov et al. and Pryor et al., but we *strongly* recommend using BsaI and the T4_18h_37C ligation data. These are the default for all configs.

## Options

All default options are set in `configs/genes_test.yml` and `configs/template.yml`. Each option can be included as command line arguments, which will override the config values. Please see below for a list of all options with a brief explanation.

OMEGA is still being refined - some arguments were created during development that will be removed or modified in the future. Those values omitted here but still included in the config files since the program requires them.

- `input_seqs`: an input fasta file with codon-optimized sequences. An example file is included in `data/fastas`.
- `primers`: file containing forward and reverse primer sequences. Please see `data/test_primers.csv` for example file format. All primer sequences should be written in 5' to 3' direction. We highly recommend using the Subramanian et al. primers since these were designed to amplify DNA from complex backgrounds with high-fidelity.
- `output_dir`: name of directory to write output files to. If the directory does not exist, OMEGA will make one.
- `enzyme`: Type IIS restriction enzyme used to assemble library. Accepted options are [BsaI, BsmBI, BbsI, SapI].
- `upstream_bbsite`: upstream vector ligation site. (ex. AATG)
- `downstream_bbsite`: downstream vector ligation site. (ex. TTAG)
- `ligation_data`: Ligation frequency data from Potapov et al. to use in fidelity calculation. All experiments in OMEGA paper used T4_18h_37C, which use T4 ligase and an 18 hour incubation at 37C. Accepted options are [T4_01h_25C, T4_18h_25C, T4_01h_37C, T4_18h_37C] from Potapov et al. and [BsaI_cycling, BbsI_cycling, BsmBI_cycling, Esp3I_cycling, SapI_cycling] from Pryor et al.
- `njunctions`: number of Golden Gate sites used to assemble each subpool. This number includes backbone vector sites, so for example if you specify njunctions=50, 48 GG sites are used to design fragments.
- `nopt_steps`: number of optimization steps used to design GG sites. The default is 3000. 
- `nopt_runs`: the number of times OMEGA will design GG sites for each subpool. Each run uses a separate random seed and the run with the best fidelity is taken as the solution. Increasing run number is recommended for improving fidelty more than `nopt_steps`.
- `add_primers`: whether primers should be added to oligopool sequences in `oligo_order.csv`. Default is True.
- `pad_oligos`: whether random DNA should be added between the GG site and primer binding site. DNA does not include Type IIS restriction enzyme indicated by `enzyme`. Future updates will add support to exclude any DNA sequence to support downstream cloning applications that may use other kinds of restriction enzymes.
- `njobs`: number of CPUs to run jobs in parallel when optimizing a single subpool. OMEGA uses `joblib` to parallelize runs defined by `nopt_runs` or `opt_seeds`. The default value is 1, but it's recommended to use more than that when optimizing pools. It significantly speeds up OMEGA.
- `oligo_len`: max oligo length.
- `opt_seeds`: Instead of indicating `nopt_runs`, instead provide a list of random seeds to use to intialize fragment design. This argument is partly an artifact from development, but can be useful for reproducibility. The number of seeds provided indicates the number of optimizations run for each pool. `opt_seeds` is mutually exclusive with `nopt_runs`. `nopt_runs` is sufficient in nearly all cases.


## References

Freschlin, C. F., Yang, K. K., Romero, P. A. Scalable and cost-efficient custom gene library assembly from oligopools. bioRxiv (2025)

Pryor, J. M. et al. Enabling one-pot Golden Gate assemblies of unprecedented complexity using data-optimized assembly design. PLoS One 15, e0238592 (2020)

Subramanian, S. K., Russ, W. P. & Ranganathan, R. A set of experimentally validated, mutually orthogonal primers for combinatorially specifying genetic components. Synth Biol 3, (2018).

Potapov, V. et al. Comprehensive profiling of four base overhang ligation fidelity by T4 DNA ligase and application to DNA assembly. ACS Synth Biol 7, 2665–2674 (2018)
