"""Design oligopool for scalable gene assembly."""

import os
from os.path import join, exists
from typing import Optional, Union
import random

from jsonargparse import CLI
import pandas as pd
import numpy as np

# Cartographer imports
from data_classes import Enzyme, EnzymeTypes, LigationDataOpt, define_ligation_data, PrimerIterator, define_enzyme
from library_classes import Library
from junctions import optimize_junctions
from Bio import SeqIO


def genes(
        input_seqs: str,
        njunctions: int,
        upstream_bbsite: str,
        downstream_bbsite: str,
        primers: str,
        output_dir: str = 'output',

        other_used_sites: Union[list[str], None] = None,

        # enzyme and ligation data arguments
        enzyme: Union[EnzymeTypes, Enzyme] = 'BsaI',
        other_enzymes: Optional[EnzymeTypes] = None,
        ligation_data: LigationDataOpt = 'T4_18h_37C',

        # oligo packaging arguments
        add_primers: bool = True,
        pad_oligos: bool = True,

        # optimization arguments
        nopt_steps: int = 1000,
        nopt_runs: Optional[int] = 5,
        opt_seeds: Optional[list[int]] = None,
        njobs: int = 1,

        # miscellaneous
        oligo_len: int = 300,
        min_size: int = 40,
        optimization: str = 'simulated_annealing',
        dev: bool = False
) -> None:
    """
    Design library for pooled golden gate assembly.

    Args:
        input_seqs: File path to library sequences in fasta format.
        upstream_bbsite: GG site in str format if applicable.
        downstream_bbsite: GG site in str format if applicable.
        njunctions: Number of junctions used per pool - should include the upstream
            and downstream sites.
        other_used_sites: Other GG sites that are included in the assembly that are not
            upstream or downstream vector ligation sites.
        enzyme: Enzyme used to digest DNA fragments.
        ligation_data: Ligation data used to optimize the library.
        oligo_len: Oligo len used in oligopool.
        add_primers: If True, primer sequences are added to the ends of each oligo
        pad_oligos: If True, spacer DNA is added between primer sequences and restrction
            binding sites to make all oligos the same length to reduce amplification bias.
        min_size: Minimum fragment size allowed in fragmentation patterns.
        output_dir: Filepath to where optimization output is saved to.
        nopt_steps: Number of steps to run each optimization run.
        nopt_runs: Number of times each pool is optimized. Each new optimization run
            uses a different random seed.
        njobs: Number of CPUs to use for simultaneous optimization runs. Each CPU executes
            one run.

    """
    #pylint: disable=too-many-arguments, too-many-locals

    # if output directory doesn't exist, write it
    if not exists(output_dir):
        os.makedirs(output_dir)

    assembly_enzyme = define_enzyme(enzyme)
    #! Right now this only recognizes other TypeIIS enzymes
    other_enzymes = [define_enzyme(e) for e in other_enzymes] if other_enzymes is not None else []

    # format input sequences into Gene objects
    input_seqs_recs = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(input_seqs, 'fasta')]
    other_used_sites = other_used_sites or np.array([])
    primers = PrimerIterator(primers, assembly_enzyme)
    ligation_data = define_ligation_data(ligation_data, assembly_enzyme)

    # print input parameters
    print(f"Read {len(input_seqs_recs)} from {input_seqs}.\nSequence lengths are {min(map(lambda x: len(x[1]), input_seqs_recs))} - {min(map(lambda x: len(x[1]), input_seqs_recs))} bp")
    print(f"Using {enzyme.name} and {ligation_data.name} data to assemble library.")
    print(f"Maximum number of GG sites allowed per pool: {njunctions}")
    print(f"Upstream backbone site: {upstream_bbsite}")
    print(f"Downstream backbone site: {downstream_bbsite}")


    library = Library(
        genes=input_seqs_recs,
        primers=primers,
        oligo_len=oligo_len,
        enzyme=assembly_enzyme,
        upstream_bbsite=upstream_bbsite,
        downstream_bbsite=downstream_bbsite,
        other_used_sites=other_used_sites,
        min_size=min_size
    )

    # assign optimization seeds - use nopt_runs to get random_opt seeds
    random_seeds = None
    if nopt_runs is not None:
        rng = np.random.default_rng(seed=42)
        random_seeds = rng.integers(1000, size=nopt_runs)
    elif (nopt_runs is None) and (opt_seeds is not None):
        random_seeds = opt_seeds
    else:
        raise ValueError('Cannot provide values for both `nopt_runs` and `opt_seeds`.')

    # assign genes to pools
    library.optimize_pools(
        njunctions=njunctions,
        nopt_steps=nopt_steps,
        opt_seeds=random_seeds,
        njobs=njobs,
        ligation_data=ligation_data.data,
        optimization=optimization
    )

    optimized_library = library.package_library(add_primers=add_primers, pad_oligo=pad_oligos)
    optimized_library.to_csv(os.path.join(output_dir, 'optimization_results.csv'))
    print(f"Finished optimization. Saved to {os.path.join(output_dir, 'optimization_results.csv')}")

    oligopool = library.package_oligos(add_primers=add_primers, pad_oligo=pad_oligos)
    oligopool.to_csv(os.path.join(output_dir, 'oligo_order.csv'))
    print(f"Designed {len(oligopool)} oligos. Saved to {os.path.join(output_dir, 'oligo_order.csv')}")

    pool_stats = pd.DataFrame.from_dict(
        [{
            'pool':p.name,
            'fidelity':fidelity,
            'min_gene':p.min_gene_fidelity,
            'min_site':p.min_site_fidelity,
            'n_genes':len(p.genes),
            'n_sites':(p.nfrags-1)*len(p.genes) + 2 + len(p.other_used_sites),
            'seed':seed,
            'enzyme':p.enzyme.name,
            'pfwd_name':p.fprimer.name,
            'pfwd_sequence':p.fprimer.sequence,
            'prev_name':p.rprimer.name,
            'prev_sequence':p.rprimer.sequence
        } for p, fidelity, seed in library.optimized_pools]
    )
    pool_stats.to_csv(join(output_dir, 'pool_stats.csv'))
    print(f"Pool-level statistics saved to {join(output_dir, 'pool_stats.csv')}")

    with open(join(output_dir, 'experiment_details.txt'), 'w') as f:
        f.write(ligation_data.experiment_information())

    # if dev, save extra data
    if dev:
        # make trajectory save dir
        trajectory_dir = join(output_dir, 'opt_trajectories')
        if not exists(trajectory_dir):
            os.mkdir(trajectory_dir)
        for pool in library.all_opt_runs:
            # print(pool[0].opt_trajectory)
            np.save(join(trajectory_dir, f'pool_{pool[0].name}_seed-{pool[2]}.npy'), np.array(pool[0].opt_trajectory))

def split_fasta_by_gene(input_fasta_file, output_directory="individual_genes"):
    """
    Reads a multi-gene FASTA file and saves each gene to a separate FASTA file.

    Args:
        input_fasta_file (str): Path to the input FASTA file.
        output_directory (str, optional): Directory to save individual FASTA files.
            Defaults to "individual_genes".
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        for record in SeqIO.parse(input_fasta_file, "fasta"):
            # Sanitize the filename.  Replace any non-alphanumeric character (except
            # underscore, period, and hyphen) with an underscore.
            filename = record.id.replace(" ", "_")
            filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
            output_filename = os.path.join(output_directory, f"{filename}.fasta")
            SeqIO.write(record, output_filename, "fasta")
            print(f"Saved gene '{record.id}' to '{output_filename}'")
    except FileNotFoundError:
        print(f"Error: Input FASTA file '{input_fasta_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def one_gene_per_pool(
        input_seqs: str,
        njunctions: int,
        upstream_bbsite: str,
        downstream_bbsite: str,
        primers: str,
        output_dir: str = 'output',

        other_used_sites: Union[list[str], None] = None,

        # enzyme and ligation data arguments
        enzyme: Union[EnzymeTypes, Enzyme] = 'BsaI',
        other_enzymes: Optional[EnzymeTypes] = None,
        ligation_data: LigationDataOpt = 'T4_18h_37C',

        # oligo packaging arguments
        add_primers: bool = True,
        pad_oligos: bool = True,

        # optimization arguments
        nopt_steps: int = 1000,
        nopt_runs: Optional[int] = 5,
        opt_seeds: Optional[list[int]] = None,
        njobs: int = 1,

        # miscellaneous
        oligo_len: int = 300,
        min_size: int = 40,
        optimization: str = 'simulated_annealing',
        dev: bool = False
) -> None:
    """
    Design library for pooled golden gate assembly.

    Args:
        input_seqs: File path to library sequences in fasta format.
        upstream_bbsite: GG site in str format if applicable.
        downstream_bbsite: GG site in str format if applicable.
        njunctions: Number of junctions used per pool - should include the upstream
            and downstream sites.
        other_used_sites: Other GG sites that are included in the assembly that are not
            upstream or downstream vector ligation sites.
        enzyme: Enzyme used to digest DNA fragments.
        ligation_data: Ligation data used to optimize the library.
        oligo_len: Oligo len used in oligopool.
        add_primers: If True, primer sequences are added to the ends of each oligo
        pad_oligos: If True, spacer DNA is added between primer sequences and restrction
            binding sites to make all oligos the same length to reduce amplification bias.
        min_size: Minimum fragment size allowed in fragmentation patterns.
        output_dir: Filepath to where optimization output is saved to.
        nopt_steps: Number of steps to run each optimization run.
        nopt_runs: Number of times each pool is optimized. Each new optimization run
            uses a different random seed.
        njobs: Number of CPUs to use for simultaneous optimization runs. Each CPU executes
            one run.

    """
    #pylint: disable=too-many-arguments, too-many-locals

    # if output directory doesn't exist, write it
    if not exists(output_dir):
        os.makedirs(output_dir)

    assembly_enzyme = define_enzyme(enzyme)
    #! Right now this only recognizes other TypeIIS enzymes
    other_enzymes = [define_enzyme(e) for e in other_enzymes] if other_enzymes is not None else []

    # format input sequences into Gene objects
    input_seqs_recs = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(input_seqs, 'fasta')]
    other_used_sites = other_used_sites or np.array([])
    primers = PrimerIterator(primers, assembly_enzyme)
    ligation_data = define_ligation_data(ligation_data, assembly_enzyme)

    # print input parameters
    print(f"Read {len(input_seqs_recs)} from {input_seqs}.\nSequence lengths are {min(map(lambda x: len(x[1]), input_seqs_recs))} - {min(map(lambda x: len(x[1]), input_seqs_recs))} bp")
    print(f"Using {enzyme.name} and {ligation_data.name} data to assemble library.")
    print(f"Maximum number of GG sites allowed per pool: {njunctions}")
    print(f"Upstream backbone site: {upstream_bbsite}")
    print(f"Downstream backbone site: {downstream_bbsite}")


    #process one gene at a time, save one gene in one pool
    concat_lib_df = pd.DataFrame()
    concat_oligo_df = pd.DataFrame()
    concat_pool_stats_df = pd.DataFrame()
    optimize_lib_list = []
    oligo_list =[]
    pool_stats_list = []
    # split multiple genes in the fasta file into individual fasta file. one gene per fasta
    split_fasta_by_gene(input_seqs)
    splitgene_output_dir = "individual_genes"
    if os.path.exists(splitgene_output_dir):
        gene_files = [f for f in os.listdir(splitgene_output_dir) if f.endswith(".fasta") or f.endswith(".fa")] #get fasta files
    else:
        print(f"Output directory not found: {splitgene_output_dir}")

    for i in range(0,len(gene_files)):
        gene_file_dir = os.path.join(splitgene_output_dir, gene_files[i])
        one_gene_input = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(gene_file_dir, 'fasta')]
        library = Library(
            genes=one_gene_input,
            primers=primers,
            oligo_len=oligo_len,
            enzyme=assembly_enzyme,
            upstream_bbsite=upstream_bbsite,
            downstream_bbsite=downstream_bbsite,
            other_used_sites=other_used_sites,
            min_size=min_size
        )

        # assign optimization seeds - use nopt_runs to get random_opt seeds
        random_seeds = None
        if nopt_runs is not None:
            rng = np.random.default_rng(seed=42)
            random_seeds = rng.integers(1000, size=nopt_runs)
        elif (nopt_runs is None) and (opt_seeds is not None):
            random_seeds = opt_seeds
        else:
            raise ValueError('Cannot provide values for both `nopt_runs` and `opt_seeds`.')

        # assign genes to pools
        library.optimize_pools(
            njunctions=njunctions,
            nopt_steps=nopt_steps,
            opt_seeds=random_seeds,
            njobs=njobs,
            ligation_data=ligation_data.data,
            optimization=optimization
        )
        # save optimized gene fragment and empty oligo
        optimized_library = library.package_library(add_primers=add_primers, pad_oligo=pad_oligos)
        optimize_lib_list.append(optimized_library)

        oligopool = library.package_oligos(add_primers=add_primers, pad_oligo=pad_oligos)
        oligo_list.append(oligopool)

        pool_stats = pd.DataFrame.from_dict(
            [{
                'pool':i,
                'fidelity':fidelity,
                'min_gene':p.min_gene_fidelity,
                'min_site':p.min_site_fidelity,
                'n_genes':len(p.genes),
                'n_sites':(p.nfrags-1)*len(p.genes) + 2 + len(p.other_used_sites),
                'seed':seed,
                'enzyme':p.enzyme.name,
                'pfwd_name':'',
                'pfwd_sequence':'',
                'prev_name':'',
                'prev_sequence':''
            } for p, fidelity, seed in library.optimized_pools]
        )
        pool_stats_list.append(pool_stats)

        # if dev, save extra data
        if dev:
            # make trajectory save dir
            trajectory_dir = join(output_dir, 'opt_trajectories')
            if not exists(trajectory_dir):
                os.mkdir(trajectory_dir)
            for pool in library.all_opt_runs:
                # print(pool[0].opt_trajectory)
                np.save(join(trajectory_dir, f'pool_{pool[0].name}_seed-{pool[2]}.npy'), np.array(pool[0].opt_trajectory))

    concat_lib_df = pd.concat(optimize_lib_list)
    concat_lib_df.to_csv(os.path.join(output_dir, 'optimization_results.csv'))
    print(f"Finished optimization. Saved to {os.path.join(output_dir, 'optimization_results.csv')}")
    concat_oligo_df = pd.concat(oligo_list)
    concat_oligo_df.to_csv(os.path.join(output_dir, 'oligo_order.csv'))
    print(f"Designed {len(oligo_list)} oligos. Saved to {os.path.join(output_dir, 'oligo_order.csv')}")
    concat_pool_stats_df = pd.concat(pool_stats_list)
    concat_pool_stats_df.to_csv(join(output_dir, 'pool_stats.csv'))
    print(f"Pool-level statistics saved to {join(output_dir, 'pool_stats.csv')}")
    with open(join(output_dir, 'experiment_details.txt'), 'w') as f:
        f.write(ligation_data.experiment_information())

    # add primers now in the final concatenated df




def junctions(
        set_size: int,
        fixed_sites: Optional[list[str]],
        excluded_sites: Optional[list[str]],
        enzyme: Union[EnzymeTypes, Enzyme] = 'BsaI',
        ligation_data: LigationDataOpt = 'T4_01h_25C',
        output_dir: str = 'output',

        nopt_steps: int = 5000,
        nopt_runs: Optional[int] = None,
        opt_seeds: Optional[list[int]] = None,
        njobs: int = 1,

        dev: bool = False
) -> None:
    """
    TODO: Make this a better docstring
    Design library for pooled golden gate assembly.

    Args:
        input_seqs: File path to library sequences in fasta format.
        upstream_bbsite: GG site in str format if applicable.
        downstream_bbsite: GG site in str format if applicable.
        njunctions: Number of junctions used per pool - should include the upstream
            and downstream sites.
        other_used_sites: Other GG sites that are included in the assembly that are not
            upstream or downstream vector ligation sites.
        enzyme: Enzyme used to digest DNA fragments.
        ligation_data: Ligation data used to optimize the library.
        oligo_len: Oligo len used in oligopool.
        add_primers: If True, primer sequences are added to the ends of each oligo
        pad_oligos: If True, spacer DNA is added between primer sequences and restrction
            binding sites to make all oligos the same length to reduce amplification bias.
        min_size: Minimum fragment size allowed in fragmentation patterns.
        output_dir: Filepath to where optimization output is saved to.
        nopt_steps: Number of steps to run each optimization run.
        nopt_runs: Number of times each pool is optimized. Each new optimization run
            uses a different random seed.
        njobs: Number of CPUs to use for simultaneous optimization runs. Each CPU executes
            one run.

    """
    #pylint: disable=too-many-arguments, too-many-locals

    # if output directory doesn't exist, write it
    if not exists(output_dir):
        os.makedirs(output_dir)

    assembly_enzyme = define_enzyme(enzyme)
    ligation_data = define_ligation_data(ligation_data, assembly_enzyme)

    # assign optimization seeds - use nopt_runs to get random_opt seeds
    random_seeds = None
    if nopt_runs is not None:
        rng = np.random.default_rng(seed=42)
        random_seeds = rng.integers(1000, size=nopt_runs)
    elif (nopt_runs is None) and (opt_seeds is not None):
        random_seeds = opt_seeds
    else:
        raise ValueError('Cannot provide values for both `nopt_runs` and `opt_seeds`.')

    # set random seed
    random.seed(42)

    # optimize pools
    optimization_results = optimize_junctions(
        name=0,
        enzyme=assembly_enzyme,
        set_size=set_size,
        fixed_sites=fixed_sites,
        excluded_sites=excluded_sites,
        ligation_data=ligation_data.data,
        opt_seeds=random_seeds,
        nopt_steps=nopt_steps,
        njobs=njobs
    )

    # save best optimized set
    optimized_set, _, random_seed = optimization_results[-1]

    opt_stats = optimized_set.package_sites() | {'random_seed':random_seed}
    pd.DataFrame.from_dict([opt_stats]).to_csv(join(output_dir, 'optimized_junction_set.csv'))

    if dev:
        # make trajectory save dir
        trajectory_dir = join(output_dir, 'opt_trajectories')
        if not exists(trajectory_dir):
            os.mkdir(trajectory_dir)
        for jset, _, seed in optimization_results:
            np.save(join(trajectory_dir, f'set_{jset.name}_seed-{seed}.npy'), np.array(optimized_set.opt_trajectory))

if __name__ == "__main__":
    CLI([genes, one_gene_per_pool, junctions], as_positional=False)
