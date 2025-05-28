"""LibraryGene class."""

from typing import Union, Any, Optional
from itertools import count, chain
import random
import string
from math import ceil

from more_itertools import sliding_window, chunked, chunked_even
import pandas as pd
import numpy as np
from Bio.Seq import Seq
from tqdm import tqdm

from data_classes import Enzyme, PrimerIterator
from helpers import clean_dna, unique_orthogonal
from predict_fidelity import predict_fidelity, predict_minimum, predict_minimum_site, geneset_fidelity

from joblib import Parallel, delayed

#pylint:disable=too-many-instance-attributes
#pylint:disable=too-many-arguments


def get_coding_space(oligo_len: int, fprimer: str, rprimer: str, enzyme: Enzyme) -> int:
    """Estimate coding space in oligo that can be used for encoding constructs."""
    #pylint:disable=line-too-long

    return oligo_len - len(fprimer) - len(rprimer) - (enzyme.padding+len(enzyme.seq))*2 - enzyme.site_size

def optimize_pools(
    name: Any, genes: list[tuple[Any, str]], enzyme: Enzyme, upstream_bbsite: str,
    downstream_bbsite: str, other_used_sites: Union[list[str], None],
    forward_primer: Union[str, None], reverse_primer: Union[str, None], oligo_len: int,
    nfrags: int, ligation_data: np.ndarray, nopt_steps: int, random_seed: int, optimization: str
) -> tuple["Pool", float, int]:
    """Optimize a single subassembly pool. Called by Library."""

    if optimization == 'greedy':
        pool = Pool(
            name=name,
            genes=genes,
            enzyme=enzyme,
            upstream_bbsite=upstream_bbsite,
            downstream_bbsite=downstream_bbsite,
            other_used_sites=other_used_sites,
            forward_primer=forward_primer,
            reverse_primer=reverse_primer,
            oligo_len=oligo_len,
            nfrags=nfrags,
            ligation_data=ligation_data
        )

    elif optimization == 'simulated_annealing':
        pool = SAPool(
            name=name,
            genes=genes,
            enzyme=enzyme,
            upstream_bbsite=upstream_bbsite,
            downstream_bbsite=downstream_bbsite,
            other_used_sites=other_used_sites,
            forward_primer=forward_primer,
            reverse_primer=reverse_primer,
            oligo_len=oligo_len,
            nfrags=nfrags,
            ligation_data=ligation_data
        )

    else:
        raise ValueError('`optimization` must be `simulated_annealing` or `greedy`.')

    fidelity = pool.optimize(nopt_steps=nopt_steps, random_seed=int(random_seed), disable_progress=True)

    return pool, fidelity, random_seed


class Library:
    """The Library object coordinates the design of genes."""

    def __init__(
        self,
        genes: list[tuple[Any, str]],
        primers: PrimerIterator,
        oligo_len: int,
        enzyme: str,
        upstream_bbsite: str,
        downstream_bbsite: str,
        other_used_sites: Union[np.ndarray, None],
        min_size: int = 40
    ):

        self.genes = genes
        self.primers = primers.primers
        self.oligo_len = oligo_len
        self.enzyme = enzyme
        self.upstream_bbsite = upstream_bbsite
        self.downstream_bbsite = downstream_bbsite
        self.other_used_sites = other_used_sites or np.array([])
        self.min_size = min_size
        self.nfrags = self.estimate_nfrags()


        self._max_primer_space = self.__get_max_primer_space()

        self.optimized_pools = None
        self.all_opt_runs = None


    #? TODO: I think you want most of the library arguments down here
    #? TODO: Be careful - you have two things labelled 'optimize_pools' - you need to change that.
    def optimize_pools(
            self, nopt_steps: int, njobs: int, njunctions: int, ligation_data: pd.DataFrame,
            optimization: str, nopt_runs: Optional[int] = None, opt_seeds: Optional[list[int]] = None
    ) -> list["Pool"]:
        """Optimize each pool in the library. Each pool is run for the number of nopt_runs
        with each run using a different random seed.
        
        Args:
            nopt_steps: Number of steps to run the optimization.
            nopt_runs: Number of times to optimize each pool.
            njobs: Number of CPUs to use for separate optimization runs.
            njunctions: Number of junctions used per pool - should include the upstream
                and downstream sites.
            ligation_data: DataFrame containing ligation freq data for all GG sites. Index
                and Columns should be str formatted GG sites.
        """
        #pylint:disable=line-too-long

        ngenes_per_pool = (njunctions - len([self.upstream_bbsite, self.downstream_bbsite]) - len(self.other_used_sites)) // (self.estimate_nfrags() - 1)
        npools = ceil(len(self.genes) / ngenes_per_pool)

        print(f"Target number of genes per pool: {ngenes_per_pool} genes assembled in {npools} pools.")
        print(f"Genes are broken into {self.estimate_nfrags()} fragments.")

        # if not enough primers for estimated pools, raise error
        if len(self.primers) < (len(self.genes)/ngenes_per_pool):
            raise ValueError('Not enough primers for estimated number of pools.')

        # run optimization for each pool
        it = zip(
            count(0),
            chunked_even(self.genes, ngenes_per_pool),
            self.primers
        )
        optimized = []
        #? fix this - make how you save opt trajectories better
        all_opt_runs = []
        print('Optimizing library...')
        for i, gene_pool, (pfor, prev) in tqdm(it, total=npools, ncols=100):
            opt_results = Parallel(n_jobs=njobs)(delayed(optimize_pools)(
                i,
                gene_pool,
                self.enzyme,
                self.upstream_bbsite,
                self.downstream_bbsite,
                self.other_used_sites,
                pfor,
                prev,
                self.oligo_len,
                self.nfrags,
                ligation_data,
                nopt_steps,
                rand_seed,
                optimization
            ) for rand_seed in opt_seeds)

            opt_results.sort(key=lambda x: x[1])
            # save all opt runs
            all_opt_runs.extend(opt_results)
            # save only best for each pool
            optimized.append(opt_results[-1])


        self.all_opt_runs = all_opt_runs
        self.optimized_pools = optimized


    def estimate_nfrags(self) -> int:
        """Esitmate the number of fragments the longest gene has to be broken into.
        This value is used to fragment all genes in the library into the same number.
        This is to ward against variable assembly efficiencies due to fragment number.

        Returns: integer indicating number of fragments library should be broken into.
        """

        # get longest pairs of primers - they should be the same length, but in case they're not...
        longest_primers = ["", ""]
        for pfor, prev in self.primers:
            if sum(map(len, [pfor.sequence, prev.sequence])) > sum(map(len, longest_primers)):
                longest_primers = [pfor.sequence, prev.sequence]
            else:
                continue

        # identify longest sequence length to fragment
        longest_gene = max(map(len, [g[1] for g in self.genes]))
        # subtract some bp from oligo_len so we do not get sequences that pefectly fit oligo_len
        coding_space = get_coding_space(
            self.oligo_len-24,
            fprimer=longest_primers[0],
            rprimer=longest_primers[1],
            enzyme=self.enzyme
        )

        for nfrags in count(1):
            if longest_gene // nfrags <= coding_space:
                return nfrags

        return None


    def package_library(self, add_primers: bool = True, pad_oligo: bool = True) -> pd.DataFrame:
        """Get optimization output for all pools in library."""
        #pylint:disable=line-too-long

        output = pd.concat(
            [p[0].package_pool(add_primers=add_primers, pad_oligo=pad_oligo) for p in self.optimized_pools],
            ignore_index=True
        )

        return output

    def package_oligos(self, add_primers: bool = True, pad_oligo: bool = True) -> pd.DataFrame:
        """Get oligo output for all pools in library."""
        #pylint:disable=line-too-long

        output = pd.concat(
            [p[0].package_oligos(add_primers=add_primers, pad_oligo=pad_oligo) for p in self.optimized_pools],
            ignore_index=True
        )

        return output

    def __get_max_primer_space(self) -> int:
        """Return largest space taken up by a primer pair in oligo"""

        return np.array(
            sum([len(pfor.sequence), len(prev.sequence)]) for pfor, prev in zip(self.primers)
        )


class Pool:
    """Class object to design GG sites for subassembly pool."""

    def __init__(
        self,
        name: Any,
        genes: list[tuple[Any, str]],
        enzyme: Enzyme,
        upstream_bbsite: str,
        downstream_bbsite: str,
        other_used_sites: Union[list[str], None],
        forward_primer: Union[str, None],
        reverse_primer: Union[str, None],
        oligo_len,
        nfrags,
        ligation_data
    ):
        self.name = name
        self.enzyme = enzyme
        self.upstream_bbsite = upstream_bbsite
        self.downstream_bbsite = downstream_bbsite
        self.other_used_sites = other_used_sites or []  # TODO: may need to change this to np.array([])
        self.fprimer = forward_primer
        self.rprimer = reverse_primer
        self.oligo_len = oligo_len
        self.nfrags = nfrags
        self.ligation_data = ligation_data

        self.genes = self.__instantiate_genes(genes)
        self.__assign_start_sites()


        # optimization information
        self.optimized_sites = None
        self.opt_trajectory = None
        self.optimized_fidelity = None

    def __instantiate_genes(self, genes) -> list["Gene"]:
        """Instantiate library sequences as Gene objects for optimization."""

        inst_genes = [
            Gene(
                i, g,
                self.upstream_bbsite,
                self.downstream_bbsite,
                self.other_used_sites,
                self.enzyme,
                self.fprimer,
                self.rprimer,
                self.oligo_len,
            ) for i,g in genes
        ]

        return inst_genes

    def __assign_start_sites(self) -> None:
        """Randomize starting GG sites for genes in subassembly pool."""
        #pylint:disable=line-too-long
        random.seed(42)

        candidates = list(chain(*[g.get_start_sites_range(self.nfrags) for g in self.genes]))
        selected_sites = np.array(
            [random.choice(c) for c in candidates]
        )

        for _ in count():

            change_idx = random.choice(range(len(candidates)))
            new_sites = np.copy(selected_sites)
            new = random.choice(candidates[change_idx])
            new_sites[change_idx] = new
            wc_new_sites = [str(Seq(s).reverse_complement()) for s in new_sites[:,0]]

            if len(set(new_sites[:,0].tolist() + wc_new_sites)) >= len(set(selected_sites[:,0].tolist() + [str(Seq(s).reverse_complement()) for s in selected_sites[:,0]])):
                selected_sites = new_sites

            if unique_orthogonal(selected_sites[:,0]):
                for sites, g in zip(chunked(selected_sites, self.nfrags-1), self.genes):
                    # need to make pos integer
                    g.assigned_sites = pd.DataFrame.from_dict([{'ggsite':s[0], 'pos':int(s[1])} for s in sites])
                break



    def optimize(self, nopt_steps: int, random_seed: int, disable_progress: bool = False) -> float:
        """Optimize GG site usage to find orthogonal site options."""

        random.seed(int(random_seed))

        current_state: list[pd.DataFrame] = [g.assigned_sites for g in self.genes]
        # best_state=np.hstack([g.assigned_sites.pos.to_numpy() for i, g in enumerate(self.genes)])

        # start optimization trajectory with starting state of subassembly pool
        opt_trajectory = [
            predict_fidelity(
                np.hstack([
                    np.hstack([df.ggsite.to_numpy() for df in current_state]),
                    np.array([self.upstream_bbsite, self.downstream_bbsite])
                ]),
                self.ligation_data
            )
        ]

        for _ in tqdm(range(nopt_steps), ncols=100, total=nopt_steps, disable=disable_progress, leave=True):

            change_idx: int = random.choice(range(len(self.genes)))
            unchanged_pool_sites: np.ndarray[str] = np.hstack([g.assigned_sites.ggsite.to_numpy() for i, g in enumerate(self.genes) if i != change_idx])
            candidates: pd.DataFrame = self.genes[change_idx].shuffle_site(pool_ggsites=unchanged_pool_sites, min_dist=40)
            
            used_sites = np.hstack([
                unchanged_pool_sites,
                candidates.ggsite.to_numpy(),
                np.array([self.upstream_bbsite, self.downstream_bbsite]),
                np.array(self.other_used_sites)
            ])

            candidate_fidelity = predict_fidelity(used_sites, self.ligation_data)

            # we are doing a greedy optimization - if better, accept
            if candidate_fidelity >= opt_trajectory[-1]:
                self.genes[change_idx].assigned_sites = candidates  # update sites
                current_state = [g.assigned_sites for g in self.genes]
                opt_trajectory.append(candidate_fidelity)
            else:
                opt_trajectory.append(opt_trajectory[-1])  # use previous fidelity

        self.optimized_sites = current_state
        self.opt_trajectory = opt_trajectory
        self.optimized_fidelity = predict_fidelity(
            np.hstack([
                np.hstack([df.ggsite.to_numpy() for df in current_state]),
                np.array([self.upstream_bbsite, self.downstream_bbsite] + self.other_used_sites)
            ]),
            self.ligation_data
        )
        self.min_gene_fidelity = predict_minimum(
            [df.ggsite.to_numpy() for df in self.optimized_sites] + [self.upstream_bbsite, self.downstream_bbsite],
            self.ligation_data
        )
        self.min_site_fidelity = predict_minimum_site(
            [np.array([df.ggsite.to_numpy() for df in self.optimized_sites]).flatten()] + [self.upstream_bbsite, self.downstream_bbsite],
            self.ligation_data
        )

        return self.optimized_fidelity

    def shuffle_site(self) -> pd.DataFrame:
        """Randomly select gene and shuffle GG site selections for that gene.
        Change only one GG site at a time.
        """

        gene_idx = random.choice(range(len(self.genes)))

        pool_sites = np.hstack(
            [g.assigned_sites.pos.to_numpy() for i, g in enumerate(self.genes) if i != gene_idx]
        )
        new_candidates = self.genes[gene_idx].shuffle_site(pool_ggsites=pool_sites, min_dist=40)

        return new_candidates


    def package_oligos(self, add_primers: bool = True, pad_oligo: bool = True) -> pd.DataFrame:
        """Format genes into fragmented oligos ready to order."""

        oligos = []
        for gene in self.genes:
            for oligo, letter in zip(gene.get_oligos(add_primers=add_primers,pad_oligo=pad_oligo), string.ascii_uppercase):
                oligos.append({'name':f"{gene.name}_{letter}", 'sequence':oligo})

        return pd.DataFrame.from_dict(oligos)

    def package_pool(self, add_primers: bool = True, pad_oligo: bool = True) -> pd.DataFrame:
        """Return optimization output for pool."""

        min_gene_fidelity = predict_minimum(
            [df.ggsite.to_numpy() for df in self.optimized_sites] + [self.upstream_bbsite, self.downstream_bbsite],
            self.ligation_data
        )
        min_site_fidelity = predict_minimum_site(
            [np.array([df.ggsite.to_numpy() for df in self.optimized_sites]).flatten()] + [self.upstream_bbsite, self.downstream_bbsite],
            self.ligation_data
        )
        print('calling')
        outputs = []
        for gene in self.genes:
            gene_output = gene.package_gene(add_primers=add_primers, pad_oligo=pad_oligo)
            gene_output = gene_output | {'pool_id':self.name, 'fidelity':self.optimized_fidelity, 'min_gene_fidelity':min_gene_fidelity, 'min_site_fidelity':min_site_fidelity}
            outputs.append(gene_output)

        return pd.DataFrame.from_dict(outputs)


class SAPool:
    """Class object to design GG sites for subassembly pool."""

    def __init__(
        self,
        name: Any,
        genes: list[tuple[Any, str]],
        enzyme: Enzyme,
        upstream_bbsite: str,
        downstream_bbsite: str,
        other_used_sites: Union[list[str], None],
        forward_primer: Union[str, None],
        reverse_primer: Union[str, None],
        oligo_len,
        nfrags,
        ligation_data
    ):
        self.name = name
        self.enzyme = enzyme
        self.upstream_bbsite = upstream_bbsite
        self.downstream_bbsite = downstream_bbsite
        self.other_used_sites = other_used_sites or []  # TODO: may need to change this to np.array([])
        self.fprimer = forward_primer
        self.rprimer = reverse_primer
        self.oligo_len = oligo_len
        self.nfrags = nfrags
        self.ligation_data = ligation_data

        self.genes = self.__instantiate_genes(genes)
        self.__assign_start_sites()


        # optimization information
        self.optimized_sites = None
        self.opt_trajectory = None
        self.optimized_fidelity = None
        self.min_gene_fidelity = None
        self.min_site_fidelity = None

    def __instantiate_genes(self, genes) -> list["Gene"]:
        """Instantiate library sequences as Gene objects for optimization."""

        inst_genes = [
            Gene(
                i, g,
                self.upstream_bbsite,
                self.downstream_bbsite,
                self.other_used_sites,
                self.enzyme,
                self.fprimer,
                self.rprimer,
                self.oligo_len,
            ) for i,g in genes
        ]

        return inst_genes


    def __assign_start_sites(self) -> None:
        """Randomize starting GG sites for genes in subassembly pool."""
        #pylint:disable=line-too-long
        random.seed(42)

        candidates = list(chain(*[g.get_start_sites_range(self.nfrags) for g in self.genes]))
        selected_sites = np.array(
            [random.choice(c) for c in candidates]
        )

        for _ in count():

            change_idx = random.choice(range(len(candidates)))
            new_sites = np.copy(selected_sites)
            new = random.choice(candidates[change_idx])
            new_sites[change_idx] = new
            wc_new_sites = [str(Seq(s).reverse_complement()) for s in new_sites[:,0]]

            if len(set(new_sites[:,0].tolist() + wc_new_sites)) >= len(set(selected_sites[:,0].tolist() + [str(Seq(s).reverse_complement()) for s in selected_sites[:,0]])):
                selected_sites = new_sites

            if unique_orthogonal(selected_sites[:,0]):
                for sites, g in zip(chunked(selected_sites, self.nfrags-1), self.genes):
                    # need to make pos integer
                    g.assigned_sites = pd.DataFrame.from_dict([{'ggsite':s[0], 'pos':int(s[1])} for s in sites])
                break


    def optimize(self, nopt_steps: int, random_seed: int, disable_progress: bool = False) -> float:
        """Optimize GG site usage to find orthogonal site options."""

        random.seed(random_seed)
        start_temp: float = 0.005
        end_temp: float = .00001

        # this is a dict of genes where the key is the gene index
        current_state: list[pd.DataFrame] = {i:g.assigned_sites for i,g in enumerate(self.genes)}
        best_state: list[pd.DataFrame] = {i:g.assigned_sites for i,g in enumerate(self.genes)}

        # start optimization trajectory with starting state of subassembly pool
        start_fidelity = predict_fidelity(
                np.hstack([
                    np.hstack([df.ggsite.to_numpy() for df in list(current_state.values())]),
                    np.array([self.upstream_bbsite, self.downstream_bbsite])
                ]),
                self.ligation_data
            )
        opt_trajectory = [(start_fidelity, start_fidelity, 0, 0, 0, 0, start_temp)]  # (best, current, candidate fidelity, energy_diff, contender, thing to beat, temp)

        # change log range
        for temp in tqdm(np.linspace(start_temp, end_temp, nopt_steps), ncols=100, total=nopt_steps, disable=disable_progress, leave=True):

            while True:
                change_idx: int = random.choice(range(len(self.genes)))
                # pool sites from the genes
                unchanged_pool_sites: np.ndarray[str] = np.hstack([g.assigned_sites.ggsite.to_numpy() for i, g in enumerate(self.genes) if i != change_idx])
                # these are a new set of GG sites from change_idx gene - they have to actually be assigned to the gene
                candidates: pd.DataFrame = self.genes[change_idx].shuffle_site(pool_ggsites=unchanged_pool_sites, min_dist=40)

                # if shuffle sites fails, pick different gene to optimize
                if candidates is not None:
                    break
            
            
            used_sites = np.hstack([
                unchanged_pool_sites,
                candidates.ggsite.to_numpy(),
                np.array([self.upstream_bbsite, self.downstream_bbsite]),
                np.array(self.other_used_sites)
            ])

            candidate_fidelity = predict_fidelity(used_sites.tolist(), self.ligation_data)
            energy_diff = candidate_fidelity - opt_trajectory[-1][1]
            thing_to_beat = random.random()
            contender = np.exp(min([0, energy_diff / temp]))

            if contender > thing_to_beat:

                # if we've gotten too far away from the best, then reset to the best
                if (opt_trajectory[-1][0] - candidate_fidelity) > 0.01:
                    current_state = best_state
                    for i, ggset in best_state.items():
                        self.genes[i].assigned_sites = ggset

                    # append the fidelity - best and current are now best, candidate is what triggered reset to best
                    opt_trajectory.append((opt_trajectory[-1][0], opt_trajectory[-1][0], candidate_fidelity, energy_diff, contender, thing_to_beat, temp))

                else:
                    self.genes[change_idx].assigned_sites = candidates
                    current_state = {i:g.assigned_sites for i,g in enumerate(self.genes)}
                
                    # if new fidelity is better than the best fidelity, update best fidelity
                    if candidate_fidelity > opt_trajectory[-1][0]:
                        best_state = current_state

                        opt_trajectory.append((candidate_fidelity, candidate_fidelity, candidate_fidelity, energy_diff, contender, thing_to_beat, temp))
                    else:
                        # don't update best, so use old version
                        opt_trajectory.append((opt_trajectory[-1][0], candidate_fidelity, candidate_fidelity, energy_diff, contender, thing_to_beat, temp))
            
            # if change is rejected, do nothing except update trajectory with old fidelity
            else:
                opt_trajectory.append([opt_trajectory[-1][0], opt_trajectory[-1][1], candidate_fidelity, energy_diff, contender, thing_to_beat, temp])


        self.optimized_sites = best_state
        self.opt_trajectory = opt_trajectory
        self.optimized_fidelity = predict_fidelity(
            np.hstack([
                np.hstack([df.ggsite.to_numpy() for df in list(best_state.values())]),
                np.array([self.upstream_bbsite, self.downstream_bbsite] + self.other_used_sites)
            ]),
            self.ligation_data
        )

        self.min_gene_fidelity = predict_minimum(
            [df.ggsite.tolist() for df in list(self.optimized_sites.values())] + [[self.upstream_bbsite], [self.downstream_bbsite]],
            self.ligation_data
        )
        self.min_site_fidelity = predict_minimum_site(
            list(chain(*[df.ggsite.tolist() for df in list(self.optimized_sites.values())])) + [self.upstream_bbsite, self.downstream_bbsite],
            self.ligation_data
        )


        return self.optimized_fidelity

    def shuffle_site(self) -> pd.DataFrame:
        """Randomly select gene and shuffle GG site selections for that gene.
        Change only one GG site at a time.
        """

        gene_idx = random.choice(range(len(self.genes)))

        pool_sites = np.hstack(
            [g.assigned_sites.pos.to_numpy() for i, g in enumerate(self.genes) if i != gene_idx]
        )
        new_candidates = self.genes[gene_idx].shuffle_site(pool_ggsites=pool_sites, min_dist=40)

        return new_candidates


    def package_oligos(self, add_primers: bool = True, pad_oligo: bool = True) -> pd.DataFrame:
        """Format genes into fragmented oligos ready to order."""

        oligos = []
        for gene in self.genes:
            for oligo, letter in zip(gene.get_oligos(add_primers=add_primers,pad_oligo=pad_oligo), string.ascii_uppercase):
                oligos.append({'name':f"{gene.name}_{letter}", 'sequence':oligo})

        return pd.DataFrame.from_dict(oligos)

    def package_pool(self, add_primers: bool = True, pad_oligo: bool = True) -> pd.DataFrame:
        """Return optimization output for pool."""

        outputs = []
        for gene in self.genes:
            gene_output = gene.package_gene(add_primers=add_primers, pad_oligo=pad_oligo)
            gene_output = gene_output | {'pool_id':self.name, 'fidelity':self.optimized_fidelity, 'min_gene_fidelity':self.min_gene_fidelity, 'min_site_fidelity':self.min_site_fidelity}
            outputs.append(gene_output)

        # add support for getting individual gene fidelities
        outputs = pd.DataFrame.from_dict(outputs)
        gg_cols = outputs.columns[outputs.columns.str.startswith('ggsite_')]
        gene_ggsites = [row.loc[gg_cols].tolist() for _, row in outputs.iterrows()] + [[self.upstream_bbsite, self.downstream_bbsite] + self.other_used_sites]
        print(gene_ggsites)
        gene_fidelities = geneset_fidelity(gene_ggsites, self.ligation_data)
        outputs['gene_fidelity'] = gene_fidelities[:-1]

        return outputs



class Gene:
    """Class object to hold construct information."""

    def __init__(
        self,
        name: Any,
        sequence: str,
        upstream_bbsite: str,
        downstream_bbsite: str,
        other_used_sites: Union[list[str], None],
        enzyme: Enzyme,
        forward_primer: str,
        reverse_primer: str,
        oligo_len: int
    ):
        self.name = name
        self.seq = sequence.upper()
        self.enzyme = enzyme
        self.upstream_bbsite = upstream_bbsite
        self.downstream_bbsite = downstream_bbsite
        self.other_used_sites = other_used_sites or []
        self.oligo_len = oligo_len

        self.fprimer = forward_primer
        self.rprimer = reverse_primer

        self.coding_space = get_coding_space(
            self.oligo_len, self.fprimer.sequence, self.rprimer.sequence, self.enzyme
        )

        self.ggsite_options = self.__populate_sites()
        self.assigned_sites = None

    def __populate_sites(self) -> pd.DataFrame:
        """Find all possible GG site options that are compatible with fragmentation constraints."""
        #pylint:disable=line-too-long

        ggsites = pd.DataFrame.from_dict([
            {'ggsite':"".join(ggsite),'pos':i} for i, ggsite in enumerate(sliding_window(self.seq, self.enzyme.site_size))
        ])
        disallowed_sites = np.hstack([
            np.array([self.upstream_bbsite, self.downstream_bbsite]),
            np.array(self.other_used_sites),
            np.array([str(Seq(self.upstream_bbsite).reverse_complement()), str(Seq(self.downstream_bbsite).reverse_complement())]),
            np.array([str(Seq(s).reverse_complement()) for s in self.other_used_sites]),
            np.array(['GCGC', 'CGCG', 'ATAT', 'TATA', 'GGCC', 'GGCC', 'AATT', 'TTAA', 'TGCA', 'AGCT','TCGA', 'ACGT', 'GATC', 'GTAC', 'CATG', 'CTAG'])
        ])
        # print(disallowed_sites)
        # retain sites that are no backbone or disallowed sites
        ggsites = ggsites.loc[~ggsites.ggsite.isin(disallowed_sites)]
        # print(ggsites)
        return ggsites

    def get_start_sites_range(self, nfrags: int) -> list[tuple]:
        """Provides list of GG sites that fit within the intial breakpoint windows."""
        #pylint:disable=line-too-long

        gene_length = len(self.seq)

        # fragment size
        step_size = ceil(gene_length / nfrags)
        centroids = np.hstack([
            np.array([0]),
            np.array(list(range(step_size, gene_length, step_size))),
            np.array([gene_length-1])
        ])
        biggest_diff = np.diff(centroids).max()
        half_step = (self.coding_space - biggest_diff) // 2  # rounds down

        start_candidates = []
        for c in centroids[1:-1]:
            start_candidates.append(self.ggsite_options.set_index('pos', drop=False).loc[c-half_step:c+half_step].sample(frac=1.0, random_state=42))

        return [list(zip(df.loc[:,'ggsite'], df.loc[:,'pos'])) for df in start_candidates]

    def shuffle_site(self, pool_ggsites: np.ndarray, min_dist: int):
        """Return pd.DataFrame of site candidates."""

        # make sure that assigned sites are in order
        self.assigned_sites.sort_values('pos', ascending=True, inplace=True)

        # try to change sites 10 times before giving up:
        for _ in range(50):
            # print(self.assigned_sites)
            change_loc = random.choice(range(self.assigned_sites.shape[0]))
            keep_sites = self.assigned_sites.iloc[
                [i for i in range(self.assigned_sites.shape[0]) if i != change_loc]
            ]

            sections = np.split(np.arange(len(self.seq)), keep_sites.pos.to_numpy())
            longest_section = np.argmax(np.array(list(map(len, sections))))

            left_site_pos = sections[longest_section][0]
            right_site_pos = sections[longest_section][-1]+1

            allowed_left_pos = np.arange(left_site_pos+min_dist, left_site_pos+self.coding_space-4)
            allowed_right_pos = np.arange(right_site_pos-self.coding_space, right_site_pos-min_dist)  #NOTE I may be missing a + or - 4 here

            permissible_pos = np.intersect1d(allowed_left_pos, allowed_right_pos)


            try:
                candidates = self.ggsite_options[self.ggsite_options.pos.between(permissible_pos[0], permissible_pos[-1])]
                
                # remove any used sites - make sure to remove the WC pair, too - makes selection more efficient
                used_sites = np.hstack([
                    np.array([self.upstream_bbsite, self.downstream_bbsite]),
                    np.array([str(Seq(self.upstream_bbsite).reverse_complement()), str(Seq(self.downstream_bbsite).reverse_complement())]),
                    keep_sites.ggsite.to_numpy(),
                    np.array([str(Seq(s).reverse_complement()) for s in keep_sites.ggsite]),
                    pool_ggsites,
                    np.array([str(Seq(s).reverse_complement()) for s in pool_ggsites])
                ])
                # print('new')
                # print(candidates[~candidates.ggsite.isin(used_sites)])
                # print(candidates[~candidates.ggsite.isin(used_sites)].shape)
                candidates = candidates[~candidates.ggsite.isin(used_sites)]
                candidate = candidates.sample(n=1, random_state=42)
                return pd.concat([keep_sites, candidate])
            except:
                continue


    def get_oligos(self, add_primers: bool = True, pad_oligo: bool = True):
        """Get oligos for gene.
        
        Args:
            primers: indicate whether primers should be added to the oligo sequence.
                Recommended for most cases unless you are adding primers yourself.
            pad_oligo: indicate whether random DNA sequence should be added between
                restriction sites and primer sites to make all oligos the same length.
                Recommended to avoid PCR bias.

        Returns:
            List of oligos.
        """
        if self.gene_assembles():
            frag_gene = self.__fragment_gene()
            frag_gene = [self.__add_restriction_sites(f) for f in frag_gene]

        if add_primers:
            frag_gene = [self.__add_primers(f, pad_oligo) for f in frag_gene]

        return frag_gene

    def package_gene(self, add_primers: bool = True, pad_oligo: bool = True):
        """Package gene optmization results."""

        output = {'gene_id':self.name, 'sequence':self.seq}
        oligos = {f'oligo_{i}':oligo for i, oligo in enumerate(self.get_oligos(add_primers, pad_oligo))}
        bbsites = {'upstream_bbsite':self.upstream_bbsite, 'downstream_bbsite':self.downstream_bbsite}
        extra_sites = {f'extra_ggsite_{i}':s for i, s in enumerate(self.other_used_sites)}
        ggsites = ggsites = {f'ggsite_{i}':site for i, site in enumerate(self.assigned_sites.sort_values('pos', ascending=True).ggsite)}
        primers = {'fwd_primer':self.fprimer.sequence, 'rev_primer':self.rprimer.sequence}

        return output | oligos | bbsites | ggsites | extra_sites | primers

    def __fragment_gene(self) -> list[str]:
        """Break gene into fragments based on optimized sites. Adds in GG sites
        Required to assemble all fragments together including any backbone sites.
        """

        broken_gene = np.array_split(
            np.array(list(self.seq)),
            self.assigned_sites.sort_values('pos', ascending=True).pos.to_numpy()
        )

        fragments = []
        for frag, site in zip(broken_gene, self.assigned_sites.sort_values('pos', ascending=True).ggsite.to_numpy()):
            fragments.append("".join(frag) + site)

        # last fragment is not added because there are fewer sites than number of fragments - do this now
        fragments.append("".join(broken_gene[-1].tolist()))

        # Add in golden gate sites for backbone ligation only to first and last fragments
        fragments[0] = self.upstream_bbsite + fragments[0]
        fragments[-1] = fragments[-1] + self.downstream_bbsite

        return fragments

    def __add_restriction_sites(self, oligo: str, pad_bp: str = 'A') -> str:
        """Regurn oligo with restriction sites placed on either side.
        
        FIXME: right now this needs to be called after you add in all GG sites.
        """
        forward_site = self.enzyme.seq + pad_bp*self.enzyme.padding
        reverse_site = pad_bp*self.enzyme.padding + self.enzyme.revc_seq

        return forward_site + oligo + reverse_site

    def __add_primers(self, oligo: str, pad_oligo: bool = True) -> str:
        """Return oligo with forward and reverse primers attached."""
        #pylint: disable=unbalanced-tuple-unpacking

        extra_space = self.oligo_len - len(self.fprimer.sequence) - len(self.rprimer.sequence) - len(oligo)
        left, right = np.array_split(np.arange(extra_space), [extra_space//2])

        if pad_oligo:
            return "".join([self.fprimer.sequence,
                            clean_dna(left.size, self.enzyme.seq).lower(),
                            oligo,
                            clean_dna(right.size, self.enzyme.seq).lower(),
                            str(Seq(self.rprimer.sequence).reverse_complement())])

        else:
            return "".join([self.fprimer.sequence,
                            oligo,
                            str(Seq(self.rprimer.sequence).reverse_complement())])


    def gene_assembles(self) -> bool:
        frag_gene = self.__fragment_gene()
        assembled = frag_gene[0]

        for frag in frag_gene[1:]:
            # check that site matches
            if  assembled[-self.enzyme.site_size:] == frag[:self.enzyme.site_size]:
                assembled = assembled + frag[self.enzyme.site_size:]
            else:
                raise ValueError('Golden Gate sites do not match between fragments.')

        # check that the AA sequence for assembled DNA is the same as the original AA sequence
        assembled_aas = str(Seq(assembled[self.enzyme.site_size:-self.enzyme.site_size]).translate())
        if assembled_aas == str(Seq(self.seq).translate()):
            return True
        else:
            raise ValueError('Amino acid sequence from assembled gene does not match \
                              the original AA sequence.')
