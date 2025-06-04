"""Special types for GG optimization."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any
from os.path import join

import pandas as pd
from Bio.Seq import Seq

from helpers import dna_contains_seq
from constants_v import LIGATION_DATA, ROOT_DIR


class EnzymeTypes(Enum):
    """List acceptable enzyme types."""
    # pylint:disable=invalid-name

    BsaI = 'BsaI'
    BbsI = 'BbsI'
    BsmBI = 'BsmBI'
    Esp3I = 'Esp3I'
    # SapI = 'SapI'

@dataclass
class Enzyme:
    """Container class for enzyme."""

    name: str
    seq: str
    revc_seq: str
    site_size: int
    padding: int

def define_enzyme(enzyme_type: Optional[EnzymeTypes] = None) -> Enzyme:
    """Get enzyme data to use for library optimization."""

    if enzyme_type.value == 'BsaI':
        return Enzyme('BsaI','GGTCTC',str(Seq('GGTCTC').reverse_complement()),4,1)
    if enzyme_type.value == 'BbsI':
        return Enzyme('BbsI','GAAGAC',str(Seq('GAAGAC').reverse_complement()),4,2)
    if enzyme_type.value == 'BsmBI':
        return Enzyme('BsmBI','CGTCTC',str(Seq('CGTCTC').reverse_complement()),4,1)
    if enzyme_type.value == 'Esp3I':
        return Enzyme('Esp3I','CGTCTC',str(Seq('CGTCTC').reverse_complement()),4,1)
    if enzyme_type.value == 'SapI':
        return Enzyme('SapI','GCTCTTC',str(Seq('GCTCTTC').reverse_complement()),3,1)

    return None

@dataclass
class Primer:
    """Container class for primer data."""

    name: str
    sequence: str
    forward: bool

class PrimerIterator:
    #pylint:disable=too-few-public-methods
    """Container for primers to use in subpool amplification."""

    def __init__(self, file_path: str, *enzymes: Enzyme) -> "PrimerIterator":
        df = pd.read_csv(file_path)

        # check that dataframe has proper columns
        if not set(['fwd_name','fwd_sequence','rev_name','rev_sequence']) <= set(df.columns):
            raise ValueError('Primer sequences missing required columns. Please ensure' +
                             'file contains `fwd_name`, `fwd_sequence`, `rev_name`, `rev_sequence`')

        self.primers = self.get_primers(df, *enzymes)

    def get_primers(self, df: pd.DataFrame, *enzymes):
        """Get list of primers that don't contain enzyme sequences
        used during cloning.
        """

        primers = []
        for _, row in df.iterrows():
            fwd_bad = dna_contains_seq(row.fwd_sequence, *[e.seq for e in enzymes])
            rev_bad = dna_contains_seq(row.rev_sequence, *[e.seq for e in enzymes])

            if not fwd_bad and not rev_bad:
                fprimer = Primer(row.fwd_name, row.fwd_sequence, True)
                rprimer = Primer(row.rev_name, row.rev_sequence, False)
                primers.append([fprimer, rprimer])

        return primers

class LigationDataOpt(Enum):
    """Set accepted options for ligation data."""
    #pylint:disable=invalid-name

    # PMID: 30335370
    T4_01h_25C = 'T4_01h_25C'
    T4_18h_25C = 'T4_18h_25C'
    T4_01h_37C = 'T4_01h_37C'
    T4_18h_37C = 'T4_18h_37C'

    # PMID: 32877448
    BsaI_cycling = 'BsaI_cycling'
    BbsI_cycling = 'BbsI_cycling'
    BsmBI_cycling = 'BsmBI_cycling'
    Esp3I_cycling = 'Esp3I_cycling'
    # SapI_cycling = 'SapI_cycling'


@dataclass
class ExperimentConditions:
    """Dataclass to hold details on experimental conditions used to collect ligation data."""

    enzyme: str
    ligase: str
    buffer: str
    incubation_time: str
    incubation_temperature: str
    site_size: int

class LigationData:
    """Container class for holding ligation data."""

    def __init__(
        self,
        name: str,
        file_path: str,
        conditions: ExperimentConditions,
        assembly_enzyme: Enzyme,
        pmid: Any

    )-> "LigationData":

        self.name = name
        self.data = pd.read_csv(join(ROOT_DIR, file_path), index_col=0)
        self.assembly_enzyme = assembly_enzyme
        self.experiment_conditions = conditions
        self.pmid = pmid

        if not self.compatible():
            raise ValueError('Ligation data and assembly enzyme are not compatible. Please check.')

    def compatible(self) -> bool:
        """Ensure that ligation data is appropriate for enzyme used in library assembly."""

        # if using enzyme, make sure it's compatible with the data
        if self.experiment_conditions.enzyme is not None:
            return self.experiment_conditions.enzyme == self.assembly_enzyme.name

        # if not, then make sure the site size is compatible with the enzyme
        else:
            return self.experiment_conditions.site_size == self.assembly_enzyme.site_size


    def experiment_information(self) -> str:
        """Print experiment data."""

        conditions = f"""
Enzyme: {self.assembly_enzyme.name}
Ligase: {self.experiment_conditions.ligase}
Buffer: {self.experiment_conditions.buffer}
Incubation temperature: {self.experiment_conditions.incubation_temperature}
Incubation time: {self.experiment_conditions.incubation_time}

Please see GitHub or OMEGA paper for more detailed experimental conditions.
        """
        return conditions



def define_ligation_data(data: LigationDataOpt, assembly_enzyme: Enzyme) -> LigationData:
    """Retrieve ligation data information."""

    # get str value of Enum type
    data = data.value
    conditions = ExperimentConditions(
        enzyme=LIGATION_DATA[data]['enzyme'],
        ligase=LIGATION_DATA[data]['ligase'],
        buffer=LIGATION_DATA[data]['buffer'],
        incubation_time=LIGATION_DATA[data]['incubation_time'],
        incubation_temperature=LIGATION_DATA[data]['incubation_temperature'],
        site_size=LIGATION_DATA[data]['site_size'],
    )

    return LigationData(
        name = data,
        file_path=LIGATION_DATA[data]['file_path'],
        conditions=conditions,
        assembly_enzyme=assembly_enzyme,
        pmid=LIGATION_DATA[data]['PMID']
    )
