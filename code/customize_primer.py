import os
from os.path import join, exists
import pandas as pd
import numpy as np
from Bio import SeqIO

# read fasta file of fw and rv primer, and merge it into the csv format that works for the omega tool
def fasta_to_dataframe(fasta_file):
    """
    Reads a FASTA file and returns a list of (record.id, str(record.seq)) tuples.

    Args:
        fasta_file (str): Path to the input FASTA file.

    Returns:
        list: A list of (record.id, str(record.seq)) tuples.  Returns an empty
              list if the file is empty or an error occurs.
    """
    try:
        records = [(rec.id, str(rec.seq)) for rec in SeqIO.parse(fasta_file, 'fasta')]
        return records
    except FileNotFoundError:
        print(f"Error: File not found: {fasta_file}")
        return []
    except Exception as e:
        print(f"An error occurred while reading {fasta_file}: {e}")
        return []


def save_primers_to_csv(pfw_file, prv_file, output_csv_file="primers.csv"):
    """
    Reads forward and reverse primer FASTA files, extracts IDs and sequences,
    and saves them to a CSV file with specified column names.

    Args:
        pfw_file (str): Path to the forward primer FASTA file.
        prv_file (str): Path to the reverse primer FASTA file.
        output_csv_file (str, optional): Path to the output CSV file.
            Defaults to "primers.csv".
    """

    pfw_data = fasta_to_dataframe(pfw_file)
    prv_data = fasta_to_dataframe(prv_file)

    if not pfw_data and not prv_data:
        print("Error: Both forward and reverse primer FASTA files were empty or not found.  No CSV file created.")
        return
    elif not pfw_data:
        print("Error: Forward primer FASTA file was empty or not found.  No CSV file created for forward primers.")
        return
    elif not prv_data:
        print("Error: Reverse primer FASTA file was empty or not found. No CSV file created for reverse primers.")
        return

    # Create data for the DataFrame.  Use a list of dictionaries.
    data = []
    for (fw_primer, fw_seq), (rv_primer, rv_seq) in zip(pfw_data, prv_data):
        data.append({
            'fwd_name': fw_primer,
            'fwd_sequence': fw_seq,
            'rev_name': rv_primer,
            'rev_sequence': rv_seq
        })
    #If the lists are uneven, process the rest of the longer list
    if (len(pfw_data) > len(prv_data)):
        for i in range(len(prv_data), len(pfw_data)):
            fw_primer, fw_seq = pfw_data[i]
            data.append({
                'fwd_name': fw_primer,
                'fwd_sequence': fw_seq,
                'rev_name': None,
                'rev_sequence': None
            })
    elif (len(prv_data) > len(pfw_data)):
        for i in range(len(pfw_data), len(prv_data)):
            rv_primer, rv_seq = prv_data[i]
            data.append({
                'fwd_name': None,
                'fwd_sequence': None,
                'rev_name': rv_primer,
                'rev_sequence': rv_seq
            })

    df = pd.DataFrame(data) #create the dataframe

    try:
        df.to_csv(output_csv_file, index=False)
        print(f"Primer data saved to: {output_csv_file}")
    except Exception as e:
        print(f"An error occurred while saving to CSV: {e}")

if __name__ == "__main__":
    pfw_file = "data/primers/forward_finalprimers.fasta"  # Replace with your forward primer FASTA file
    prv_file = "data/primers/reverse_finalprimers.fasta"  # Replace with your reverse primer FASTA file
    save_primers_to_csv(pfw_file, prv_file)
    print("finished")
