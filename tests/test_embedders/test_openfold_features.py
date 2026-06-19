"""Tests for OpenFold FASTA/MSA utilities."""

from interplm.embedders.openfold_features import (
    a3m_to_msa_tensor,
    parse_uniprot_id,
    read_fasta,
)


def test_parse_uniprot_id():
    assert parse_uniprot_id("sp|P12345|NAME") == "P12345"
    assert parse_uniprot_id("P99999 description") == "P99999"


def test_read_fasta(tmp_path):
    fasta = tmp_path / "test.fasta"
    fasta.write_text(
        ">sp|P11111|AAA\n"
        "MKTAY\n"
        ">sp|P22222|BBB\n"
        "ACDEF\n"
    )
    records = read_fasta(fasta)
    assert records == [("P11111", "MKTAY"), ("P22222", "ACDEF")]


def test_a3m_to_msa_tensor(tmp_path):
    a3m = tmp_path / "P11111.a3m"
    a3m.write_text(
        ">query\n"
        "MK-TAY\n"
        ">match1\n"
        "MK-TAY\n"
        ">match2\n"
        "MK-TaY\n"
    )
    msa = a3m_to_msa_tensor(a3m)
    assert msa.shape[0] == 1
    assert msa.shape[1] == 3
    assert msa.shape[2] == 5
