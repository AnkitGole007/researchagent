import sys

sys.path.insert(0, ".")
from data_pipeline.schema import connect_lancedb, get_or_create_papers_table, PaperRecord


def test_db_creates_table(tmp_path):
    db = connect_lancedb(str(tmp_path / "test_lancedb"))
    table = get_or_create_papers_table(db)
    assert "papers" in db.list_tables().tables
    assert "arxiv_id" in table.schema.names


def test_paper_record_fields():
    p = PaperRecord(
        arxiv_id="2301.00001",
        s2_id="abc",
        title="T",
        abstract="A",
        authors=["Alice"],
        submitted_date="2024-01-01",
        venue="NeurIPS",
        citation_count=42,
        max_author_citations=1200,
        pdf_url="https://arxiv.org/pdf/2301.00001",
        arxiv_url="https://arxiv.org/abs/2301.00001",
        fields_of_study=["Computer Science"],
    )
    assert p.citation_count == 42
