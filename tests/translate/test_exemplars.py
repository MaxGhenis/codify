"""Fixture-driven tests for the running-exemplar pool."""

from __future__ import annotations

from codify.translate.exemplars import Exemplar, ExemplarPool


class TestExemplarPool:
    def test_picks_deontic_rich_english_clause(self) -> None:
        pool = ExemplarPool(target_language="French", source_language="eng")
        source = [
            {"eid": "art_1", "body": "A general definition applies to this Act."},
            {
                "eid": "art_2",
                "body": "Every project owner shall submit an annual report to the Ministry.",
            },
        ]
        translated = [
            {"eid": "art_1", "lines": ["Une definition generale s'applique."]},
            {
                "eid": "art_2",
                "lines": [
                    "Chaque proprietaire de projet doit soumettre un rapport annuel au Ministere."
                ],
            },
        ]
        picked = pool.pick_from(source, translated)
        assert picked is not None
        assert "shall submit" in picked.source
        assert "doit soumettre" in picked.target
        assert pool.entries == [picked]

    def test_picks_deontic_rich_arabic_clause(self) -> None:
        pool = ExemplarPool(target_language="English", source_language="ara")
        source = [
            {
                "eid": "art_1",
                "body": "يجب على كل صاحب مشروع أن يقدم تقريراً سنوياً إلى الوزارة المختصة.",
            },
        ]
        translated = [
            {"eid": "art_1", "lines": ["Every project owner shall submit an annual report."]},
        ]
        picked = pool.pick_from(source, translated)
        assert picked is not None
        assert "يجب" in picked.source

    def test_skips_short_clauses(self) -> None:
        pool = ExemplarPool(target_language="English", source_language="eng")
        source = [{"eid": "art_1", "body": "Person shall pay."}]
        translated = [{"eid": "art_1", "lines": ["Person shall pay."]}]
        assert pool.pick_from(source, translated) is None
        assert pool.entries == []

    def test_skips_long_clauses(self) -> None:
        pool = ExemplarPool(target_language="English", source_language="eng")
        body = "A person shall pay a fine of one hundred Zerzuran dinars " * 20
        source = [{"eid": "art_1", "body": body}]
        translated = [{"eid": "art_1", "lines": [body]}]
        assert pool.pick_from(source, translated) is None

    def test_skips_non_deontic_prose(self) -> None:
        pool = ExemplarPool(target_language="English", source_language="eng")
        source = [
            {
                "eid": "art_1",
                "body": "The following words in this Act have the meanings described.",
            }
        ]
        translated = [
            {"eid": "art_1", "lines": ["The following words in this Act have the meanings."]}
        ]
        assert pool.pick_from(source, translated) is None

    def test_capacity_fifo_evicts_earliest(self) -> None:
        pool = ExemplarPool(target_language="English", source_language="eng", capacity=2)
        for i in range(4):
            pool.add(
                Exemplar(
                    source=f"A person shall pay {i} dinars per unit produced under this Act.",
                    target=f"Person shall pay {i} dinars.",
                )
            )
        entries = pool.entries
        assert len(entries) == 2
        # Last two survive.
        assert "2 dinars" in entries[0].source
        assert "3 dinars" in entries[1].source

    def test_render_block_empty_when_no_entries(self) -> None:
        pool = ExemplarPool(target_language="English", source_language="eng")
        assert pool.render_block() == ""

    def test_render_block_carries_source_and_target(self) -> None:
        pool = ExemplarPool(target_language="English", source_language="eng")
        pool.add(
            Exemplar(
                source="Every project owner shall submit a report to the Ministry annually.",
                target="Every project owner shall submit a report to the Ministry annually.",
            )
        )
        block = pool.render_block()
        assert "Already-agreed translations" in block
        assert "shall submit a report" in block

    def test_missing_source_language_falls_back_to_no_pick(self) -> None:
        pool = ExemplarPool(target_language="English", source_language=None)
        source = [
            {
                "eid": "art_1",
                "body": "Every project owner shall submit an annual report to the Ministry.",
            }
        ]
        translated = [
            {"eid": "art_1", "lines": ["Every project owner shall submit an annual report."]}
        ]
        assert pool.pick_from(source, translated) is None

    def test_translated_missing_for_source_eid_returns_none(self) -> None:
        pool = ExemplarPool(target_language="English", source_language="eng")
        source = [
            {
                "eid": "art_1",
                "body": "Every project owner shall submit an annual report to the Ministry.",
            }
        ]
        translated: list[dict[str, object]] = []
        assert pool.pick_from(source, translated) is None
