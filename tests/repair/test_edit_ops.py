"""Transactional AKN edit ops, the safety spine under all guards."""

from __future__ import annotations

from datetime import date
from typing import Any

from lxml import etree

from codify.akn import AKN_NS
from codify.pipeline.enrich.bluebell import parse_to_akn
from codify.pipeline.enrich.validator import validate_akn
from codify.repair.edit_ops import _renumber_sequence, _split, apply_op, apply_plan, find_by_eid
from codify.repair.ops import (
    Delete,
    Merge,
    Move,
    RenumberSequence,
    SetBody,
    SetMoneyNumeral,
    SetNum,
    Split,
)


def _fixture() -> str:
    """A schema-valid act whose ARTICLE 2 has a num but no body (empty_article)."""
    bb = "BODY\n  SECTION 1\n    ARTICLE 1\n      Body of article one.\n    ARTICLE 2\n"
    return parse_to_akn(bb, country="gb", doctype="act", number="1", date="2020-01-01")


def _finding(xml: str, check: str) -> dict[str, Any] | None:
    return next((i for i in validate_akn(xml) if i["check"] == check), None)


def test_find_by_eid() -> None:
    root = etree.fromstring(_fixture().encode())
    empty = _finding(_fixture(), "empty_article")
    assert empty is not None
    assert find_by_eid(root, empty["eid"]) is not None
    assert find_by_eid(root, "nope") is None


def test_set_body_fills_empty_article_and_clears_finding() -> None:
    xml = _fixture()
    f = _finding(xml, "empty_article")
    assert f is not None
    res = apply_op(
        xml,
        SetBody(eid=f["eid"], bluebell="The body of the second article."),
        country="gb",
        target_finding=f,
    )
    assert res.ok, res.error
    assert _finding(res.xml, "empty_article") is None
    assert "second article" in res.xml


def test_delete_not_permitted_for_empty_article() -> None:
    # The load-bearing guard: deleting a content-bearing finding is never a repair.
    xml = _fixture()
    f = _finding(xml, "empty_article")
    assert f is not None
    res = apply_op(xml, Delete(eid=f["eid"]), country="gb", target_finding=f)
    assert not res.ok
    assert "not permitted" in res.error
    assert res.xml == xml


def test_structural_body_line_is_rejected() -> None:
    # A body that smuggles a structural keyword would mis-nest, reject.
    xml = _fixture()
    f = _finding(xml, "empty_article")
    assert f is not None
    res = apply_op(
        xml,
        SetBody(eid=f["eid"], bluebell="ARTICLE 6\nsneaky nested structure"),
        country="gb",
        target_finding=f,
    )
    assert not res.ok
    assert "structural keyword" in res.error


def test_op_that_introduces_a_finding_is_rejected() -> None:
    # Renumbering the empty article to "1" makes a duplicate under the section.
    xml = _fixture()
    f = _finding(xml, "empty_article")
    assert f is not None
    res = apply_op(xml, SetNum(eid=f["eid"], num="1"), country="gb")
    assert not res.ok
    assert "introduced" in res.error
    assert res.xml == xml


def test_missing_eid_rejected_cleanly() -> None:
    xml = _fixture()
    res = apply_op(xml, SetBody(eid="ghost", bluebell="x"), country="gb")
    assert not res.ok
    assert "not found" in res.error


def test_renumber_sequence_fixes_interleaved_collapse() -> None:
    # 7 points OCR-collapsed to 1,1,1,1,2,2,2 (two overlapping duplicate findings);
    # renumbering the whole run 1..7 in one op is the only fix.
    pts = "".join(
        f'<point eId="art_6__p{i}"><num>{"(1)" if i <= 4 else "(2)"}</num>'
        f"<content><p>text {i}</p></content></point>"
        for i in range(1, 8)
    )
    xml = (
        f'<akomaNtoso xmlns="{AKN_NS}"><act><body>'
        f'<article eId="art_6"><num>6</num><content>{pts}</content></article>'
        "</body></act></akomaNtoso>"
    )
    root = etree.fromstring(xml.encode())
    _renumber_sequence(
        root,
        RenumberSequence(parent_eid="art_6", kind="point", nums=[f"({n})" for n in range(1, 8)]),
    )
    nums = [p.findtext(f"{{{AKN_NS}}}num") for p in root.findall(f".//{{{AKN_NS}}}point")]
    assert nums == ["(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)"]


def test_renumber_sequence_ignores_nested_points() -> None:
    # A nested point belongs to its own inner run, renumbering the outer run
    # must not sweep it in (it did, when selection used .iter over all depths).
    xml = (
        f'<akomaNtoso xmlns="{AKN_NS}"><act><body><article eId="a"><num>1</num><content>'
        '<point eId="a__p1"><num>(1)</num><content><p>x</p>'
        '<point eId="a__p1__p1"><num>(9)</num><content><p>inner</p></content></point>'
        "</content></point>"
        '<point eId="a__p2"><num>(1)</num><content><p>y</p></content></point>'
        "</content></article></body></act></akomaNtoso>"
    )
    root = etree.fromstring(xml.encode())
    # only two direct children of the article: p1, p2, nums length must match those.
    _renumber_sequence(root, RenumberSequence(parent_eid="a", kind="point", nums=["(1)", "(2)"]))
    assert root.find(f".//{{{AKN_NS}}}point[@eId='a__p1']").findtext(f"{{{AKN_NS}}}num") == "(1)"
    assert root.find(f".//{{{AKN_NS}}}point[@eId='a__p2']").findtext(f"{{{AKN_NS}}}num") == "(2)"
    # the nested point keeps its own number, untouched by the outer renumber.
    inner = root.find(f".//{{{AKN_NS}}}point[@eId='a__p1__p1']")
    assert inner.findtext(f"{{{AKN_NS}}}num") == "(9)"


def test_renumber_sequence_count_mismatch_rejects() -> None:
    import pytest

    from codify.repair.edit_ops import _RejectedOp

    xml = (
        f'<akomaNtoso xmlns="{AKN_NS}"><act><body><article eId="a"><num>1</num><content>'
        '<point eId="a__p1"><num>(1)</num><content><p>x</p></content></point>'
        "</content></article></body></act></akomaNtoso>"
    )
    root = etree.fromstring(xml.encode())
    with pytest.raises(_RejectedOp):
        _renumber_sequence(
            root, RenumberSequence(parent_eid="a", kind="point", nums=["(1)", "(2)"])
        )


def test_apply_plan_atomic_renumber_clears_duplicate() -> None:
    # Three articles the OCR collapsed to all "1". Renumbering two of them clears
    # the duplicate only when applied together, per-op rejects each because the
    # intermediate state still collides.
    bb = (
        "BODY\n  SECTION 1\n    ARTICLE 1\n      First article body.\n"
        "    ARTICLE 1\n      Second article body.\n"
        "    ARTICLE 1\n      Third article body.\n"
    )
    xml = parse_to_akn(bb, country="gb", doctype="act", number="1", date="2020-01-01")
    dup = _finding(xml, "duplicate_number")
    if dup is None or len(dup.get("eids", [])) < 3:
        import pytest

        pytest.skip("fixture did not produce a 3-way duplicate")
    eids = dup["eids"]
    plan = [SetNum(eid=eids[1], num="2"), SetNum(eid=eids[2], num="3")]
    # per-op: the first set_num leaves (1,2,1), still a duplicate, so it's rejected
    assert not apply_op(xml, plan[0], country="gb", target_finding=dup).ok
    # atomic: both land together and the duplicate clears
    res = apply_plan(xml, plan, country="gb", target_finding=dup)
    assert res.ok, res.error
    assert _finding(res.xml, "duplicate_number") is None
    assert "First article body" in res.xml and "Third article body" in res.xml


def test_merge_clears_duplicate_number() -> None:
    bb = (
        "BODY\n  SECTION 1\n    ARTICLE 1\n      First body of the article.\n"
        "    ARTICLE 1\n      Second body of the article.\n"
    )
    xml = parse_to_akn(bb, country="gb", doctype="act", number="1", date="2020-01-01")
    dup = _finding(xml, "duplicate_number")
    if dup is None or len(dup.get("eids", [])) < 2:
        import pytest

        pytest.skip("fixture did not produce a duplicate_number pair")
    res = apply_op(
        xml, Merge(eid_a=dup["eids"][0], eid_b=dup["eids"][1]), country="gb", target_finding=dup
    )
    assert res.ok, res.error
    assert _finding(res.xml, "duplicate_number") is None
    assert "First body" in res.xml and "Second body" in res.xml  # both preserved


def test_reduction_allowed_for_cleaning_findings() -> None:
    # ocr_garble/body_artefact repairs legitimately shorten the body; the prose
    # non-decrease guard must not reject them (regression: it once did).
    xml = _fixture()  # sec_1__art_1 body = "Body of article one."
    res = apply_op(
        xml,
        SetBody(eid="sec_1__art_1", bluebell="Clean."),
        country="gb",
        target_finding={"check": "ocr_garble", "eid": "sec_1__art_1"},
    )
    assert res.ok, res.error
    assert "Clean." in res.xml


def test_reduction_still_guarded_for_non_cleaning_ops() -> None:
    # A Move that dropped prose would still be caught (guard not blanket-disabled).
    xml = _fixture()
    res = apply_op(
        xml,
        SetBody(eid="sec_1__art_1", bluebell="Short."),
        country="gb",
        target_finding={"check": "empty_article", "eid": "sec_1__art_1"},
    )
    assert not res.ok
    assert "shrank" in res.error


def test_split_cuts_swallowed_marker_into_sibling() -> None:
    # An article whose <p> swallowed a second enumerator's text.
    xml = (
        f'<akomaNtoso xmlns="{AKN_NS}"><act><body><section eId="sec_1"><num>1</num>'
        '<article eId="sec_1__art_1"><num>1</num><content>'
        '<p eId="p1">First part of the text. (2) Second part that was swallowed.</p>'
        "</content></article></section></body></act></akomaNtoso>"
    )
    root = etree.fromstring(xml.encode())
    _split(root, Split(eid="sec_1__art_1", marker="(2)"))
    arts = root.findall(f".//{{{AKN_NS}}}article")
    assert len(arts) == 2  # a new sibling was created
    head, tail = "".join(arts[0].itertext()), "".join(arts[1].itertext())
    assert "First part of the text." in head and "(2)" not in head  # marker cut from head
    assert "Second part that was swallowed." in tail
    assert arts[1].findtext(f"{{{AKN_NS}}}num") == "2"  # numbered from the marker


def _money_fixture() -> str:
    """An act whose article 8 carries a recorded defect: a numeral
    reading 500,000 beside words reading fifty thousand."""
    bb = "BODY\n  ARTICLE 8\n    Second category: not less than (٥٠٠,٠٠٠) خمسين ألف دينار زرزوري.\n"
    return parse_to_akn(bb, country="xz", doctype="act", number="12", date="1994-01-01")


def _money_finding() -> dict[str, Any]:
    """From the validator, not hand-written: the check emits the paragraph's
    eid, and a hardcoded article eid makes the clearance guard vacuous."""
    finding = _finding(_money_fixture(), "money_words_mismatch")
    assert finding is not None, "fixture no longer reproduces the money defect"
    return finding


def test_set_money_numeral_rewrites_only_the_numeral() -> None:
    res = apply_plan(
        _money_fixture(),
        [SetMoneyNumeral(eid=_money_finding()["eid"], old="٥٠٠,٠٠٠", new="٥٠,٠٠٠")],
        country="xz",
        target_finding=_money_finding(),
    )
    assert res.ok, res.error
    assert "(٥٠,٠٠٠) خمسين ألف دينار زرزوري" in res.xml
    assert "٥٠٠,٠٠٠" not in res.xml


def test_set_money_numeral_refuses_a_prose_surface() -> None:
    """The narrow op keeps a money correction from becoming a rewrite."""
    res = apply_plan(
        _money_fixture(),
        [SetMoneyNumeral(eid=_money_finding()["eid"], old="خمسين", new="50000")],
        country="xz",
        target_finding=_money_finding(),
    )
    assert not res.ok
    assert "not a numeral" in res.error


def test_set_money_numeral_refuses_an_absent_or_ambiguous_numeral() -> None:
    res = apply_plan(
        _money_fixture(),
        [SetMoneyNumeral(eid=_money_finding()["eid"], old="٩٩٩", new="١")],
        country="xz",
        target_finding=_money_finding(),
    )
    assert not res.ok
    assert "expected 1" in res.error


def test_money_finding_may_not_rewrite_the_body() -> None:
    res = apply_plan(
        _money_fixture(),
        [SetBody(eid=_money_finding()["eid"], bluebell="not less than 50,000 Dinars.")],
        country="xz",
        target_finding=_money_finding(),
    )
    assert not res.ok
    assert "not permitted" in res.error


def test_set_money_numeral_refuses_a_no_op_rewrite() -> None:
    """An identical rewrite passes every guard and fixes nothing."""
    res = apply_plan(
        _money_fixture(),
        [SetMoneyNumeral(eid=_money_finding()["eid"], old="٥٠٠,٠٠٠", new="٥٠٠,٠٠٠")],
        country="xz",
        target_finding=_money_finding(),
    )
    assert not res.ok
    assert "unchanged" in res.error


def test_set_money_numeral_refuses_a_digit_script_switch() -> None:
    """An Arabic act must not acquire Latin digits through a money repair."""
    res = apply_plan(
        _money_fixture(),
        [SetMoneyNumeral(eid=_money_finding()["eid"], old="٥٠٠,٠٠٠", new="50,000")],
        country="xz",
        target_finding=_money_finding(),
    )
    assert not res.ok
    assert "digit script" in res.error


def test_set_money_numeral_will_not_rewrite_a_digit_run_inside_a_longer_one() -> None:
    """Correcting 50 must not find the 50 inside 500."""
    xml = parse_to_akn(
        "BODY\n  ARTICLE 8\n    a fine of 500 dinars.\n",
        country="xz",
        doctype="act",
        number="12",
        date="1994-01-01",
    )
    res = apply_plan(
        xml,
        [SetMoneyNumeral(eid="art_8", old="50", new="5,000")],
        country="xz",
        target_finding={"check": "money_words_mismatch", "eid": "art_8"},
    )
    assert not res.ok
    assert "occurs 0 times" in res.error


def test_set_money_numeral_refuses_a_mixed_script_surface() -> None:
    """Two mixed surfaces compare equal under a naive script check."""
    res = apply_plan(
        _money_fixture(),
        [SetMoneyNumeral(eid=_money_finding()["eid"], old="٥0٠,٠٠٠", new="0٥,٠٠٠")],
        country="xz",
        target_finding=_money_finding(),
    )
    assert not res.ok
    assert "digit script" in res.error


def test_set_money_numeral_will_not_match_across_a_grouping_separator() -> None:
    """A separator is part of the numeral, so digit-only boundaries let `50`
    match inside `50,000` and `000` match its tail."""
    xml = parse_to_akn(
        "BODY\n  ARTICLE 8\n    a fine of 50,000 dinars.\n",
        country="xz",
        doctype="act",
        number="12",
        date="1994-01-01",
    )
    finding = {"check": "money_words_mismatch", "eid": "art_8"}
    for old in ("50", "000"):
        res = apply_plan(
            xml,
            [SetMoneyNumeral(eid="art_8", old=old, new="7")],
            country="xz",
            target_finding=finding,
        )
        assert not res.ok, old
        assert "occurs 0 times" in res.error


def test_duplicate_number_policy_allows_restore_and_remove() -> None:
    # #702: a mis-parse duplicate is fixed by SetBody (restore the truncated
    # body) + Move (the real nested unit out) + Delete (the false wrapper), not
    # a cosmetic renumber. Those ops must clear the op-policy gate; the sandbox
    # grounding floor + risk escalation (delete/ungrounded -> high -> queued)
    # stay the binding safety.
    from codify.repair.edit_ops import _OP_POLICY

    assert {SetBody, Move, Delete}.issubset(_OP_POLICY["duplicate_number"])


def test_set_body_reaches_gates_for_duplicate_number() -> None:
    # Previously "set_body not permitted for duplicate_number"; now it passes
    # the policy gate (this plan is then rejected only because it does not clear
    # the duplicate, proving the block is the content gate, not the policy).
    bb = (
        "BODY\n  SECTION 1\n    ARTICLE 1\n      First article body.\n"
        "    ARTICLE 1\n      Second article body.\n"
    )
    xml = parse_to_akn(bb, country="gb", doctype="act", number="1", date="2020-01-01")
    dup = _finding(xml, "duplicate_number")
    if dup is None:
        import pytest

        pytest.skip("fixture did not produce a duplicate_number")
    res = apply_op(
        xml,
        SetBody(eid=dup["eids"][0], bluebell="restored body text for the article."),
        country="gb",
        target_finding=dup,
    )
    assert "not permitted" not in res.error


def test_restore_and_remove_shape_clears_duplicate_number() -> None:
    # #702: the mis-parse fix (set_body the truncated body + delete the false
    # copy) applies as one plan and clears the duplicate, the path the widened
    # policy enables and the agent instructions now describe.
    bb = (
        "BODY\n  SECTION 1\n    ARTICLE 1\n      Truncated body.\n"
        "    ARTICLE 1\n      Stray false wrapper body.\n"
    )
    xml = parse_to_akn(bb, country="gb", doctype="act", number="1", date="2020-01-01")
    dup = _finding(xml, "duplicate_number")
    if dup is None or len(dup.get("eids", [])) < 2:
        import pytest

        pytest.skip("fixture did not produce a duplicate_number pair")
    plan = [
        SetBody(eid=dup["eids"][0], bluebell="Restored full body of the real article."),
        Delete(eid=dup["eids"][1]),
    ]
    res = apply_plan(xml, plan, country="gb", target_finding=dup)
    assert res.ok, res.error
    assert _finding(res.xml, "duplicate_number") is None
    assert "Restored full body" in res.xml


def test_a_repair_leaves_emitted_paragraph_identity_alone() -> None:
    """_reindex derives an eId from the tag and its <num>. An intro has no
    <num>, so giving the container an eId made the repair pass renumber it to
    intro_0 and drop its paragraph from p_1 to p_0. The emitter leaves those
    containers bare, which keeps them invisible here."""
    from codify.akn import Document, Paragraph, Section, to_akn
    from codify.repair.edit_ops import _reindex

    doc = Document(
        frbr_work_uri="/akn/gb/act/1981/64",
        frbr_expression_uri="/akn/gb/act/1981/64/eng@1981-10-30",
        language="eng",
        expression_date=date(1981, 10, 30),
        body=[
            Section(
                akn_eid="sec_1",
                akn_type="section",
                position=0,
                number="1",
                intro="The Minister may:",
                children=[
                    Paragraph(
                        akn_eid="sec_1__para_1",
                        akn_type="paragraph",
                        position=0,
                        number="1",
                        text="First limb.\nSecond limb.",
                    ),
                ],
                wrap_up="and no other.",
            ),
        ],
    )
    root = etree.fromstring(to_akn(doc).encode())
    section = root.find(f".//{{{AKN_NS}}}section")
    before = [el.get("eId") for el in section.iter() if el.get("eId")]

    remap: dict[str, str] = {}
    _reindex(section, None, remap)

    assert remap == {}
    assert [el.get("eId") for el in section.iter() if el.get("eId")] == before
    assert "sec_1__intro__p_1" in before
    assert "sec_1__para_1__p_1" in before


def test_a_moved_section_takes_its_paragraph_identities_with_it() -> None:
    """intro, wrapUp and content carry a prefix without holding an eId, so a
    walk that only follows eId-bearing elements never reaches their paragraphs
    and leaves them pointing at the old ancestor."""
    from codify.repair.edit_ops import _reindex

    section = etree.fromstring(
        f'<section xmlns="{AKN_NS}" eId="sec_1"><num>1</num>'
        f'<intro><p eId="sec_1__intro__p_1">Intro.</p></intro>'
        f'<content><p eId="sec_1__p_1">First.</p><p eId="sec_1__p_2">Second.</p></content>'
        f'<wrapUp><p eId="sec_1__wrapup__p_1">Wrap.</p></wrapUp></section>'.encode()
    )
    remap: dict[str, str] = {}
    _reindex(section, "part_2", remap)

    assert [el.get("eId") for el in section.iter(f"{{{AKN_NS}}}p")] == [
        "part_2__sec_1__intro__p_1",
        "part_2__sec_1__p_1",
        "part_2__sec_1__p_2",
        "part_2__sec_1__wrapup__p_1",
    ]
    # Every rename is recorded, so _rewrite_refs can repoint hrefs at them.
    assert remap["sec_1__p_2"] == "part_2__sec_1__p_2"
    assert remap["sec_1__wrapup__p_1"] == "part_2__sec_1__wrapup__p_1"
