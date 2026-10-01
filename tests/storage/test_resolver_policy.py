"""The rules that changed at RESOLVER_VERSION 2, and the re-stamp they need.

Both came from reading 108 stamped rows: references resolved onto an
explanatory entry rather than the unit it explains, and most of one corpus's
law stamps landed on one ordinance no citing text named.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from codify.akn import AKN_NS
from codify.akn.elements import ElementBase
from codify.akn.references import InlineReference
from codify.storage.resolve_refs import RESOLVER_VERSION, _text_names_law


def test_the_citing_text_must_name_the_instrument() -> None:
    """A law stamp nothing in the text supports is worse than a gap: the row
    leaves every dangling count and keeps no record that it was a guess."""
    interpretation_ordinance = "قانون تنقيح التشاريع الأساسي رقم 30 لسنة 1934"
    assert not _text_names_law(
        "اذا بيع السم الى طبيب مجاز او طبيب اسنان مجاز",
        "/akn/xa/act/1934/30",
        interpretation_ordinance,
    )
    assert not _text_names_law(
        "كل من سمح باهماله لاية حيوانات بالحاق ضرر باية شجرة",
        "/akn/xa/act/1934/30",
        interpretation_ordinance,
    )


@pytest.mark.parametrize(
    ("text", "uri", "title"),
    [
        # Arabic-Indic digits: the number is named, in the script the source uses.
        ("قانون الخدمة المدنية رقم ٤ لسنة ١٩٩٨", "/akn/xa/act/1998/4", None),
        # Named by number in ASCII.
        (
            "means Regulation (EU) 2016/679 of the European Parliament",
            "/akn/xe/act/reg/2016/679",
            "REGULATION (EU) 2016/679 OF THE EUROPEAN PARLIAMENT",
        ),
        # Named by title, and the title is shorter than the match window: an act
        # called "Про запобігання корупції" is cited whole or not at all.
        (
            'Законом України "Про запобігання корупції" декларацію',
            "/akn/xy/act/2014/1700-VII",
            "Про запобігання корупції",
        ),
    ],
)
def test_a_named_instrument_still_resolves(text: str, uri: str, title: str | None) -> None:
    assert _text_names_law(text, uri, title)


def test_the_number_match_does_not_fire_on_a_substring() -> None:
    """`30` inside `1930` is not a citation of law 30."""
    assert not _text_names_law("صدر سنة ١٩٣٠ بشأن الأملاك", "/akn/xa/act/1934/30", None)


def test_the_version_is_ahead_of_the_first_policy() -> None:
    """The selector re-examines rows below this; leaving it at 1 would strand
    every wrong stamp the first policy wrote."""
    assert RESOLVER_VERSION >= 2


def test_a_title_of_short_tokens_confirms_nothing() -> None:
    """An empty word list left an empty match window, and `"" in text` is true
    of every text: the check would have confirmed whatever it was asked."""
    assert not _text_names_law("unrelated prose", "/akn/x/act/2024/zz", "AB CD")
    assert not _text_names_law("unrelated prose", "/akn/x/act/2024/zz", "")


def test_a_publisher_href_is_not_mistaken_for_a_reading() -> None:
    """The discriminator that string shape cannot make.

    Publisher-authored AKN writes relative `/akn/...` hrefs of exactly the form
    this pipeline mints when it reads prose. Judging by the string alone filed
    every authored link as a reading and put it through the confirmation gate,
    and filed any minted absolute URL as authored. The writer records it
    instead: the parser marks what it read, the markup pass marks what it made.
    """
    from lxml import etree

    from codify.akn._parser import _build_ref

    ns = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
    authored = etree.Element(f"{{{ns}}}ref", attrib={"href": "/akn/xx/act/2000/36"})
    minted = etree.Element(
        f"{{{ns}}}ref", attrib={"href": "/akn/xx/act/2000/36", "class": "derived"}
    )

    # Identical hrefs, opposite provenance.
    parsed = _build_ref(authored, 0, 3, "the")
    assert parsed is not None
    assert parsed.origin == "href"
    parsed = _build_ref(minted, 0, 3, "the")
    assert parsed is not None
    assert parsed.origin == "text"


def test_origin_survives_a_round_trip_through_akn() -> None:
    """The writer has to emit what the parser reads.

    `parse_akn(to_akn(doc))` turned every reading back into a publisher-authored
    link, because only the parser knew the marker. Provenance that does not
    survive serialisation is provenance the pipeline loses the first time a
    document is written out and read back.
    """
    from datetime import date

    from codify.akn import Article, Document, Paragraph, Title
    from codify.akn.io import parse_akn, to_akn
    from codify.akn.references import CrossReference

    doc = Document(
        frbr_work_uri="/akn/xx/act/2024/1",
        frbr_expression_uri="/akn/xx/act/2024/1/eng@2024-01-01",
        language="eng",
        expression_date=date(2024, 1, 1),
        body=[
            Title(
                akn_eid="ttl_1",
                akn_type="title",
                position=1,
                children=[
                    Article(
                        akn_eid="art_1",
                        akn_type="article",
                        position=1,
                        children=[
                            Paragraph(
                                akn_eid="art_1__p_1",
                                akn_type="paragraph",
                                position=1,
                                text="As provided in Article 2.",
                                references=[
                                    CrossReference(
                                        start_offset=15,
                                        end_offset=24,
                                        text_snippet="Article 2",
                                        target_eid="art_2",
                                        origin="text",
                                    )
                                ],
                            )
                        ],
                    )
                ],
            )
        ],
    )
    back = parse_akn(to_akn(doc))

    def walk(nodes: Sequence[ElementBase]) -> list[InlineReference]:
        found: list[InlineReference] = []
        for node in nodes:
            found.extend(getattr(node, "references", []) or [])
            found.extend(walk(getattr(node, "children", []) or []))
        return found

    refs = walk(back.body)
    assert refs, "the round trip lost the reference itself"
    assert refs[0].origin == "text", "a reading came back as an authored link"


def test_the_amendment_class_does_not_erase_the_marker() -> None:
    """Both tokens or neither, through the pass that sets one of them.

    The amendment pass replaced `class` outright, so a reference this pipeline
    minted and then classified as an amendment came back as publisher-authored
    and skipped the confirmation gate entirely. Driven through
    `_classify_amendment_refs` rather than a hand-built element, or the test
    proves only that the parser can read two tokens.
    """
    from lxml import etree

    from codify.akn._parser import _build_ref
    from codify.akn.references import DERIVED_REF_CLASS
    from codify.pipeline.enrich.inline_markup import _classify_amendment_refs

    body = etree.fromstring(
        f'<body xmlns="{AKN_NS}"><p>In section 9 of the 2000 Act, for '
        f'<ref href="/akn/xx/act/2000/36" class="{DERIVED_REF_CLASS}">the 2000 Act</ref> '
        "substitute the following.</p></body>"
    )
    assert _classify_amendment_refs(body, "gb") >= 1, "the pass did not classify the ref"

    ref = body.find(f".//{{{AKN_NS}}}ref")
    tokens = (ref.get("class") or "").split()
    assert DERIVED_REF_CLASS in tokens, "classifying the amendment erased the provenance"
    assert any(t.startswith("amendment-") for t in tokens)
    parsed = _build_ref(ref, 0, 3, "the")
    assert parsed is not None
    assert parsed.origin == "text"


@pytest.mark.parametrize(
    ("ref_type", "target_uri"),
    [
        ("citation", "/akn/xx/act/2000/36"),
        ("cross_reference", "#art_2"),
        ("cross_reference", "/akn/xx/act/2000/36"),
        ("amendment_replace", "/akn/xx/act/2000/36"),
    ],
)
def test_reading_a_row_back_keeps_its_provenance(ref_type: str, target_uri: str) -> None:
    """Recorded forward, dropped on the way back.

    `_row_to_ref` defaulted every reconstructed reference to `href`, so a
    document read out of the database carried the opposite provenance to the
    one stored, on every variant. Parametrised over all four, because the
    default was in each of them and fixing one says nothing about the others.
    """
    import uuid as _uuid

    from codify.storage.mappers import _row_to_ref
    from codify.storage.models import CrossReference as CrossReferenceRow

    row = CrossReferenceRow(
        source_provision_id=_uuid.uuid4(),
        target_uri=target_uri,
        ref_type=ref_type,
        edge_class="mod_textual" if ref_type.startswith("amendment") else "freetext_reference",
        resolution_origin="text",
    )
    assert _row_to_ref(row).origin == "text"


@pytest.mark.parametrize(
    ("work_uri", "expected"),
    [
        # The year slot is the one before the number. Scanning every segment
        # read a four-digit number as its own year, so an instrument numbered
        # 1451 confirmed on "1451" as though the text had written a year.
        ("/akn/xx/act/si/2006/1451", ("1451", "2006")),
        # A suffix is part of what the instrument is called and must survive.
        ("/akn/xx/act/2014/1700-VII", ("1700-VII", "2014")),
        # A full date in the year slot still names its year.
        ("/akn/xx/act/2015-02-12/33-A", ("33-A", "2015")),
        # A URI pads where a drafter does not.
        ("/akn/xx/act/1998/04", ("4", "1998")),
    ],
)
def test_the_number_and_year_come_from_the_slots_that_hold_them(
    work_uri: str, expected: tuple[str, str]
) -> None:
    from codify.storage.resolve_refs import _number_and_year

    assert _number_and_year(work_uri) == expected


def test_the_gate_accepts_either_name_the_instrument_answers_to() -> None:
    """Ingest stores the long title; a drafter writes the short one.

    Checking only the stored title refused every citation made by the name
    people actually use.
    """
    long_title = "An Act to make provision about the processing of information"
    short_title = "Data Protection Act 2018"
    assert _text_names_law(
        "under the Data Protection Act 2018", "/akn/xx/act/2018/12", long_title, short_title
    )
    assert _text_names_law(f"under {long_title}", "/akn/xx/act/2018/12", long_title, short_title)
    assert not _text_names_law(
        "unrelated prose entirely", "/akn/xx/act/2018/12", long_title, short_title
    )


@pytest.mark.parametrize(
    "stub", ["nisi/2004/3078 (latest)", "uksi/2016/1035 (latest)", "/akn/xx/act/2018/12"]
)
def test_a_stub_title_confirms_nothing(stub: str) -> None:
    """The acquisition lane mints a row whose title is a path, not a name.

    A citing text that happens to repeat that path then confirmed the law by
    its "title", which is confirmation by coincidence. Only the number-and-year
    rule may speak for such a row.
    """
    assert not _text_names_law(f"the {stub} order applies here", "/akn/xx/act/9999/1", stub)


def test_a_real_name_still_confirms() -> None:
    """The control: the guard must not swallow names that look terse."""
    assert _text_names_law(
        "the Data Protection Act 2018 applies", "/akn/xx/act/9999/1", "Data Protection Act 2018"
    )


def test_an_amendment_survives_the_round_trip_with_its_operation() -> None:
    """The provenance marker and the operation share one attribute.

    The emitter assigned the marker over whatever `_set_ref_href` had written,
    so a text-derived amendment came back as a plain citation and the operation
    was lost on every write-and-read.
    """
    from datetime import date

    from codify.akn import Article, Document, Paragraph, Title
    from codify.akn.io import parse_akn, to_akn
    from codify.akn.references import AmendmentReference

    doc = Document(
        frbr_work_uri="/akn/xx/act/2024/1",
        frbr_expression_uri="/akn/xx/act/2024/1/eng@2024-01-01",
        language="eng",
        expression_date=date(2024, 1, 1),
        body=[
            Title(
                akn_eid="ttl_1",
                akn_type="title",
                position=1,
                children=[
                    Article(
                        akn_eid="art_1",
                        akn_type="article",
                        position=1,
                        children=[
                            Paragraph(
                                akn_eid="art_1__p_1",
                                akn_type="paragraph",
                                position=1,
                                text="In the 2000 Act, section 9 is repealed.",
                                references=[
                                    AmendmentReference(
                                        start_offset=3,
                                        end_offset=15,
                                        text_snippet="the 2000 Act",
                                        amends_uri="/akn/xx/act/2000/36",
                                        operation="delete",
                                        origin="text",
                                    )
                                ],
                            )
                        ],
                    )
                ],
            )
        ],
    )

    def walk(nodes: Sequence[ElementBase]) -> list[InlineReference]:
        found: list[InlineReference] = []
        for node in nodes:
            found.extend(getattr(node, "references", []) or [])
            found.extend(walk(getattr(node, "children", []) or []))
        return found

    refs = walk(parse_akn(to_akn(doc)).body)
    assert refs, "the round trip lost the reference"
    assert refs[0].kind == "amendment_reference", "the amendment came back as a citation"
    assert isinstance(refs[0], AmendmentReference)
    assert refs[0].operation == "delete", "the operation was lost"
    assert refs[0].origin == "text"


@pytest.mark.parametrize("name", ["Constitution", "الدستور", "Anti-Corruption", "Правопорядок"])
def test_a_one_word_name_is_not_a_stub(name: str) -> None:
    """The stub test is a path, not the absence of spaces.

    A pattern of "no whitespace" called every single-token name a placeholder,
    in any script, and silenced the title rule for all of them.
    """
    from codify.storage.resolve_refs import _is_placeholder_title

    assert not _is_placeholder_title(name)
    assert _text_names_law(f"under the {name}, a duty arises", "/akn/xx/act/9999/1", name)


@pytest.mark.parametrize(
    "stub", ["nisi/2004/3078 (latest)", "/akn/xx/act/2018/12", "uksi/2016/1035"]
)
def test_a_path_is_still_a_stub(stub: str) -> None:
    from codify.storage.resolve_refs import _is_placeholder_title

    assert _is_placeholder_title(stub)


def test_the_scope_pass_marks_what_it_mints() -> None:
    """The amendment-scope pass builds its own `<ref>`.

    Those carried the operation and no provenance marker, so they landed as
    publisher-authored and skipped the confirmation gate entirely — the one
    place a minted reference could still pass itself off as an authored one.
    """
    from lxml import etree

    from codify.akn._parser import _build_ref
    from codify.akn.references import DERIVED_REF_CLASS
    from codify.pipeline.enrich.inline_markup import _mark_scoped_amendments

    body = etree.fromstring(
        f'<body xmlns="{AKN_NS}"><hcontainer><p>The '
        f'<ref href="/akn/xx/act/2000/36">2000 Act</ref> is amended as follows.</p>'
        "<p>In section 9, omit subsection (2).</p></hcontainer></body>"
    )
    assert _mark_scoped_amendments(body) == 1
    paragraphs = list(body.iter(f"{{{AKN_NS}}}p"))
    opener, op_para = paragraphs[0], paragraphs[1]

    # The one this pass built: marked, and read back as a reading.
    minted = op_para.find(f"{{{AKN_NS}}}ref")
    assert minted is not None, "the scope pass minted nothing"
    assert DERIVED_REF_CLASS in (minted.get("class") or "").split()
    parsed = _build_ref(minted, 0, 3, "the")
    assert parsed is not None
    assert parsed.origin == "text"

    # The opener's ref came from the document and keeps its provenance: it is
    # classified, not minted, and marking it would be a lie the other way.
    authored = opener.find(f"{{{AKN_NS}}}ref")
    assert DERIVED_REF_CLASS not in (authored.get("class") or "").split()
    parsed = _build_ref(authored, 0, 3, "the")
    assert parsed is not None
    assert parsed.origin == "href"


@pytest.mark.parametrize("origin", ["href", "text", "registry"])
def test_the_model_knows_every_origin_the_resolver_writes(origin: str) -> None:
    """A value the model does not know is a crash on the read path.

    The resolver writes `registry` for an upstream register's own identifier.
    The reference model listed only `href` and `text`, and the cast in the
    mapper is a static hint with no runtime effect, so reading such a row back
    raised a ValidationError rather than mislabelling anything — on a lens path,
    for a whole jurisdiction's references.
    """
    import uuid as _uuid

    from codify.storage.mappers import _row_to_ref
    from codify.storage.models import CrossReference as CrossReferenceRow

    row = CrossReferenceRow(
        source_provision_id=_uuid.uuid4(),
        target_uri="/go/772-19" if origin == "registry" else "/akn/xa/act/2000/36",
        ref_type="citation",
        edge_class="freetext_reference",
        resolution_origin=origin,
    )
    assert _row_to_ref(row).origin == origin


def test_a_series_citation_read_from_prose_is_marked() -> None:
    """The last unmarked minting site.

    `make_si_ref` built its own element, so a series citation matched in prose
    landed with no class at all and read back as publisher-authored — the one
    remaining way a reading could pass itself off as an authored link.
    """
    import asyncio

    from codify.akn._parser import _build_ref
    from codify.akn.references import DERIVED_REF_CLASS
    from codify.pipeline.enrich.inline_markup import emit_inline_markup

    akn = (
        f'<akomaNtoso xmlns="{AKN_NS}"><act><meta><identification source="#s">'
        '<FRBRWork><FRBRthis value="/akn/xz/act/2024/1"/>'
        '<FRBRuri value="/akn/xz/act/2024/1"/>'
        '<FRBRdate date="2024-01-01" name="enacted"/></FRBRWork>'
        '<FRBRExpression><FRBRthis value="/akn/xz/act/2024/1/eng@2024-01-01"/>'
        '<FRBRuri value="/akn/xz/act/2024/1/eng@2024-01-01"/>'
        '<FRBRdate date="2024-01-01" name="validFrom"/>'
        '<FRBRlanguage language="eng"/></FRBRExpression></identification></meta>'
        '<body><section eId="section-1"><num>1</num><content>'
        "<p>Revoke S.I. 2016/1035 accordingly.</p>"
        "</content></section></body></act></akomaNtoso>"
    )
    out = asyncio.run(emit_inline_markup(akn, "gb", "act", None))
    from lxml import etree

    refs = list(etree.fromstring(out.encode()).iter(f"{{{AKN_NS}}}ref"))
    minted = [r for r in refs if "/si/" in (r.get("href") or "") or "2016" in (r.get("href") or "")]
    assert minted, "the series lane matched nothing to check"
    for ref in minted:
        assert DERIVED_REF_CLASS in (ref.get("class") or "").split()
        parsed = _build_ref(ref, 0, 3, "the")
        assert parsed is not None
        assert parsed.origin == "text"


def test_the_scope_opener_keeps_a_marker_it_already_had() -> None:
    """M4: the opener branch composes too.

    The scope pass classifies the opener's own ref in a separate branch from
    the one that mints. Where that opener ref was itself minted by an earlier
    pass, assigning the operation over its class erased the provenance, and
    nothing failed: the minting branch's test does not reach this one.
    """
    from lxml import etree

    from codify.akn._parser import _build_ref
    from codify.akn.references import DERIVED_REF_CLASS
    from codify.pipeline.enrich.inline_markup import _mark_scoped_amendments

    body = etree.fromstring(
        f'<body xmlns="{AKN_NS}"><hcontainer><p>The '
        f'<ref href="/akn/xz/act/2000/36" class="{DERIVED_REF_CLASS}">2000 Act</ref>'
        " is amended as follows.</p>"
        "<p>In section 9, omit subsection (2).</p></hcontainer></body>"
    )
    _mark_scoped_amendments(body)
    opener = list(body.iter(f"{{{AKN_NS}}}p"))[0].find(f"{{{AKN_NS}}}ref")
    tokens = (opener.get("class") or "").split()
    assert DERIVED_REF_CLASS in tokens, "classifying the opener erased its provenance"
    assert any(t.startswith("amendment-") for t in tokens)
    parsed = _build_ref(opener, 0, 3, "the")
    assert parsed is not None
    assert parsed.origin == "text"


@pytest.mark.parametrize("digits", ["30", "2018", "١٩٩٨"])
def test_a_title_of_only_digits_is_not_a_name(digits: str) -> None:
    """A number is not a name, in any script.

    A law stored with a title of "30" confirmed on any citing text carrying a
    30 — a duration, a quantity, a date — which is exactly the coincidence the
    number-and-year rule exists to avoid, arriving through the title rule
    instead.
    """
    text = f"within {digits} days of publication the duty arises"
    assert not _text_names_law(text, "/akn/xz/act/9999/1", digits)


def test_a_short_name_in_a_dense_script_is_still_a_name() -> None:
    """The control on the same rule: the test is digits, not length."""
    assert _text_names_law("بموجب الدستور الزرزوري", "/akn/xz/act/9999/1", "الدستور")


@pytest.mark.parametrize(
    ("text", "title"),
    [
        # The name is a substring of a longer word, not a mention of it.
        ("the Actuarial Standards Board reported", "Act"),
        ("a preconstitutional settlement", "Constitution"),
        ("reconstituted the panel", "Constitution"),
    ],
)
def test_a_name_inside_a_longer_word_is_not_a_mention(text: str, title: str) -> None:
    """The title rule matches on word boundaries.

    Without them a title matched anywhere it appeared as a substring, so a law
    called "Act" was named by every text containing "Actuarial".
    """
    assert not _text_names_law(text, "/akn/xz/act/9999/1", title)


def test_the_name_itself_still_matches() -> None:
    """The control: boundaries must not cost a genuine mention."""
    assert _text_names_law("under the Act, a duty arises", "/akn/xz/act/9999/1", "Act")


@pytest.mark.parametrize("classification", ["", "amendment-delete", "derived"])
def test_registry_links_keep_their_origin_through_the_mapper(classification: str) -> None:
    import uuid

    from lxml import etree

    from codify.akn._parser import _build_ref
    from codify.storage.mappers import _ref_to_row, _row_to_ref

    element = etree.Element("ref", href="/go/772-19", attrib={"class": classification})
    parsed = _build_ref(element, 0, 3, "Act")
    assert parsed is not None
    expected = "text" if classification == "derived" else "registry"
    assert parsed.origin == expected
    row = _ref_to_row(parsed, uuid.uuid4())
    assert row is not None
    assert row.resolution_origin == expected
    assert _row_to_ref(row).origin == expected


@pytest.mark.parametrize(
    ("text", "rival", "expected"),
    [
        (
            "Widgets Act and Companies Consolidation Act apply.",
            "Companies Consolidation Act",
            False,
        ),
        ("Widgets Act (Amendment) applies.", "Widgets Act (Amendment)", True),
        ("Widgets Act applies.", "Widgets Act", True),
        (
            "Widgets Act applies; Widgets Act (Amendment) also applies.",
            "Widgets Act (Amendment)",
            False,
        ),
        ("Widgets Actuarial applies alongside Widgets Act.", "Widgets Actuarial", False),
    ],
)
def test_only_overlapping_title_occurrences_compete(text: str, rival: str, expected: bool) -> None:
    import uuid

    from codify.storage.resolve_refs import _title_occurrences_outmatched

    target, other = uuid.UUID(int=1), uuid.UUID(int=2)
    names: list[tuple[uuid.UUID, str | None, str | None]] = [
        (target, "Widgets Act", None),
        (other, rival, None),
    ]
    assert _title_occurrences_outmatched(target, text, names) is expected
    assert _title_occurrences_outmatched(target, text, list(reversed(names))) is expected


@pytest.mark.parametrize("number", ["1234567", "01234567"])
def test_a_padded_uri_accepts_its_canonical_citation_number(number: str) -> None:
    work_uri = f"/akn/xx/act/2015-02-12/{number}-VIII"
    assert (
        _text_names_law("As set out in Law No. 1234567-VIII of 2015.", work_uri, None)
        == "number_and_year"
    )


@pytest.mark.parametrize(
    ("named", "rival"),
    [
        ("Tax Act", "Tax Act"),
        ("Widgets Act", "Widgets  Act"),
        ("STRASSE ACT", "Straße Act"),
    ],
)
def test_equivalent_title_spellings_are_ambiguous(named: str, rival: str) -> None:
    import uuid

    from codify.storage.resolve_refs import _title_occurrences_outmatched

    target, other = uuid.UUID(int=1), uuid.UUID(int=2)
    text = f"Under the {named}, a duty arises."
    assert _text_names_law(text, "/akn/xz/act/9999/unknown", rival) == "title"
    names: list[tuple[uuid.UUID, str | None, str | None]] = [
        (target, named, None),
        (other, rival, None),
    ]
    assert _title_occurrences_outmatched(target, text, names)
    assert _title_occurrences_outmatched(other, text, names)


async def test_candidate_normalisation_is_shared_across_law_lookups() -> None:
    import uuid
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, Mock, patch

    from codify.storage import resolve_refs as resolver

    first, second = uuid.uuid4(), uuid.uuid4()
    rows = [
        SimpleNamespace(id=first, title="Widgets Act", short_title=None),
        SimpleNamespace(id=second, title="Widgets  Act", short_title=None),
    ]
    session = AsyncMock()
    session.execute.return_value = Mock(all=Mock(return_value=rows))
    cache = resolver._LawCache()
    with patch.object(
        resolver, "_normalised_title_candidates", wraps=resolver._normalised_title_candidates
    ) as prepare:
        assert await resolver._a_longer_name_matches(session, cache, first, "Widgets Act")
        assert await resolver._a_longer_name_matches(session, cache, second, "Widgets Act")
        assert prepare.call_count == 1
        assert session.execute.await_count == 1
    assert cache.names_by_law[first] is cache.names_by_law[second]


def test_registry_provenance_survives_mapper_round_trip() -> None:
    import uuid

    from codify.akn.references import CrossReference
    from codify.storage.mappers import _ref_to_row, _row_to_ref

    ref = CrossReference(
        start_offset=0, end_offset=4, text_snippet="Link", target_uri="/go/123", origin="registry"
    )
    row = _ref_to_row(ref, source_provision_id=uuid.uuid4())
    assert row is not None
    restored = _row_to_ref(row)
    assert isinstance(restored, CrossReference)
    assert restored.origin == "registry"
    assert restored.target_uri == "/go/123"
