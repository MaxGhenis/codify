"""Numeric-token extraction + sentinel encode/decode round-trip tests."""

from __future__ import annotations

import pytest

from codify.translate.numeric_extract import (
    decode_sentinels,
    encode_sentinels,
    extract_tokens,
    find_stray_sentinels,
)


class TestExtractTokens:
    def test_bare_digits_extracted(self):
        m = extract_tokens("The period is 30 days from receipt.")
        assert len(m.tokens) == 1
        assert m.tokens[0].kind == "digits"
        assert m.tokens[0].surface == "30"
        assert m.tokens[0].expected_target_surface == "30"

    def test_arabic_indic_digits_folded_to_ascii(self):
        # Arabic-Indic ٣٠ (U+0663 U+0660) → 30
        m = extract_tokens("مدة ٣٠ يوماً من تاريخ الاستلام.")
        assert len(m.tokens) == 1
        assert m.tokens[0].surface == "٣٠"
        assert m.tokens[0].expected_target_surface == "30"

    def test_article_ref_english_labelled(self):
        m = extract_tokens("Any person who violates the provisions of Article 62.")
        assert any(t.kind == "article_ref" for t in m.tokens)
        art = next(t for t in m.tokens if t.kind == "article_ref")
        assert art.surface == "Article 62"
        # Label-free target surface
        assert art.expected_target_surface == "62"

    def test_article_ref_arabic_labelled(self):
        m = extract_tokens("يعاقب كل من يخالف أحكام المادة (14) من هذا القانون.")
        art = next(t for t in m.tokens if t.kind == "article_ref")
        assert "14" in art.surface
        assert art.expected_target_surface == "14"

    def test_statute_ref_arabic(self):
        m = extract_tokens("قانون رقم 41 لسنة 2042 بشأن المرصد")
        stat = next(t for t in m.tokens if t.kind == "statute_ref")
        # Statute ref contains both the number and year in expected surface
        assert "41" in stat.expected_target_surface
        assert "2042" in stat.expected_target_surface

    def test_statute_ref_english(self):
        m = extract_tokens("issued under Law No. 41 of 2042 concerning the Observatory.")
        stat = next(t for t in m.tokens if t.kind == "statute_ref")
        assert "41" in stat.expected_target_surface

    def test_money_with_currency_english(self):
        m = extract_tokens("shall be punished with 1000 Zerzuran Dinars.")
        money = next(t for t in m.tokens if t.kind == "money")
        assert money.expected_target_surface == "1000"

    def test_money_arabic_currency(self):
        m = extract_tokens("غرامة مالية قدرها 500 دينار.")
        money = next(t for t in m.tokens if t.kind == "money")
        assert money.expected_target_surface == "500"

    def test_date_slash_form(self):
        m = extract_tokens("in its session held on 3/2/2000 AD, we have issued")
        date = next(t for t in m.tokens if t.kind == "date")
        assert date.expected_target_surface == "3/2/2000"

    def test_arabic_indic_date(self):
        m = extract_tokens("بتاريخ ٣/٢/٢٠٠٠")
        date = next(t for t in m.tokens if t.kind == "date")
        assert date.expected_target_surface == "3/2/2000"

    def test_synthetic_fee_range_shape(self):
        """An invented fee schedule preserves its citation and both monetary bounds."""
        source = (
            "The fictional observatory applies Article (47) when setting a deposit "
            "between 2400 Dinars and 6800 Dinars for borrowed instruments."
        )
        m = extract_tokens(source)
        kinds = [t.kind for t in m.tokens]
        assert "article_ref" in kinds
        # Two monetary amounts (min + max) plus the article ref
        money_tokens = [t for t in m.tokens if t.kind == "money"]
        assert len(money_tokens) == 2
        surfaces = {t.expected_target_surface for t in money_tokens}
        assert surfaces == {"2400", "6800"}

    def test_reading_order_sentinel_ids(self):
        m = extract_tokens("30 days after Article 5, but before 1999.")
        # IDs allocated in reading order
        ids = [t.sentinel_id for t in m.tokens]
        assert ids == sorted(ids)

    def test_start_id_offset(self):
        m = extract_tokens("30 days", start_id=100)
        assert m.tokens[0].sentinel_id == "N100"

    def test_empty_text_returns_empty_manifest(self):
        m = extract_tokens("")
        assert m.tokens == []

    def test_no_digits_returns_empty(self):
        m = extract_tokens("The environmental committee shall convene.")
        assert m.tokens == []


@pytest.mark.parametrize(
    "phrase",
    [
        "1000 Dinars",
        "1000 Zerzuran Dinars",
        "1000 Dirhams",
        "1000 Riyals",
        "1000 Dollars",
        "1000 Shekels",
        "1000 Euros",
        "1000 Pounds",
        "1000 Liras",
        "1000 Lire",
        "1000 Francs",
        "1000 دينار",
        "1000 دنانير",
        "1000 درهم",
        "1000 ريال",
        "1000 دولار",
        "1000 شيقل",
        "1000 جنيه",
        "1000 ليرة",
        "1000 يورو",
        "1000 فرنك",
        "1000 שקל",
        "1000 דינר",
        "1000 דולר",
        "1000 יורו",
        "1000 לירה",
    ],
)
def test_every_listed_currency_marks_money(phrase: str) -> None:
    kinds = {
        t.kind: t.expected_target_surface for t in extract_tokens(f"a fine of {phrase}.").tokens
    }
    assert kinds.get("money") == "1000", phrase


def test_a_non_currency_noun_is_not_money() -> None:
    kinds = [t.kind for t in extract_tokens("a fine of 1000 Books.").tokens]
    assert "money" not in kinds


class TestSentinelEncodeDecode:
    def test_encode_places_sentinels_at_token_positions(self):
        source = "30 days after Article 5."
        m = extract_tokens(source)
        encoded = encode_sentinels(source, m)
        # Every sentinel appears in the encoded text
        for tok in m.tokens:
            assert m.sentinel_for(tok) in encoded

    def test_decode_restores_target_surfaces(self):
        source = "Article 62 shall punish with 1000 Zerzuran Dinars."
        m = extract_tokens(source)
        encoded = encode_sentinels(source, m)
        # Simulate LLM that preserves sentinels verbatim
        decoded, missing = decode_sentinels(encoded, m)
        assert missing == []
        assert "62" in decoded
        assert "1000" in decoded

    def test_round_trip_arabic_indic_digits(self):
        """Source has ٣٠, decoded output has 30 (ASCII-folded)."""
        source = "مدة ٣٠ يوماً"
        m = extract_tokens(source)
        encoded = encode_sentinels(source, m)
        decoded, missing = decode_sentinels(encoded, m)
        assert missing == []
        assert "30" in decoded
        assert "٣٠" not in decoded

    def test_missing_sentinel_reported(self):
        source = "Article 5 and Article 10."
        m = extract_tokens(source)
        encoded = encode_sentinels(source, m)
        # Simulate LLM dropping the second sentinel
        second_sentinel = m.sentinel_for(m.tokens[1])
        broken = encoded.replace(second_sentinel, "")
        decoded, missing = decode_sentinels(broken, m)
        assert m.tokens[1].sentinel_id in missing
        assert len(missing) == 1

    def test_fallback_delimiter_when_primary_in_source(self):
        """If the source somehow contains ⟨ or ⟩, use the fallback pair."""
        # Rare in practice but must be handled
        source = "⟨decorative⟩ Article 5."
        m = extract_tokens(source)
        assert m.open_char == "‹"
        assert m.close_char == "›"

    def test_encode_leaves_prose_untouched(self):
        source = "The Minister shall issue Regulations."
        m = extract_tokens(source)
        encoded = encode_sentinels(source, m)
        assert encoded == source  # No sentinels means no change

    def test_no_index_shift_on_multiple_tokens(self):
        source = "Between Article 5 and Article 10, within 30 days."
        m = extract_tokens(source)
        encoded = encode_sentinels(source, m)
        # Every source token has been replaced
        for tok in m.tokens:
            assert tok.surface not in encoded or tok.expected_target_surface != tok.surface


class TestStraySentinelDetection:
    def test_detects_primary_delimiter_sentinel(self):
        text = "The output contains ⟨N005⟩ which shouldn't be here."
        assert find_stray_sentinels(text) == ["⟨N005⟩"]

    def test_detects_fallback_delimiter_sentinel(self):
        text = "The output contains ‹N005› which shouldn't be here."
        assert find_stray_sentinels(text) == ["‹N005›"]

    def test_no_sentinels_returns_empty(self):
        assert find_stray_sentinels("plain prose with no markers") == []

    def test_ignores_angle_quotes_without_id(self):
        # Bare quotation marks without an N-ID inside must not be flagged
        assert find_stray_sentinels("She said ⟨hello⟩ warmly.") == []


class TestSyntheticFeeFixture:
    """Invented instrument deposits exercise missing citations and lower bounds."""

    def test_fee_schedule_full_round_trip(self):
        source = (
            "For telescope loans under Article (47), the fictional observatory "
            "requires a deposit between 2400 Dinars and 6800 Dinars, "
            "refundable after eight weeks."
        )
        manifest = extract_tokens(source)
        encoded = encode_sentinels(source, manifest)
        decoded, missing = decode_sentinels(encoded, manifest)
        assert missing == []
        assert {t.kind for t in manifest.tokens} == {"article_ref", "money"}
        for number in ("47", "2400", "6800"):
            assert number in decoded

    def test_dropped_citation_and_lower_bound_are_reported(self):
        source = "Article (47) requires a deposit of 2400 Dinars to 6800 Dinars."
        manifest = extract_tokens(source)
        encoded = encode_sentinels(source, manifest)
        lost = [t for t in manifest.tokens if t.expected_target_surface in {"47", "2400"}]
        assert len(lost) == 2
        for token in lost:
            encoded = encoded.replace(manifest.sentinel_for(token), "")
        decoded, missing = decode_sentinels(encoded, manifest)
        assert set(missing) == {t.sentinel_id for t in lost}
        assert "6800" in decoded


class TestCompressedCitations:
    """`(N/M)` means article N paragraph M in some Arabic drafting, and a
    model left to paraphrase it reads the slash as a list separator or a
    literal. The whole citation is claimed as one span and decoded from a
    template."""

    def _round_trip(self, src: str, language: str | None) -> str:
        from codify.translate.numeric_extract import (
            decode_sentinels,
            encode_sentinels,
            extract_tokens,
        )

        manifest = extract_tokens(src)
        encoded = encode_sentinels(src, manifest)
        decoded, missing = decode_sentinels(encoded, manifest, target_language=language)
        assert missing == []
        return decoded

    def test_article_slash_paragraph_is_one_token(self) -> None:
        from codify.translate.numeric_extract import extract_tokens

        manifest = extract_tokens("خالف أحكام المادة (4/9) من هذا القانون")
        kinds = [t.kind for t in manifest.tokens]
        assert kinds == ["article_para_ref"]
        # The label rides inside the claimed span, so the model never sees it
        # and cannot drop the word "Article" or invent a second reference.
        assert manifest.tokens[0].surface == "المادة (4/9)"

    def test_renders_as_a_single_english_reference(self) -> None:
        out = self._round_trip("خالف أحكام المادة (4/9) من هذا القانون", "English")
        assert "Article 4 paragraph 9" in out
        assert "4/9" not in out

    def test_paragraph_of_article_keeps_its_order(self) -> None:
        from codify.translate.numeric_extract import extract_tokens

        manifest = extract_tokens("مع مراعاة أحكام الفقرة (3) من المادة (20)")
        assert [t.kind for t in manifest.tokens] == ["para_of_article_ref"]
        out = self._round_trip("مع مراعاة أحكام الفقرة (3) من المادة (20)", "English")
        # Not "Article 3 of Article 20": paragraph and article do not swap.
        assert "Paragraph 3 of Article 20" in out

    def test_hebrew_target_uses_its_own_shape(self) -> None:
        out = self._round_trip("خالف أحكام المادة (4/9)", "Hebrew")
        assert "4" in out and "9" in out
        assert "Article" not in out

    def test_unknown_language_keeps_the_numeric_shape(self) -> None:
        # An unworded citation is recoverable; a mis-worded one is not.
        out = self._round_trip("خالف أحكام المادة (4/9)", "Klingon")
        assert "4/9" in out

    def test_no_language_keeps_the_numeric_shape(self) -> None:
        out = self._round_trip("خالف أحكام المادة (4/9)", None)
        assert "4/9" in out

    def test_plain_article_ref_is_unaffected(self) -> None:
        from codify.translate.numeric_extract import extract_tokens

        manifest = extract_tokens("وفقاً لأحكام المادة (5) من القانون")
        assert [t.kind for t in manifest.tokens] == ["article_ref"]
        assert manifest.tokens[0].expected_target_surface == "5"

    def test_arabic_indic_digits_in_a_compound_citation(self) -> None:
        out = self._round_trip("خالف أحكام المادة (٤/٩)", "English")
        assert "Article 4 paragraph 9" in out


def test_repair_path_decodes_citations_in_the_target_language() -> None:
    # The repair loop re-decodes its own output; without the target language a
    # repaired compound citation falls back to the numeric shape and the
    # deterministic wording is lost on exactly the provisions that needed a fix.
    import inspect

    from codify.translate import repair

    src = inspect.getsource(repair)
    assert "decode_sentinels(line, manifest, target_language=target_language)" in src


def test_compound_citation_drops_the_model_written_noun() -> None:
    """A generated noun must not duplicate the decoded compound label."""
    manifest = extract_tokens("الفقرة (٣) من المادة (٢٠) من هذا القرار")
    sentinel = manifest.sentinel_for(manifest.tokens[0])
    decoded, missing = decode_sentinels(
        f"Article {sentinel} of this Decision by Law shall be amended",
        manifest,
        target_language="English",
    )
    assert decoded.startswith("Paragraph 3 of Article 20 of this Decision by Law")
    assert not missing


def test_compound_citation_drops_a_repeated_noun_of_either_kind() -> None:
    manifest = extract_tokens("المادة (٢/٢٥) من ذات القرار بقانون")
    sentinel = manifest.sentinel_for(manifest.tokens[0])
    decoded, _ = decode_sentinels(
        f"as stated in Article {sentinel} of the same Decree-Law",
        manifest,
        target_language="English",
    )
    assert "Article Article" not in decoded
    assert "in Article 2 paragraph 25 of the same Decree-Law" in decoded


def test_plain_article_ref_keeps_the_noun_the_surface_does_not_supply() -> None:
    """A bare reference decodes to a number, so the model's noun must survive."""
    manifest = extract_tokens("المادة (٢٠) من هذا القرار")
    sentinel = manifest.sentinel_for(manifest.tokens[0])
    decoded, _ = decode_sentinels(
        f"Article {sentinel} of this Decree-Law", manifest, target_language="English"
    )
    assert decoded == "Article 20 of this Decree-Law"


def test_bare_digit_does_not_become_a_citation() -> None:
    """A page-footer digit must not become an invented article reference."""
    manifest = extract_tokens("77 تُحفظ العدسات في صندوق أزرق.")
    sentinel = manifest.sentinel_for(manifest.tokens[0])
    decoded, _ = decode_sentinels(
        f"Article {sentinel} The lenses are kept in a blue box.",
        manifest,
        target_language="English",
    )
    assert decoded.startswith("77 The lenses are kept")


def test_plural_label_governs_its_list_so_members_stay_bare() -> None:
    """A plural citation governs every member of its list."""
    source = "تلغى المواد (9 مكرر 2)، (20)، (27) من القانون الأصلي."
    manifest = extract_tokens(source)
    s = [manifest.sentinel_for(t) for t in manifest.tokens]
    decoded, _ = decode_sentinels(
        f"Articles (Article {s[0]} bis Article {s[1]}), (Article {s[2]}), "
        f"and (Article {s[3]}) shall be repealed.",
        manifest,
        target_language="English",
    )
    assert decoded == "Articles (9 bis 2), (20), and (27) shall be repealed."


def test_singular_label_the_patterns_miss_still_licenses_the_model_noun() -> None:
    """`مكرر` splits the citation, so the label stays in the source for the
    model to translate and its noun must not be stripped as invented."""
    manifest = extract_tokens("بعد المادة (6 مكرر)")
    sentinel = manifest.sentinel_for(manifest.tokens[0])
    decoded, _ = decode_sentinels(
        f"after Article {sentinel} bis)", manifest, target_language="English"
    )
    assert decoded == "after Article 6 bis)"


def test_article_ref_claims_parentheses_in_pairs() -> None:
    """A lone opening paren made the span `المادة (6 ` and shipped `Article 6 bis)`."""
    manifest = extract_tokens("بعد المادة (6 مكرر) تحمل الرقم")
    assert [t.kind for t in manifest.tokens] == ["digits"]
    assert encode_sentinels("بعد المادة (6 مكرر) تحمل الرقم", manifest).count(")") == 1


def test_arabic_proclitic_is_consumed_with_the_label() -> None:
    """`للمادة` left its first lam stranded in front of the sentinel."""
    source = "طبقاً للمادة (61) من النظام الأصلي"
    manifest = extract_tokens(source)
    assert [t.kind for t in manifest.tokens] == ["article_ref"]
    assert "ل⟨" not in encode_sentinels(source, manifest)
    decoded, _ = decode_sentinels(
        f"In accordance with Article {manifest.sentinel_for(manifest.tokens[0])} of the Regulation",
        manifest,
        target_language="English",
    )
    assert decoded == "In accordance with Article 61 of the Regulation"


def test_label_role_survives_a_language_with_no_templates() -> None:
    """Hebrew has no article-ref template, so the number decodes unworded rather
    than guessing a noun the strip rule cannot later de-duplicate."""
    manifest = extract_tokens("المادة (٢٠) من هذا القرار")
    decoded, _ = decode_sentinels(
        manifest.sentinel_for(manifest.tokens[0]), manifest, target_language="Hebrew"
    )
    assert decoded == "20"


def test_statute_ref_claims_spaced_parentheses_as_a_pair() -> None:
    """`قانون رقم ( 7 )` claimed `قانون رقم ( 7`, orphaning the closing paren."""
    source = "قانون رقم ( 7 ) لسنة 1999"
    manifest = extract_tokens(source)
    assert [t.kind for t in manifest.tokens] == ["statute_ref"]
    assert manifest.tokens[0].surface == source
    assert encode_sentinels(source, manifest).count(")") == 0


def test_statute_ref_consumes_the_definite_article() -> None:
    """`القرار بقانون` left `ال` stranded in front of the sentinel."""
    source = "المعدل بموجب القرار بقانون رقم (٧) لسنة ٢٠١٠"
    manifest = extract_tokens(source)
    assert manifest.tokens[0].surface.startswith("القرار")
    assert encode_sentinels(source, manifest) == "المعدل بموجب ⟨N000⟩"


def test_a_three_part_citation_keeps_its_point_letter() -> None:
    """`المادة (28/1/د)` claimed only `المادة (28/1`, stranding `/د)`."""
    source = "المشار إليها في المادة (28/1/د) من القانون"
    manifest = extract_tokens(source)
    encoded = encode_sentinels(source, manifest)
    assert encoded.count("(") == encoded.count(")") == 1
    assert "/د)" in encoded


def test_paragraph_of_article_leaves_a_bis_article_intact() -> None:
    """The compound pattern claimed `المادة (6 `, splitting the bis number."""
    source = "من الفقرة (1) من المادة (6 مكرر) من القانون الأصلي"
    manifest = extract_tokens(source)
    encoded = encode_sentinels(source, manifest)
    assert encoded.count("(") == encoded.count(")")
    assert "مكرر)" in encoded


def test_a_label_separated_by_a_number_word_still_licenses_the_noun() -> None:
    """`المادة رقم (5)` and `Article No. 5` are not claimed by the patterns, so
    the strip rule has to see the label anyway or it deletes a real citation."""
    for source in ("المادة رقم (5) من القانون", "Article No. 5 of the Law"):
        manifest = extract_tokens(source)
        sentinel = manifest.sentinel_for(manifest.tokens[0])
        decoded, _ = decode_sentinels(
            f"Article {sentinel} of the Law", manifest, target_language="English"
        )
        assert decoded == "Article 5 of the Law", source


def test_a_plural_label_does_not_license_a_singular_noun_on_its_members() -> None:
    """`المواد` is not a claimable label, so no member is near one and a
    singular noun in front of a member is the model's own."""
    source = "تلغى المواد (9 مكرر 2)، (20)، (27) من القانون الأصلي."
    manifest = extract_tokens(source)
    s = [manifest.sentinel_for(t) for t in manifest.tokens]
    decoded, _ = decode_sentinels(
        f"Articles (Article {s[0]} bis Article {s[1]}), (Article {s[2]}), "
        f"and (Article {s[3]}) shall be repealed.",
        manifest,
        target_language="English",
    )
    assert decoded == "Articles (9 bis 2), (20), and (27) shall be repealed."


def test_a_label_within_reach_licenses_a_noun_on_a_later_list_member() -> None:
    """Ranges and و-lists put connectives between the label and later members,
    so an adjacency-only rule would delete a real citation noun."""
    source = "المادة (5) الى (10) من القانون"
    manifest = extract_tokens(source)
    s = [manifest.sentinel_for(t) for t in manifest.tokens]
    decoded, _ = decode_sentinels(
        f"Article {s[0]} to Article {s[1]} of the Law",
        manifest,
        target_language="English",
    )
    assert decoded == "Article 5 to Article 10 of the Law"


def test_a_bare_digit_mid_sentence_keeps_the_clause_before_it() -> None:
    """The strip returns everything left of the noun, so a wrong anchor here
    would eat the preceding clause rather than one word."""
    manifest = extract_tokens("77 يكون الوفاء")
    sentinel = manifest.sentinel_for(manifest.tokens[0])
    decoded, _ = decode_sentinels(
        f"as stated in Article {sentinel} performance is due",
        manifest,
        target_language="English",
    )
    assert decoded == "as stated in 77 performance is due"


def test_a_plural_label_does_not_match_through_its_singular() -> None:
    """ "Article" sits inside "Articles", so an unbounded proximity match would
    treat every list member as governed and keep the repeated singular noun."""
    manifest = extract_tokens("Articles 9 and 20 are repealed.")
    assert [t.near_a_label for t in manifest.tokens] == [False, False]
    s = [manifest.sentinel_for(t) for t in manifest.tokens]
    decoded, _ = decode_sentinels(
        f"Articles (Article {s[0]}), (Article {s[1]}) repealed.",
        manifest,
        target_language="English",
    )
    assert decoded == "Articles (9), (20) repealed."
