# Jurisdiction Profile: Finland (Suomi / Finland)

## Legal System

- **Tradition(s)**: Civil law (Nordic/Scandinavian). Finland's legal system belongs to the Nordic legal tradition, itself a branch of the broader civil law family. It shares strong structural affinities with Swedish, Danish, and Norwegian law; unsurprisingly, given that Finland was part of Sweden from the 13th century until 1809, then an autonomous Grand Duchy of Russia until independence in 1917. Swedish legal influence is therefore foundational, and Swedish remains a constitutionally equal co-official language. The Finnish civil law tradition is codified (no binding precedent doctrine) but the codes are less comprehensive than French or German codes; judges have more interpretive latitude. Academic legal doctrine (_oikeuskirjallisuus_) has traditionally high persuasive authority.
- **Legal continuity**: Continuous legal system since independence (6 December 1917). The 1919 Form of Government (_Hallitusmuoto_) served as the de facto constitution until replaced by the current Constitution of Finland (_Suomen perustuslaki_, 731/1999), which entered into force 1 March 2000. The transition did not invalidate pre-existing statute law. Finland joined the EU on 1 January 1995, which imposed an additional supranational legal layer. The Åland Islands (Ahvenanmaa) have constitutionally guaranteed autonomy with their own legislature (Lagting/Lagtinget), which can enact regional law within defined subject areas.
- **Legislature**: Unicameral parliament: **Eduskunta** (Finnish) / **Riksdag** (Swedish). 200 members (_kansanedustajaa_ / _riksdagsledamöter_) elected by proportional representation for 4-year terms. The Eduskunta exercises sole legislative authority at national level. The Constitutional Law Committee (_Perustuslakivaliokunta_) reviews constitutionality of bills before enactment (ex ante constitutional review). No upper house; no senate. The Grand Committee (_Suuri valiokunta_ / _Stora utskottet_) handles EU affairs. The Åland Islands send 1 member to the Eduskunta.
- **Head of state role**: Presidential assent (_vahvistaminen_). Bills passed by the Eduskunta are submitted to the **President of the Republic** (_Tasavallan presidentti_) for assent. If the President declines to assent within 3 months, the bill returns to the Eduskunta. If the Eduskunta re-passes it unchanged, it enters into force without presidential assent. Decrees (_asetukset_) are issued by the President or the Government (_Valtioneuvosto_) under delegating authority from acts.
- **Official languages**: **Finnish** (_suomi_) and **Swedish** (_svenska_) are both constitutionally equal official languages (Constitution, s. 17). Both language versions of acts are equally authoritative originals; this is not a translation relationship. In practice, Finnish is the dominant drafting language for most legislation. The Åland Islands are constitutionally monolingual Swedish. Sami languages have official status in northernmost municipalities for administrative purposes (not legislative).
- **Calendar system**: Gregorian. No ambiguity. All FRBR URI dates use Gregorian calendar.
- **Official gazette**: **Suomen säädöskokoelma** (Finnish) / **Finlands författningssamling** (Swedish): "Statutes of Finland". Published by the Ministry of Justice (_Oikeusministeriö_) / the Finnish Government. All acts (_lait_), decrees (_asetukset_), and other normative instruments must be published in the Suomen säädöskokoelma to have legal force. Each act is assigned a sequential number within the calendar year (e.g., _laki 731/1999_). Previously published as a physical gazette; now available electronically via Finlex. The Finnish Government Decree (_valtioneuvoston asetus_) and Presidential Decree (_tasavallan presidentin asetus_) are also published here.
- **Existing digital presence**: see "Digital Presence: Tier 1" below. Finlex (finlex.fi), the official online legal database operated by the Ministry of Justice, also provides: (1) current consolidated versions (_ajantasa_) of acts in structured HTML with section-level anchors, (2) original enacted versions (_alkuperainen_), (3) Swedish language versions (_ruotsinkielinen käännös_), (4) English unofficial translations for many major acts. No Laws.Africa coverage. WIPO Lex covers Finnish IP legislation.

## ISO Codes

- **Country**: `fi` (ISO 3166-1 alpha-2)
- **Languages**: `fin` (Finnish, ISO 639-3), `swe` (Swedish, ISO 639-3)
- **Authoritative language**: Both equally authoritative. Finnish dominant in practice for the majority of legislation.

## Digital Presence: Tier 1 (pure AKN native)

[Finlex Open Data](https://opendata.finlex.fi/finlex/avoindata/v1/akn/fi/act/statute/2000/731) returns native Akoma Ntoso XML. `GET opendata.finlex.fi/finlex/avoindata/v1/akn/fi/act/statute/2000/731` returns `Content-Type: application/xml` with root element `<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">`, full `<FRBRWork>` / `<FRBRExpression>` blocks, and Finlex-namespaced attributes for parliamentary metadata. Finland joins Italy (Normattiva) and the UK (legislation.gov.uk) in the AKN-native top tier.

The Finlex API returns _only_ AKN XML per its documentation: no PDF or HTML alternate, which makes this the strongest possible Tier 1 signal. Ingest is direct import, not extract-and-repair.

## Numbering Conventions

- **Act numbering**: Sequential number within calendar year, format `NNN/YYYY` (e.g., `731/1999`, `86/2023`, `1/1999`). This is the canonical citation for all Finnish legislation. Example citations: _perustuslaki 731/1999_, _sairausvakuutuslaki 1224/2004_, _rikoslaki 39/1889_ (Criminal Code, dating from 1889; the same act, heavily amended). Numbers restart at 1 each calendar year. Very high-numbered acts exist in prolific years (e.g., 1995 had over 1500 acts due to EU accession harmonisation).
- **Section numbering** (pykälät): Sections (_pykälät_, singular _pykälä_) are the basic unit of Finnish legislation. They are numbered with the § symbol followed by an Arabic numeral: **§ 1**, **§ 2**, **§ 3**. Numbering is continuous throughout the act regardless of chapter divisions; it does not restart per chapter. Sections are cited as "1 §", "5 §", "10 §" (number before §). In cross-references within legislation: "1 §:ssä" (in section 1), "5 §:n mukaan" (according to section 5).
- **Subdivision numbering**:
  - **Momentti** (subsection/paragraph of a pykälä): Numbered with Arabic numerals in parentheses: (1), (2), (3). These are called _momentti_ (plural _momentit_) in Finnish. The first momentti is often unnumbered in older legislation but numbered in modern practice. Cited as "1 §:n 1 momentti" or "1 §:n 2 momentti".
  - **Kohta** (point/subparagraph): Numbered with Arabic numerals followed by a closing parenthesis: 1), 2), 3); or with letters: a), b), c). These are called _kohdat_ (singular _kohta_). Note: Finnish uses Arabic numerals for kohta, not always letters.
  - **Alakohta** (sub-point): Lettered a), b), c) or with dashes. Less common.
- **Chapter numbering** (luvut): Chapters (_luvut_, singular _luku_) are numbered with Arabic numerals: **1 luku**, **2 luku**, **3 luku**. Chapters carry descriptive headings. Numbering is continuous throughout the act.
- **SI/Decree numbering**: Government Decrees (_valtioneuvoston asetukset_, VNA) and Presidential Decrees (_tasavallan presidentin asetukset_, TPA) use the same `NNN/YYYY` numbering scheme as acts, in the same sequential register. Presidential orders and ministry decisions use the same system. All share the single Suomen säädöskokoelma number sequence.
- **Dual numbering?**: No. Finland uses a single `NNN/YYYY` system throughout. Old acts from before 1980 may have older chapter-based identifiers in some legal databases but `NNN/YYYY` is the canonical reference.
- **Insertion numbering**: Uses alphanumeric insertion: e.g., § 5 a, § 5 b (written with space: "5 a §", "5 b §"). Chapters use similar convention: "1 a luku". Sections are not renumbered on amendment.

## FRBR URI Patterns

- **Acts**: `/akn/fi/act/{year}/{number}`
- **Government Decrees (VNA)**: `/akn/fi/act/vna/{year}/{number}`
- **Presidential Decrees (TPA)**: `/akn/fi/act/tpa/{year}/{number}`
- **Ministry Decisions (päätös)**: `/akn/fi/act/paatos/{year}/{number}`
- **Bills (hallituksen esitys, HE)**: `/akn/fi/bill/{year}/{number}`
- **Constitution**: `/akn/fi/act/constitution/1999/731`
- **Åland Regional Acts**: `/akn/fi-ax/act/{year}/{number}` (see Notes on Åland)

**Note on number segment**: The `{number}` uses the Finnish sequential number without leading zeros. E.g., act 731/1999 → `/akn/fi/act/1999/731`; act 86/2023 → `/akn/fi/act/2023/86`.

### Sample URIs

Constructed from the numbering conventions above rather than read off live documents.

| Document                                                                                                                    | URI                         |
| --------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Constitution of Finland (_Suomen perustuslaki_ 731/1999)                                                                    | `/akn/fi/act/1999/731`      |
| Criminal Code (_Rikoslaki_ 39/1889)                                                                                         | `/akn/fi/act/1889/39`       |
| Act on the Exercise of Freedom of Expression in Mass Media (_Laki sananvapauden käyttämisestä joukkoviestinnässä_ 460/2003) | `/akn/fi/act/2003/460`      |
| Personal Data Act (_Henkilötietolaki_ 523/1999, now repealed)                                                               | `/akn/fi/act/1999/523`      |
| Data Protection Act (_Tietosuojalaki_ 1050/2018)                                                                            | `/akn/fi/act/2018/1050`     |
| Sickness Insurance Act (_Sairausvakuutuslaki_ 1224/2004)                                                                    | `/akn/fi/act/2004/1224`     |
| HE 309/1993 vp (Government Bill for Constitution reform)                                                                    | `/akn/fi/bill/1993/309`     |
| Government Decree on vehicle taxation                                                                                       | `/akn/fi/act/vna/2021/1226` |

## Hierarchy Mapping

| Local Term (Finnish / Swedish) | AKN Element    | Level            | eId Abbreviation | Notes                                                                                                                                                                                                                                                                                        |
| ------------------------------ | -------------- | ---------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Osa / Del                      | `part`         | Highest division | `part`           | Rarely used; appears in the largest codes (e.g., Criminal Code). Arabic numerals: "1 osa", "2 osa". Heading on separate line.                                                                                                                                                                |
| Luku / Kapitel                 | `chapter`      | Higher division  | `chap`           | **Primary higher division**. Arabic numerals: "1 luku", "2 luku". Descriptive heading follows. Used in virtually all acts of any length.                                                                                                                                                     |
| Pykälä (§) / Paragraf (§)      | `article`      | **Basic unit**   | `art`            | The fundamental citable unit. § symbol with Arabic numeral: "1 §", "5 §". Continuous numbering throughout act. Heading (otsikko) above each pykälä in modern acts.                                                                                                                           |
| Momentti / Moment              | `subsection`   | Subdivision L1   | `subsec`         | Numbered (1), (2), (3) in modern acts. First momentti may be unnumbered in pre-1980 legislation. Cited as "1 §:n 1 momentti".                                                                                                                                                                |
| Kohta / Punkt                  | `paragraph`    | Subdivision L2   | `para`           | Listed items within a momentti. Numbered 1), 2), 3) with Arabic numerals or a), b), c) with letters. Cited as "1 kohta", "2 kohta". Which of the two forms predominates, and whether the eId should follow the numeric or the alphabetic designation, is not yet settled against the source. |
| Alakohta / Underpunkt          | `subparagraph` | Subdivision L3   | `subpara`        | Sub-items within a kohta. Lettered a), b), c) or with dashes. Less common in modern drafting.                                                                                                                                                                                                |
| Liite / Bilaga                 | `attachment`   | Attachment       | `att`            | Appendices/Schedules. Named "Liite 1", "Liite 2" or "Liite" (single). Wrapped in `<attachments><attachment>`.                                                                                                                                                                                |
| Otsikko / Rubrik               | `heading`      | Inline           | --               | Section heading above each pykälä. In modern Finlex presentation, each pykälä has a heading. Maps to `<heading>` within `<section>`.                                                                                                                                                         |
| Pitkä nimike / Lång rubrik     | `longTitle`    | Preface          | --               | Long title of the act. In `<preface><longTitle>`.                                                                                                                                                                                                                                            |
| Johdantolause / Inledning      | `formula`      | Preamble         | --               | Enacting formula. In `<preamble><formula name="enactingFormula">`.                                                                                                                                                                                                                           |

**Key structural notes:**

- Finnish acts do NOT use "Part" (osa) except in the largest codification statutes. The typical structure is: **Luku** (chapter) → **§** (section) → **Momentti** (subsection) → **Kohta** (point). Smaller acts may omit chapters entirely (luku-free).
- The local term is "pykälä" (§), not "article", but per the jurisdiction config the AKN element is `article`: pykälä is the smallest independently citable unit and numbering is sequential, structurally equivalent to the civil law article. The § symbol is the visual marker, not a change in AKN element type.
- Swedish-language versions use identical structure with Swedish terminology. The § symbol is universal to both Finnish and Swedish versions.

## eId Conventions

- **Basic unit abbreviation**: `art` (for pykälä / §)
- **Chapter abbreviation**: `chap` (for luku)
- **Insertion numbering**: Alphanumeric with space: "5 a §" → eId `art_5a`; "1 a luku" → eId `chap_1a`
- **Momentti numbering**: Arabic numerals → eId `subsec_1`, `subsec_2`
- **Kohta numbering**: Depends on whether numeric (1), 2)) or alpha (a), b)):
  - Numeric kohta → eId `para_1`, `para_2`
  - Alpha kohta → eId `para_a`, `para_b`

**Example eId chains:**

```
Chapter 2, Section 5, Subsection 1, Point (a):
  chap_2__art_5__subsec_1__para_a

Chapter 3, Section 10a, Subsection 2:
  chap_3__art_10a__subsec_2

Chapter 1a, Section 3, Subsection 1:
  chap_1a__art_3__subsec_1

No chapter (luku-free act), Section 7, Subsection 2, Point 3:
  art_7__subsec_2__para_3

Schedule 1:
  att_1

Section 4 within Schedule 1:
  att_1__art_4
```

## Enacting Formula

Finnish legislative tradition does not use a single standardised enacting formula in the way Commonwealth jurisdictions do. The enacting formula varies by instrument type:

**Acts passed by the Eduskunta** (current standard):

```
Eduskunnan päätöksen mukaisesti säädetään:
```

Swedish: _I enlighet med riksdagens beslut föreskrivs:_
English translation: "In accordance with the decision of the Parliament, it is enacted:"

**Alternative formulation** (used when act amends another):

```
Eduskunnan päätöksen mukaisesti
muutetaan [act reference] [section reference]:
```

**Presidential/Government Decrees** (not acts; enacted by executive):

```
Valtioneuvoston asetuksella säädetään [act reference] nojalla:
```

(Government Decree: "By Government Decree, pursuant to [act], it is provided:")

**Constitutional acts** (require 2/3 majority, or simple majority if declared urgent):
Same Eduskunta formula but the preamble records the constitutional procedure.

**Position**: Beginning of the act body, before the first section (§ 1). Formatted as a standalone paragraph before "1 §". Maps to `<preamble><formula name="enactingFormula">`.

**Historical note**: Older acts (pre-1990s) may use a longer formula referencing the President of the Republic's role more explicitly, reflecting the stronger presidential system of the pre-2000 Constitution. The 2000 Constitution reduced presidential power substantially.

## Amendment Patterns

- **Style**: Textual amendment: the amending act explicitly quotes the replacement text.
- **Typical phrasing** (Finnish):
  - Substitution: _"muutetaan [act NNN/YYYY] [§ reference] momentti seuraavasti:"_ (amended to read as follows)
  - Addition: _"lisätään [act NNN/YYYY] uusi [§ reference] seuraavasti:"_ (new section added as follows)
  - Repeal: _"kumotaan [act NNN/YYYY] [§ reference]"_ (is repealed)
- **Cross-reference format**: Acts are cited by their Suomen säädöskokoelma number: _"laki NNN/YYYY"_ or the short title if one exists. Example: _"sairausvakuutuslakia (1224/2004)"_
- **Consolidated versions**: Finlex maintains up-to-date consolidated (_ajantasa_) versions. The current consolidated text shows amendments integrated; the original (_alkuperäinen_) text shows the act as enacted. Both are available.
- **Amendment acts**: Amendment acts receive their own `NNN/YYYY` number and are separate FRBR works. The relationship between amending and amended act is recorded in `<activeModifications>` / `<passiveModifications>` metadata.

## Institutional TLCs

| Entity                                                | TLC Class       | eId                      | href                                      | showAs                 |
| ----------------------------------------------------- | --------------- | ------------------------ | ----------------------------------------- | ---------------------- |
| Eduskunta (Parliament)                                | TLCOrganization | `eduskunta`              | `/ontology/org/fi/eduskunta`              | Eduskunta              |
| Tasavallan presidentti (President)                    | TLCRole         | `president`              | `/ontology/role/fi/president`             | Tasavallan presidentti |
| Valtioneuvosto (Government / Cabinet)                 | TLCOrganization | `valtioneuvosto`         | `/ontology/org/fi/valtioneuvosto`         | Valtioneuvosto         |
| Oikeusministeriö (Ministry of Justice)                | TLCOrganization | `oikeusministerio`       | `/ontology/org/fi/oikeusministerio`       | Oikeusministeriö       |
| Suomen säädöskokoelma                                 | TLCObject       | `saadoskokoelma`         | `/ontology/obj/fi/saadoskokoelma`         | Suomen säädöskokoelma  |
| Codify                                                | TLCOrganization | `codify`                 | `/ontology/org/codify`                    | Codify                 |
| Riksdag för Åland (Åland Legislature)                 | TLCOrganization | `lagtinget`              | `/ontology/org/fi-ax/lagtinget`           | Lagting                |
| Perustuslakivaliokunta (Constitutional Law Committee) | TLCOrganization | `perustuslakivaliokunta` | `/ontology/org/fi/perustuslakivaliokunta` | Perustuslakivaliokunta |

## Bluebell Keyword Mapping

| Local Term                | AKN Element    | Bluebell Keyword | Notes                                    |
| ------------------------- | -------------- | ---------------- | ---------------------------------------- |
| Osa (Part)                | `part`         | `PART`           | Rarely used; only in very large codes    |
| Luku (Chapter)            | `chapter`      | `CHAPTER`        | Primary higher division                  |
| § / Pykälä (Article)      | `article`      | `ARTICLE`        | Basic unit; § symbol is presentational   |
| Momentti (Subsection)     | `subsection`   | `SUBSECTION`     | Numbered (1), (2), (3)                   |
| Kohta (Point/Paragraph)   | `paragraph`    | `PARAGRAPH`      | Items within a momentti                  |
| Alakohta (Sub-point)      | `subparagraph` | `SUBPARAGRAPH`   | Sub-items within a kohta                 |
| Liite (Schedule/Appendix) | `attachment`   | `SCHEDULE`       | Bluebell SCHEDULE maps to AKN attachment |

**Structuring prompt additions**: Finland-specific `structuring.prompt_additions`, which the structurer appends to its jurisdiction context, would:

1. Reflect that `CHAPTER` (luku) is the primary higher division (not PART)
2. Handle the § symbol as the article marker (normalising "5 §" to an ARTICLE heading)
3. Handle Finnish/Swedish bilingual structure; both language versions use identical hierarchy
4. Correctly identify the Eduskunta enacting formula as the boundary between preamble and body

## Applicable Supranational Frameworks

| Body                                             | Member?                    | Direct effect?                                                                              | Subject areas affected                                                                                                                                                                         | Profile reference          |
| ------------------------------------------------ | -------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| **European Union**                               | Yes (since 1 January 1995) | Regulations: yes (directly applicable). Directives: require transposition into Finnish law. | Comprehensive: competition, product safety, food, environment, data protection (GDPR as EU Regulation 2016/679 applies directly), financial services, employment, consumer protection, customs | `/akn/eu/`: AKN4EU profile |
| **Council of Europe**                            | Yes                        | ECHR binding via judgment; Convention requires domestic implementation                      | Human rights, fair trial, privacy                                                                                                                                                              | Not separately profiled    |
| **United Nations**                               | Yes                        | SC Chapter VII resolutions binding; treaties require ratification + domestic implementation | International obligations, sanctions                                                                                                                                                           | `/akn/un/`: AKN4UN profile |
| **Nordic Council / Nordic Council of Ministers** | Yes                        | Non-binding recommendations only; Nordic conventions require ratification                   | Social security reciprocity, passport-free travel, labour market mobility                                                                                                                      | Not separately profiled    |

**EU law note**: GDPR (EU Regulation 2016/679) applies directly in Finland without transposition. The Finnish Data Protection Act (1050/2018) supplements but does not transpose GDPR; it exercises national margins of discretion. This pattern repeats across many EU Regulations. When displaying Finnish data protection law, the EU layer is essential context.

## Document Types Present

- [x] Primary legislation: Acts (_lait_) enacted by the Eduskunta
- [x] Government Decrees (_valtioneuvoston asetukset_, VNA): subsidiary legislation with delegated authority
- [x] Presidential Decrees (_tasavallan presidentin asetukset_, TPA): subsidiary legislation
- [x] Ministry Decisions (_ministeriön päätökset_) and Regulations (_ministeriön asetukset_): subsidiary
- [x] Bills: Government Bills (_hallituksen esitykset_, HE) + Private Member Bills (_lakialoitteet_)
- [x] Constitutional instruments: Constitution (731/1999) and constitutional amendments
- [x] Parliamentary debates: Eduskunnan täysistunnon pöytäkirjat (Hansard equivalent)
- [x] Åland regional legislation: enacted by the Lagting under autonomous competence
- [ ] Court judgments: out of scope for legislation ingestion

## Bilingual Modelling

Finland's bilingual equal-authority model requires care:

- Finnish and Swedish versions are **parallel originals**, not translations. Use separate FRBR expressions, not `<FRBRtranslation>`.
- Finnish expression: `/akn/fi/act/1999/731/fin@2000-03-01`
- Swedish expression: `/akn/fi/act/1999/731/swe@2000-03-01`
- Both expressions share the same FRBR work URI: `/akn/fi/act/1999/731`
- Where only one language version can be retrieved, the other's availability is recorded in metadata; neither is a translation of the other.
- Unofficial English translations on Finlex are NOT authoritative expressions. Model as `<FRBRtranslation>` pointing to the Finnish expression as source.
- English expression (unofficial): `/akn/fi/act/1999/731/eng@2000-03-01` with `<FRBRtranslation>` metadata.

## Åland Islands Autonomous Legislation

The Åland Islands (Ahvenanmaa in Finnish; Ahvenanmaa/Åland) have guaranteed autonomy under the Autonomy Act (1991/1144, itself a Finnish national act). The Åland Lagting enacts regional legislation (_landskapslag_) in Swedish within its competence areas (education, local tax, radio/TV, municipal administration, etc.).

- **ISO 3166-2**: `FI-01` (Åland). FRBR locality code: `fi-ax` (using IANA/ISO 3166-1 alpha-2 for Åland as territory: `ax`)
- **URI pattern**: `/akn/fi-ax/act/{year}/{number}` for Lagting acts
- **Language**: Swedish only (Åland is constitutionally monolingual Swedish)
- **Official gazette**: _Ålands författningssamling_ (ÅFS): separate from the Finnish säädöskokoelma
- **Pipeline note**: Åland legislation is a separate pipeline configuration requiring the `fi-ax` sub-jurisdiction. Finnish national acts also apply in Åland where competence is not devolved.

## Digitisation Maturity Target

The levels below are this project's incremental-digitisation ladder, not part of the
OASIS AKN standard, which defines its own naming-convention compliance levels.

- **Initial**: Level 1 (valid AKN structure + FRBR identification + publication metadata)
- **Short-term target**: Level 2 (cross-references between acts; Finlex's structured HTML makes link extraction feasible)
- **Long-term target**: Level 3 (lifecycle events; amendment chains are traceable via Finlex consolidated versions)

## Unresolved Ambiguities

1. **Kohta numbering scheme**: Finnish acts use both Arabic-numeric kohdat (1), 2), 3)) and alphabetic kohdat (a), b), c)). The choice of eId (`para_1` vs `para_a`) depends on which convention a specific act uses. A runtime detection rule is needed: if kohdat use numerals, map to `para_1`, `para_2`; if letters, map to `para_a`, `para_b`. Which convention predominates in modern drafting has not been checked against Finlex.

2. **Momentti unnumbered in first position**: Older acts (pre-1990) may have the first momentti unnumbered (i.e., the first paragraph of a § with no "(1)" marker), with subsequent momentin numbered (2), (3). The pipeline needs to handle this and assign `subsec_1` to the unnumbered first momentti. How common this remains in post-2000 acts has not been measured; modern drafting guidance appears to mandate explicit numbering throughout.

3. **§ numbering with spaces in insertion**: Finnish inserted sections are cited as "5 a §" (with space before letter). The eId should normalise to `art_5a` (no space). Whether that is the pattern Finlex uses, and whether double insertions such as "5 b §" occur, has not been checked.

4. **Luku-free acts**: Short acts (under ~10 pykälät) often have no luku divisions at all; the act goes directly from the enacting formula to §1 with no chapter wrapper. A structuring model must therefore not require a chapter level. This is a common Finnish drafting pattern, though its prevalence has not been measured.

5. **Presidential Decree vs Government Decree pipeline differentiation**: Both TPA and VNA use the same säädöskokoelma numbering. The type can be detected from the document header ("Tasavallan presidentin asetus" vs "Valtioneuvoston asetus"). The pipeline should classify accordingly. URI subtype (`tpa` vs `vna`) needs to be assigned based on header detection, not number pattern.

6. **Åland sub-jurisdiction**: Confirm whether `fi-ax` is the appropriate locality code or whether Åland should be modelled as a fully separate jurisdiction with country code `ax`. Under ISO 3166-1, Åland has its own alpha-2 code `AX`. Given that Åland has a fully separate legislature and legal system, treating it as `ax` rather than `fi-ax` may be more appropriate. Recommended: `ax`, the independent ISO 3166-1 code. Unresolved.

7. **English translations on Finlex**: Finlex hosts unofficial English translations for many major acts. They are not authoritative and must never be presented as such. Confirm metadata handling to mark `<FRBRtranslation>` and display a disclaimer wherever they are shown.

## Notes

- **Numbering direction**: Finnish statutory citation reads number-before-symbol: "5 §", not "§ 5". The pipeline should handle both input formats but canonicalise to the Finnish citation form.
- **"Laki" vs "Asetus"**: Finnish distinguishes _laki_ (act of parliament) from _asetus_ (decree). Both are published in the säädöskokoelma. The pipeline classification step should distinguish these as different document subtypes.
- **Two Finlex surfaces**: the Open Data API above returns AKN XML and is the route to prefer. The finlex.fi website serves the same statutes as structured HTML with clear section markers; where only that is available, its DOM is more reliable than PDF OCR, and an HTML extraction pathway, rather than the full model pipeline, may be appropriate.
- **Very long-lived acts**: The Finnish Criminal Code (rikoslaki 39/1889) has been in force since 1889 and has been amended hundreds of times. The FRBR Work URI `/akn/fi/act/1889/39` is correct; the expression URI will reflect the current consolidation date. Act numbers in 1889 were assigned in a pre-modern system; the `NNN/YYYY` system as it currently exists was formalised in the 20th century, in a year this profile has not established.
- **Nordic harmony**: Finnish legislation frequently mirrors Danish, Norwegian, and Swedish equivalents due to Nordic legal cooperation via the Nordic Council. This may assist cross-jurisdiction quality checks.
- **Repealed acts**: Finnish law is regularly consolidated and outdated provisions repealed. Finlex marks repealed provisions clearly, so repeal status is available as source metadata.
- **EU transposition acts**: A significant proportion of Finnish legislation since 1995 transposes EU Directives. These acts often note the Directive in the preamble or explanatory materials (_hallituksen esitys_). This relationship should be modelled as a `TLCReference` pointing to the EU instrument.

## Research Sources

- Finlex.fi: Official Finnish legal database (Ministry of Justice). Finnish and Swedish versions of all legislation. URL: https://www.finlex.fi
- Finnish Ministry of Justice legislative drafting guidelines: _Lainlaatijan opas_ (Drafting Guide), available via oikeusministerio.fi
- Constitution of Finland (731/1999): English translation at finlex.fi/en
- Nordic legal tradition: Bogdan, M. (ed.), _Nordic Law — Between Tradition and Dynamism_ (2010)
- JuriGlobe classification: University of Ottawa, www.juriglobe.ca; Finland classified under Nordic civil law
- Eur-lex.europa.eu: EU legislation applicable in Finland
- Åland Islands official portal: www.regeringen.ax; Lagting acts and Ålands författningssamling
- General knowledge of Finnish legislative structure, a well-documented jurisdiction. Where direct document inspection was not possible, the claim says so in place.
