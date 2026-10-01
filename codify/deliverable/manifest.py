"""The bulk-deliverable manifest: the operator's declaration of what to ingest.

One entry per source PDF. FRBR work URIs are human-decided, never guessed: at
batch scale the classifier's ~85% accuracy fails ~15% of laws every run, so the
operator pins identity here. Validation is strict because a malformed manifest
should fail loudly at upload, not halfway through a 1000-PDF batch.
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from codify.jurisdictions import frbr_country, work_uri_is_homed


class LawEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    frbr_work_uri: str
    # Provenance, carried into the deliverable's quality report so every law cites
    # where it came from and the doctype the URI reflects (e.g. corpus id C06-089,
    # doctype "qanun"). Optional: absent means the source is not catalogued.
    source_id: str | None = None
    source_doctype: str | None = None

    @field_validator("frbr_work_uri")
    @classmethod
    def _akn_prefixed(cls, v: str) -> str:
        if not v.startswith("/akn/"):
            raise ValueError(f"frbr_work_uri must start with /akn/, got {v!r}")
        return v.rstrip("/")


class DeliverableManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    jurisdiction_code: str
    source_language: str
    target_languages: list[str] = Field(default_factory=list)
    laws: list[LawEntry]
    # Grade-and-annotate: translate blocking-grade laws too and ship them graded-C,
    # flagged in the quality report, rather than withholding them. Off by default.
    include_blocking: bool = False

    @field_validator("jurisdiction_code")
    @classmethod
    def _canonical_code(cls, v: str) -> str:
        """Lowered here so the homing check compares like with like: the column
        is case-sensitive and canonicalisation elsewhere is piecemeal."""
        return v.strip().lower()

    @model_validator(mode="after")
    def _no_duplicates(self) -> DeliverableManifest:
        if not self.laws:
            raise ValueError("manifest has no laws")
        files = [law.filename for law in self.laws]
        uris = [law.frbr_work_uri for law in self.laws]
        for label, values in (("filename", files), ("frbr_work_uri", uris)):
            dupes = {v for v in values if values.count(v) > 1}
            if dupes:
                raise ValueError(f"duplicate {label} in manifest: {sorted(dupes)}")
        # Target languages must be distinct and none may equal the source: a
        # source-equal target is a no-op translation that then overwrites the
        # source entry when the deliverable is assembled by language.
        if len(self.target_languages) != len(set(self.target_languages)):
            raise ValueError(f"duplicate target languages: {self.target_languages}")
        if self.source_language in self.target_languages:
            raise ValueError(f"target language equals source: {self.source_language}")
        return self

    @model_validator(mode="after")
    def _laws_homed_in_jurisdiction(self) -> DeliverableManifest:
        """A work URI naming another country files the law under that tenant.

        The entry cannot check itself: only the manifest sees both the URI and
        the jurisdiction it is being ingested under.
        """
        strays = [
            f"{law.filename} -> {law.frbr_work_uri}"
            for law in self.laws
            if not work_uri_is_homed(law.frbr_work_uri, self.jurisdiction_code)
        ]
        if strays:
            expected = frbr_country(self.jurisdiction_code)
            raise ValueError(
                f"work URIs outside jurisdiction {self.jurisdiction_code!r} "
                f"(expected /akn/{expected}/...): {strays}"
            )
        return self

    @classmethod
    def from_yaml(cls, text: str) -> DeliverableManifest:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("manifest must be a YAML mapping")
        return cls.model_validate(data)


if __name__ == "__main__":
    # ponytail: smallest check that fails if validation logic breaks.
    good = DeliverableManifest.from_yaml(
        "jurisdiction_code: xz\nsource_language: ara\ntarget_languages: [en]\n"
        "laws:\n  - {filename: a.pdf, frbr_work_uri: /akn/xz/act/2005/1/}\n"
    )
    assert good.laws[0].frbr_work_uri == "/akn/xz/act/2005/1"  # trailing slash stripped
    _law = "laws:\n  - {filename: a.pdf, frbr_work_uri: /akn/xz/1}\n"
    _hdr = "jurisdiction_code: xz\nsource_language: ara\n"
    for bad in (
        f"{_hdr}laws: []\n",  # empty
        "jurisdiction_code: xz\nsource_language: ara\nlaws:\n"
        "  - {filename: a.pdf, frbr_work_uri: /akn/xz/1}\n"
        "  - {filename: a.pdf, frbr_work_uri: /akn/xz/2}\n",  # dup filename
        f"{_hdr}laws:\n  - {{filename: a.pdf, frbr_work_uri: xz/act/1}}\n",  # missing /akn/
        f"{_hdr}target_languages: [ara]\n{_law}",  # target == source
        f"{_hdr}target_languages: [en, en]\n{_law}",  # dup target
        f"{_hdr}bogus: 1\n{_law}",  # removed/unknown field
    ):
        try:
            DeliverableManifest.from_yaml(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for: {bad!r}")
    print("manifest self-check ok")
