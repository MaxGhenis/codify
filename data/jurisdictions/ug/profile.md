# Uganda

## How the law is published

Parliament passes Acts in English. Each Act as passed is printed in the Acts
Supplement to the Uganda Gazette by the Uganda Printing and Publishing
Corporation, Entebbe, and Parliament's document repository serves the same
prints as scanned PDFs without a text layer. The Laws of Uganda revised edition
consolidates Acts under chapter numbers, so the same statute is cited as
"Act 8 of 1985" as passed and as "Cap. 230" once consolidated; revised-edition
volumes reach the web as scans too. Statutory instruments appear in a separate
Statutory Instruments Supplement. ulii.org carries unofficial consolidations
built by Laws.Africa.

The tier is 4: the government publishes PDFs, most of them images.

## The shape of an Act

Westminster drafting. Parts carry a roman numeral and a heading in capitals
("PART IV—CONTRIBUTIONS"). Sections are the basic unit and print as the number,
a full stop and the heading on one line, with no keyword: "11. Employee's share
of standard contribution". Subsections are "(1)", paragraphs "(a)", subparagraphs
"(i)". Schedules follow the body; a schedule that carries regulations divides
into Divisions. An amending Act names the principal Act in section 1, calls it
"the principal Act" thereafter, and amends by inserting, substituting or
repealing quoted text introduced by "the following—". The enacting formula is
"BE IT ENACTED by Parliament as follows:", with a colon or an em dash.

Because the section keyword is absent, the section level declares
`marker_form: "arabic_period"`; without it the scan finds Parts and nothing
under them.

## Identifiers

An Act's FRBR number is its number as passed ("Act 8 of 2008"), read from the
cover or the running head. A revised-edition chapter number ("Cap. 230")
identifies the consolidation, not the work, so minting a work URI from a
revised-edition text needs the year and number of the original enactment; the
revised edition prints them in the History line at the end of the Act. For the
same reason the Act hierarchy has no chapter level: a revised-edition cover line
"CHAPTER 230" names the Act, and reading it as a container would wrap the whole
Act in a phantom chapter.

## What the scan reads and what it gets wrong

Four Acts were ingested end to end while this profile was written: the National
Social Security Fund Act (Cap. 230, revised edition, 38 scanned pages), the
Local Governments (Amendment) (No. 2) Act, 2008 (12 scanned pages), the Income
Tax (Amendment) Act, 2012 (4 scanned pages) and the Excise Duty (Amendment) Act,
2024 (text layer). The anchor scan reads 53 of the NSSF Act's body sections, 12
schedule items and the three closing schedules, and 18 sections in the 2008 Act; the 2008
Act's scaffold round-trips through Bluebell with no findings.

Schedules use the scanner's own schedule handling rather than a declared
container, so a schedule named in running prose ("Schedule 1 to this Act")
stays text and the closing schedules anchor at the top level.

Three things remain open. A body section is dropped when the sentence before it
names a schedule and that schedule holds an item with the section's number: in
the NSSF Act, section 8 ends "specified in Schedule 2 to this Act", Schedule 2
has an item 9, and "9. Cancellation of registration" is not anchored, while
coverage still reads 54 of 54 because the schedule's item supplies the number.
A schedule that itself divides into Parts or Divisions,
as a schedule of regulations does, is not anchored, and its Divisions nest under
the body's last Part. And no Statutory Instruments Supplement has been read
through the pipeline, so the `si` class is a hypothesis carried over from the
Act hierarchy.

## Running heads

Revised-edition pages carry a running head with the volume page number and the
chapter ("8216 Cap. 230.] National Social Security Fund Act" on the left-hand
page, the mirror on the right); Acts as passed carry "Act 8 ... Act 2008". The
header patterns strip these when they land on their own line, and the inline
pattern catches the case where a page break fuses the head into a sentence.
