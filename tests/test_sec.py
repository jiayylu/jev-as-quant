import pandas as pd

from jevquant.data.sec import exhibit99, first_tradable_open, html_to_text, lead_of_release, sp500_from_wikitext

WIKI = """{| class="wikitable sortable" id="constituents"
|-
![[Ticker symbol|Symbol]]
! Security !! Sector !! Sub-Industry !! HQ !! Date added !! CIK !! Founded
|-
|| {{NyseSymbol|MMM}}
|| [[3M]]
|| Industrials
|| Industrial Conglomerates
|| [[Saint Paul, Minnesota]]
|| 1957-03-04
|| 0000066740
|| 1902
|-
|| {{NasdaqSymbol|NVDA}}
|| [[Nvidia|NVIDIA]]
|| Information Technology
|| Semiconductors
|| [[Santa Clara, California]]
|| 2001-11-30
|| 1045810
|| 1993
|}"""


def test_sp500_parse():
    df = sp500_from_wikitext(WIKI)
    assert list(df["symbol"]) == ["MMM", "NVDA"]
    assert list(df["cik"]) == ["0000066740", "0001045810"]
    assert df.loc[1, "name"] == "NVIDIA" and df.loc[0, "added"] == "1957-03-04"


def test_first_tradable_open_respects_the_bell_and_weekends():
    days = pd.bdate_range("2026-07-27", "2026-08-07")  # Mon..Fri x2
    t = pd.Series(pd.to_datetime([
        "2026-07-28 07:05", "2026-07-28 09:31", "2026-07-28 16:30", "2026-07-31 18:00", "2026-08-01 10:00",
    ]).tz_localize("America/New_York"))
    main = first_tradable_open(t, days)
    assert [d.strftime("%a %d") for d in main] == ["Tue 28", "Wed 29", "Wed 29", "Mon 03", "Mon 03"]
    cons = first_tradable_open(t, days, conservative=True)
    assert [d.strftime("%a %d") for d in cons] == ["Wed 29", "Wed 29", "Wed 29", "Mon 03", "Mon 03"]


def test_lead_skips_boilerplate_and_contacts():
    html = """<p>Exhibit 99.1</p><p>Refer to: Jane Doe; jane@corp.com (Media)</p>
    <p>ACME Reports Record Second-Quarter Results; Raises Full-Year Guidance</p><p>Highlights</p>
    <p>SPRINGFIELD -- ACME today reported revenue of $5 billion, up 12% from a year ago, and raised its outlook.</p>"""
    head, lead = lead_of_release(html_to_text(html))
    assert head.startswith("ACME Reports Record") and "raised its outlook" in lead and "@" not in head


class FakeClient:
    def __init__(self, pages):
        self.pages = pages

    def get(self, url):
        return self.pages.get(url)


def test_exhibit99_uses_the_index_page_types():
    idx = ('<table><tr><td>1</td><td><a href="/Archives/edgar/data/1045810/000104581026000073/nvda-8k.htm">nvda-8k.htm</a></td>'
           '<td>8-K</td></tr><tr><td>2</td><td><a href="/Archives/edgar/data/1045810/000104581026000073/q2fy27pr.htm">q2fy27pr.htm</a></td>'
           '<td>EX-99.1</td></tr></table>')
    pr = "<p>NVIDIA Announces Financial Results for Second Quarter Fiscal 2027</p><p>Revenue of $96.2 billion, up 106% from a year ago, a record for the company this quarter.</p>"
    base = "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000073/"
    c = FakeClient({base + "0001045810-26-000073-index.htm": idx.encode(), base + "q2fy27pr.htm": pr.encode()})
    ex = exhibit99(c, "0001045810", "0001045810-26-000073")
    assert ex["exhibit"] == "EX-99.1" and ex["document"] == "q2fy27pr.htm"
    assert ex["headline"].startswith("NVIDIA Announces") and "106%" in ex["lead"]
    assert exhibit99(FakeClient({}), "0001045810", "0001045810-26-000073") is None


def test_lead_skips_filenames_addresses_company_lines_and_forms():
    html = ("<p>wbd2q25earningsrelease08.htm</p><p>Mayfield Village, Ohio 44143</p><p>The Travelers Companies, Inc.</p>"
            "<p>Kris Hinson, VP &amp; Treasurer</p><p>January 23, 2024; 4pm Pacific</p>"
            "<p>Warner Bros. Discovery Reports Second-Quarter 2025 Results</p>"
            "<p>NEW YORK -- Warner Bros. Discovery today reported second-quarter results with revenue up one percent.</p>")
    head, lead = lead_of_release(html_to_text(html))
    assert head == "Warner Bros. Discovery Reports Second-Quarter 2025 Results"
    assert lead.startswith("NEW YORK")
    form = "<p>" + " ".join(["Information or documents not available now must be given to ASX"] * 3) + "</p>"
    assert lead_of_release(html_to_text(form)) == ("", "")
