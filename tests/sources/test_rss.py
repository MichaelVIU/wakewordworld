from __future__ import annotations

import httpx
import respx

from wakewordworld.licences import LicenceTier
from wakewordworld.sources.fetchers import fetch_source

from .conftest import make_spec

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"
  xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd"
  xmlns:podcast="https://podcastindex.org/namespace/1.0"
  xmlns:creativeCommons="http://backend.userland.com/creativeCommonsRssModule">
<channel>
  <title>Test Show</title>
  <itunes:author>Host Person</itunes:author>
  <podcast:license url="https://creativecommons.org/licenses/by/3.0/de/">CC-BY-3.0-DE</podcast:license>
  <item>
    <title>Episode 1</title>
    <link>https://example.org/ep1</link>
    <guid>ep1</guid>
    <pubDate>Mon, 01 Jan 2024 10:00:00 +0000</pubDate>
    <itunes:duration>1:30:00</itunes:duration>
    <enclosure url="https://cdn.example.org/ep1.mp3" type="audio/mpeg" length="1000"/>
    <podcast:transcript url="https://example.org/ep1.vtt" type="text/vtt"/>
  </item>
  <item>
    <title>Episode 2 (NC)</title>
    <link>https://example.org/ep2</link>
    <pubDate>Tue, 02 Jan 2024 10:00:00 +0000</pubDate>
    <itunes:duration>600</itunes:duration>
    <creativeCommons:license>https://creativecommons.org/licenses/by-nc-sa/4.0/</creativeCommons:license>
    <enclosure url="https://cdn.example.org/ep2.mp3" type="audio/mpeg"/>
  </item>
  <item>
    <title>Short teaser</title>
    <itunes:duration>30</itunes:duration>
    <enclosure url="https://cdn.example.org/teaser.mp3" type="audio/mpeg"/>
  </item>
  <item>
    <title>No enclosure</title>
  </item>
</channel>
</rss>
"""

ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Show</title>
  <entry>
    <title>A1</title>
    <id>a1</id>
    <updated>2024-03-01T00:00:00Z</updated>
    <link rel="alternate" href="https://example.org/a1"/>
    <link rel="enclosure" href="https://cdn.example.org/a1.ogg" type="audio/ogg" length="5"/>
  </entry>
</feed>
"""


@respx.mock
def test_rss_items_licences_and_filters(client: httpx.Client) -> None:
    respx.get("https://example.org/feed.xml").mock(return_value=httpx.Response(200, text=FEED))
    spec = make_spec(
        {"type": "rss", "url": "https://example.org/feed.xml"},
        filters={"min_duration_s": 60},
    )
    items = fetch_source(spec, client=client)
    assert [i.title for i in items] == ["Episode 1", "Episode 2 (NC)"]
    ep1, ep2 = items
    assert ep1.url == "https://cdn.example.org/ep1.mp3"
    assert ep1.duration_s == 5400
    assert ep1.page_url == "https://example.org/ep1"
    assert ep1.transcript_url == "https://example.org/ep1.vtt"
    assert ep1.author == "Host Person"
    assert ep1.language == "de"
    # channel-level podcast:license applies
    assert ep1.licence.spdx == "CC-BY-3.0-DE"
    assert ep1.licence.tier is LicenceTier.A
    assert ep1.licence.origin == "item"
    # item-level creativeCommons:license overrides channel
    assert ep2.licence.spdx == "CC-BY-NC-SA-4.0"
    assert ep2.licence.tier is LicenceTier.B
    assert ep1.item_id != ep2.item_id
    assert len(ep1.item_id) == 16


@respx.mock
def test_rss_max_items_and_no_item_licence(client: httpx.Client) -> None:
    respx.get("https://example.org/feed.xml").mock(return_value=httpx.Response(200, text=FEED))
    spec = make_spec(
        {"type": "rss", "url": "https://example.org/feed.xml", "read_item_licence": False},
        filters={"min_duration_s": 0, "max_items": 1},
    )
    items = fetch_source(spec, client=client)
    assert len(items) == 1
    assert items[0].licence.origin == "source"
    assert items[0].licence.spdx == "CC-BY-SA-4.0"


@respx.mock
def test_atom_feed(client: httpx.Client) -> None:
    respx.get("https://example.org/atom.xml").mock(return_value=httpx.Response(200, text=ATOM))
    spec = make_spec({"type": "rss", "url": "https://example.org/atom.xml"})
    items = fetch_source(spec, client=client)
    assert len(items) == 1
    assert items[0].url == "https://cdn.example.org/a1.ogg"
    assert items[0].page_url == "https://example.org/a1"
    assert items[0].media_type == "audio/ogg"


@respx.mock
def test_rss_http_error_raises_fetcherror(client: httpx.Client) -> None:
    import pytest

    from wakewordworld.sources.fetchers import FetchError

    respx.get("https://example.org/feed.xml").mock(return_value=httpx.Response(503))
    spec = make_spec({"type": "rss", "url": "https://example.org/feed.xml"})
    with pytest.raises(FetchError, match="HTTP 503"):
        fetch_source(spec, client=client)
