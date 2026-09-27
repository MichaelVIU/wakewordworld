# Openly licensed French & Spanish speech for a wake-word benchmark — verified survey (26 Sep 2026)

Method: every licence read from the primary source (RSS `<podcast:license>` / `<creativeCommons:license>` / `<copyright>`, show licence page, PeerTube `/api/v1/videos` `licence` field, media.ccc.de API, FOSDEM pentabarf XML, Wikimedia Commons `extmetadata`, archive.org advancedsearch + metadata API, HF dataset cards, openslr.org, ORTOLANG OAI-PMH `dc:rights`). Hours computed from duration fields where available, otherwise "est.".

Labels: **R** = redistributable (CC0 / CC BY / CC BY-SA / PD / permissive). **I** = internal only (NC / ND / research-only / no licence). **M** = mixed, filter per item. PDM = Public Domain Mark (uploader-declared, legally weak; get written CC0 confirmation before redistributing).

## 1. Podcasts

### 1a. French

| Show | Feed / URL | Licence (evidence) | Format | Volume | Transcripts | Label |
|---|---|---|---|---|---|---|
| **Libre à vous !** (April / Radio Cause Commune) | https://www.libreavous.org/rss | CC BY-SA 2.0+ (libreavous.org/apropos); cut music segments | Weekly 90-min live magazine, host + chroniqueurs + guests, first names constantly | ~660 h full episodes (2018–2026) | **Full transcripts** on librealire.org | **R — best FR vocative-rich source** |
| Radio Cause Commune, all talk shows | https://cause-commune.fm/feed/podcast/<slug> (57 shows) | station-wide CC BY-SA "sauf mention contraire" (mentions-legales) | volunteer talk shows | several hundred h | No | **R (check per show)** |
| L'Écho des Gnous (Radio Campus Lille) | https://www.echodesgnous.org/feed/podcast | `<copyright>` CC-by-sa | multi-host weekly, ~1 h | up to ~300 h | No | R |
| Projets Libres ! | https://podcast.projets-libres.org/@projetslibres/feed.xml | per item CC BY-SA 4.0+ | interviews | ~78 h | Yes | R |
| Rien De Grave Patron | https://rdgp.fr/@rdgp/feed.xml | `<copyright>` CC BY-SA 4.0 | 2 co-hosts | ~44 h | No | R |
| La Voix Est Libre (Picasoft) | https://podcast.picasoft.net/@la_voix_est_libre/feed.xml | CC BY 3.0 FR | student radio, several hosts | ~48 h | No | R |
| CKRL 89,1 (Québec) archive.org | https://archive.org/details/ckrl891 | 222/514 CC BY-SA 4.0, 151 PDM | news, 2-host shows | ~490 h total | No | M → R subset |
| Podcast des Accros (board games) | https://archive.org/details/PodcastDesAccros20130309 | 45/69 CC0 | multi-host chat | ~120 h CC0 | No | R subset |
| Other archive.org FR | EDAPodcast (CC0, 44 h), larouteduchemin (PDM), Entrez sans payer (BY-SA ~23 h) | per-item | mixed | ~150 h | No | M → R |
| Tabletop-RPG actual plays (Thomas Munier) | archive.org `creator:"Thomas Munier"` (359 items) | PDM (uploader-declared) | **multi-player conversation, constant first/character names** | ~350 h | No | R-ish (confirm CC0) |
| Tech Café | audiomeans feed | CC BY-NC 4.0 | host + co-hosts | ~634 h | No | I |
| Podcast Science | acast feed | CC BY-NC-SA 4.0 | live multi-host | ~805 h | partial | I |
| Les Cast Codeurs | https://lescastcodeurs.com/feed.atom | CC BY-NC-ND 4.0 | 5-host roundtable, first names constantly | ~310 h | No | I |
| Le Comptoir Sécu / Librement Linux / Méta de Choc / CPU | various | NC variants | roundtable / interviews | 40–100 items each | No | I |
| Arte Radio | arteradio.com | CC BY-NC-ND 2.0 FR at best | produced docs | large | No | I |
| No licence: Le Rendez-vous Tech (~977 h), NoLimitSecu, Les Technos, Trench Tech, Radio Panik, Radio Campus, Radio Debout, Radio France | | | | | | skip |

### 1b. Spanish

| Show | Feed / URL | Licence (evidence) | Variety / format | Volume | Transcripts | Label |
|---|---|---|---|---|---|---|
| **Podcast Linux** + Linux Express | https://podcastlinux.com/feed ; archive.org/details/podcast_linux | `<copyright>` CC-BY-SA 4.0 + per-episode | ES (Canarias); monologue + interviews | ~165 h + ~33 h | No | R |
| **KDE España podcast** | iVoox feed; archive.org `creator:"KDE España"` | own archive.org uploads CC BY-SA 4.0 | ES; **3–6 member round table, first names throughout** | ~94 h | No | R (confirm by e-mail) |
| KDE Express | https://kdeexpress.gitlab.io/feed | `<podcast:license>` CC-BY-SA-4.0 | monologue | ~21 h | **Yes** | R |
| Atareao con Linux | anchor.fm feed; atareao.es | site-wide CC BY 4.0 badge | monologue | ~313 h | No | R (confirm) |
| NOlegaltech Radio | feedburner | CC BY-SA 4.0 | 2 hosts + guests | ~15 h | No | R |
| **D-Strip-Ando** | feedburner; archive.org `creator:"D-Strip-Ando"` | 104 items CC BY-SA 4.0 on own archive.org uploads | **Mexico; 4 hosts, banter, first names constantly** | ~75 h BY-SA subset | 78 auto | M → R subset |
| Malditos Veganos | anchor.fm; archive.org | 69 PDM, 111 BY-NC-SA | 3 hosts + guests | ~60 h PDM | No | M → R-ish |
| Vida en Salud | archive.org `creator:"Diana Valeria"` | 81 items CC BY-SA 4.0 | interviews | ~45 h | No | M → R |
| Wikicafé (Wikimedia Chile) | iVoox feed | site-wide CC BY-SA 4.0 | Chile; interviews | ~22 h | No | R (confirm) |
| IA community radios (see §6) | Una Radio Muchas Voces (AR, ~1,160 h, CC BY-SA/BY), Espika FM (UY, ~377 h, CC BY-SA 4.0), Radio CCE (EC, PDM, thousands h), Radio Vox (AR, PDM), FM Moreno (AR) | per-item | LatAm talk/news/community radio | | No | R (BY-SA ones); PDM = R-ish |
| HistoCast | feedburner | per item CC BY-NC-ND 4.0 | **4–6 host tertulia, 2–5 h eps** | ~1,480 h | No | I — largest multi-host ES corpus |
| Entre Dev y Ops | rss | BY-NC-SA 3.0 | 3–4 hosts | ~150–200 h | No | I |
| Compilando Podcast | feed | CC BY-NC 4.0 | monologue + interviews | ~64 h | No | I |
| Radio Skylab / Doble Densidad / El Salto / GNU/Linux València | | NC variants | multi-host | 22–271 h | No | I |
| **Radio Ambulante / El hilo** | NPR feed | iHeartMedia: AI/TDM use strictly prohibited | | | | **Do not use** |
| No licence: Coffee Break (1,570 h), Salmorejo Geek, Eduardo Collado, Cienciaes, La Órbita de Endor, Todopoderosos, uGeek, Onda Hostil (no enclosures) | | | | | | skip |

## 2. Conference / talk archives

Hard negatives: media.ccc.de has 1 French event (12 min) and 1 Spanish (2.8 h). FOSDEM is 100 % English.

### 2a. French (per-video licence from PeerTube API)

| Source | URL | Licence | FR hours (R subset) | Label |
|---|---|---|---|---|
| **Capitole du Libre** | https://videos.capitoledulibre.org | CC BY on 413/414 h | 379 h (2012–2025), tables rondes | R |
| **Pas Sage en Seine** | https://video.passageenseine.fr | CC BY-SA 368 h | 368 h; FR auto captions | R |
| **JdLL Lyon + RMLL** | https://videos-libr.es | JdLL: 225 h CC0 + 146 h BY-SA + ~97 h NC-SA; RMLL 93 h BY-SA | R ≈ 510 h; tables rondes, débats | M → R |
| Ubuntu Party Paris | https://videos.ubuntu-paris.org | BY-SA 180 h, BY 9 h | 188 h | R |
| OSM France / SotM-FR | https://peertube.openstreetmap.fr | BY 275 h, BY-SA 91 h, PD 15 h | ~300 h | R |
| DINUM tube.numerique.gouv.fr | https://tube.numerique.gouv.fr | BY 361 h, BY-SA 26 h | ~370 h public-sector webinars | R |
| aperi.tube / Framatube | | BY-SA + BY | ~260 h / ~117 h | M → R |
| La Quadrature du Net | https://video.lqdn.fr | 384 of 437 h no licence | ~40 h R | I by default; ask LQDN |
| PyConFR / AFPy | https://indymotion.fr | all 173 videos unlicensed | 92 h | I (ask AFPy) |
| Devoxx France | YouTube | standard licence | est. 900+ h | I |
| BreizhCamp / Mix-IT | YouTube | 1 sample each = CC BY | est. 300 / 120 h | likely R, verify per video |
| Paris Web | | CC BY-NC-SA 3.0 FR | 250–300 h | I |
| Canal-U | https://www.canal-u.tv/oai | 0 CC BY in 1,200 sampled | huge | I |
| DebConf / MiniDebConf FR | meetings-archive.debian.net | MIT-style | 25–40 h | R |
| Wikimedia Commons: Videos in French | | 227 h, 100 % CC | | R |
| WikiConvention francophone 2016–2025 | Commons | 30 h BY-SA/BY | tables rondes | R |
| Spoken French Wikipedia | Commons | ~104 h read | | R |
| Sepia Search (fediverse, lang=fr, ≥10 min) | https://sepiasearch.org | CC BY 14,574 / BY-SA 4,777 / PD 1,487 videos ≈ 11,526 h (not all speech) | | M |

### 2b. Spanish

| Source | URL | Licence | ES hours (R subset) | Label |
|---|---|---|---|---|
| esLibre (2019–2025) | https://eslib.re ; tube.kockatoo.org, fediverse.tv | BY-SA 40, BY 4 of first 100 | est. 30–60 h | R subset |
| **Akademy-es / KDE España** | tube.kockatoo.org; YouTube; archive.org | BY-SA 37.7 h; archive.org CC0/PDM/BY-SA | ~100 h | R |
| **fediverse.tv** | https://fediverse.tv | BY-SA 635 h, BY 135 h, PD 91 h | R ≈ 670 h (charlas, entrevistas, mesas redondas) | M → R |
| tube.undernet.uy (Uruguay) | https://tube.undernet.uy | CC BY 186 h, BY-SA 71 h | ~250 h | M → R |
| FLISoL (LatAm) | PeerTube + archive.org | mostly BY-SA / CC BY | 80–100 h | R |
| Commit Conf / PyCon Argentina | YouTube | 1 sample each = CC BY | est. 250 / 15 h | likely R, verify |
| PyConES / T3chFest / PyCon Colombia / OpenExpo | YouTube | unverified / standard | | I until verified |
| UNED / UPV / UNAM / Tec de Monterrey | | no CC | large | I |
| Wikimedia Commons: Videos in Spanish | | 3,068 h: PD 2,838 h (**Venezuelan government broadcasts**, Asamblea Nacional), non-govt CC ≈ 230 h | | R (political) |
| **Jornadas de Wikimedia España** | Commons | 244 h BY-SA 4.0 continuous sessions with Q&A | | R |
| Spoken Spanish Wikipedia | Commons | ~119 h; **exclude "hablado por voz AI"** | | R (filter TTS) |
| Sepia Search (lang=es) | | 2,851 videos ≈ 2,158 h R | | M |

PeerTube ingestion: `GET /api/v1/videos?count=100&start=N&isLocal=true` → `licence.id` (1 BY, 2 BY-SA, 3 BY-ND, 4 BY-NC, 5 BY-NC-SA, 6 BY-NC-ND, 7 PD) and `language.id`; `/api/v1/videos/{uuid}` → `files[].fileDownloadUrl`; captions at `/api/v1/videos/{uuid}/captions`.

## 3. Speech corpora

### 3a. Large / general

| Corpus | Licence | FR | ES | Style | Names | Label |
|---|---|---|---|---|---|---|
| **Common Voice 27.0 Scripted** | CC0 | validated 1,108 h, 21,116 spk | validated 598 h, 26,960 spk | read | 3rd-person names in sentences | R |
| Common Voice Spontaneous Speech 5.0 | CC0 | 0.71 h validated | 0.08 h validated | spontaneous | no | R, tiny |
| **VoxPopuli** | CC0 | 211 h transcribed (534 spk) + ~4.5k h unlabelled | 166 h (305 spk) + ~4.4k h | EP plenary | surnames, formal address | R |
| **YODAS** | CC BY 3.0 | fr000 2,424 h manual + ~18.4k h auto | es000 3,738 h + ~20.5k h auto | in-the-wild YouTube | unannotated | R (attribute channels; ID obfuscation caveat) |
| Emilia-YODAS | CC BY 4.0 | 7.4k h | **no ES** | talk shows/podcasts, denoised | | R |
| Granary / YouTube-Commons / MOSEL | CC BY 4.0 | | | pseudo-labels / transcripts | | R |
| MLS (SLR94) / CML-TTS | CC BY 4.0 | ~1,077 h / 391 h | ~918 h / 449 h | read audiobooks | narration | R |
| FLEURS | CC BY 4.0 | ~12 h | ~12 h | read | | R |
| **MSWC** | CC BY 4.0 | 2,480,867 clips / 11,377 spk | 1,208,209 clips / 12,461 spk | 1-s words from CV | **FR: jean 2,761, pierre 1,684, louis 1,028, marie 985, paul 922, françois 789, jacques 756, michel 587, martin 495, philippe 493, nicolas 371, thomas 316, antoine 289, julien 229, anne 220, catherine 210, claire 170. ES: juan 603, maría 529, francisco 364, carlos 343, luis 294, antonio 266, pedro 253, manuel 225, miguel 220, david 203, ricardo 185, fernando 166, jorge 159, rosa 147, diego 139, ana 119, pablo 116, alberto 113, isabel 103, daniel 97, carmen 93, alejandro 91** | R |
| MediaSpeech (SLR108) | CC BY 4.0 | 10 h | 10 h | news | | R |
| African Accented French (SLR57) | Apache 2.0 | ~22 h, 232 spk | | read | | R |
| Google LatAm SLR61/71–75 | CC BY-SA 4.0 | | ~3–7 h each | read | | R |
| Heroico (SLR39) | Apache 2.0 | | 14 h | | | R |
| **CIEMPIESS family** (HF ciempiess/*) | CC BY-SA 4.0 | | ≈ 90 h spontaneous Mexican radio (LIGHT 18.4 h, BALANCE 18.3 h, FEM 13.9 h, TEST 8.1 h, TELEconCIENCIA 28.3 h) | spontaneous | hosts name guests | R |
| Europarl-ST v1.1 | CC BY-NC 4.0 | 32 h | 22 h | EP | | I |
| Multilingual TEDx (SLR100) / TEDx Spanish (SLR67) | CC BY-NC-ND 4.0 | ~50 h | ~189 h / 24 h | talks | | I |
| Audiocité (SLR139) | per-file CC | ~6,600 h read books | | read | | M |
| Snips SmartSpeaker FR | research only | 1,500 queries | | commands | | I |

### 3b. French spoken / interactional corpora (all ORTOLANG, CC BY-NC-SA unless noted) — I
ESLO 1+2 (>300 h, interviews, phone calls, **repas de famille**), CEFC/ORFEO (~200 h, >2,500 spk, meetings), CFPP2000 (~40 h, BY-NC-ND), TCOF (146 h, family/meal recordings), CLAPI (67 h, workplace/shops/meals; also TalkBank), CID (8 h dyads), PACO / **Diners familiaux parisiens** (16 family dinners), MPF, PFC, OFROM+, NCCFr (35 h casual, agreement), ESTER/ETAPE/REPERE (ELRA paid).

### 3c. Spanish spoken / colloquial corpora — I
**AMERESCO** (CC BY-NC-SA 4.0; 187 conversations, 16 cities, 707 speakers, audio + ELAN downloadable), Val.Es.Co 3.0 (~20 h, NC assumed), ESLORA (CC BY-NC 4.0; 60 h interviews + 24 h conversations), COSER (BY-NC-ND), PRESEEA (BY-NC-ND), C-ORAL-ROM (ELRA), Fisher/CALLHOME Spanish (LDC), **TalkBank CABank CallFriend/CallHome Spanish** (CC BY-NC-SA 3.0; friends/family phone calls, audio downloadable without LDC fee, very high vocative density), CORLEC (text only), CORPES XXI (web query only).

## 4. Parliament / government
- Assemblée nationale: multimedia "ne peuvent être reproduits sans accord préalable" → I. Sénat: conditional reuse with logo, no re-editing → I.
- Congreso de los Diputados: no statement covering Canal Parlamento video → I; HF `hsilvosa/congreso-debates` CC BY 4.0 text only.
- ParlamentParla (SLR59): CC BY 4.0, 320 h, but **Catalan**.
- LatAm congresses (MX, CL, AR, CO, EC, PE): no licence found → I.
- Venezuelan National Assembly / government on Commons: PD, 2,838 h → R (politically loaded).
- **VoxPopuli FR/ES**: CC0, the safe option.

## 5. Meeting / dinner-party corpora with vocatives
Nothing both natural multi-party and permissively licensed exists in either language. Internal-only: FR — Diners familiaux parisiens, TCOF repas, ESLO repas, CLAPI, CID, CEFC réunions, NCCFr, CallFriend Québécois; ES — AMERESCO, Val.Es.Co, ESLORA, TalkBank CallHome/CallFriend Spanish, C-ORAL-ROM. Closest redistributable substitutes: FR — Libre à vous! round-tables, Thomas Munier RPG actual plays (PDM), Cause Commune shows, PSES/JdLL débats; ES — KDE España round table, D-Strip-Ando BY-SA subset, Malditos Veganos PDM, Una Radio Muchas Voces / Espika FM, Jornadas Wikimedia España.

## 6. Internet Archive counts (`mediatype:audio AND language:(…)`)

| Licence | FR | ES |
|---|---|---|
| all audio items | 74,840 | 449,437 |
| CC0 | 754 | 3,718 |
| Public Domain Mark | 5,849 | 54,856 |
| CC BY | 606 | 4,351 |
| CC BY-SA | 639 | 5,695 |
| CC BY-NC-SA | 2,180 | 10,899 |
| CC BY-NC-ND | 13,432 | 44,181 |
| CC0+BY+BY-SA excl. music/78rpm/LibriVox/religion | **1,839 items** | **13,473 items** |

FR free subset dominated by mis-tagged music, CKRL Québec radio, audiobooks. ES dominated by Argentine/Uruguayan community radio: Una Radio Muchas Voces (Córdoba AR, 2,538 items, CC BY-SA 4.0/CC BY, est. ~1,160 h, debates, interviews, phone-ins), Espika FM 90.7 (UY, 244 items, CC BY-SA 4.0, est. ~377 h). Large PDM radio archives (Radio CCE Ecuador 22,418 items; Radio Vox 3,528; FM Moreno) need CC0 confirmation.

## (a) Ranked shortlist — redistributable

**French**
1. Libre à vous! + Radio Cause Commune talk shows — CC BY-SA; ~660 h + several hundred h; multi-voice, first names, full LAV transcripts.
2. VoxPopuli FR — CC0; 211 h transcribed + ~4.5k h unlabelled.
3. YODAS fr000 (+ Emilia-YODAS FR, Granary) — CC BY; 2,424 h manual-caption.
4. Libre-conference PeerTube set: Capitole du Libre (379 h), Pas Sage en Seine (368 h), JdLL/RMLL (~510 h), Ubuntu Party (188 h), OSM-FR (~300 h), tube.numerique.gouv.fr (~370 h) — ~2,100 h talks with Q&A.
5. Common Voice 27 FR (1,108 h, 21k spk) + MSWC FR first-name tokens.
6. Wikimedia Commons FR: Videos in French (227 h), WikiConvention (30 h), Spoken French Wikipedia (~104 h).
7. Smaller CC BY-SA podcasts: L'Écho des Gnous, Projets Libres!, RdGP, La Voix Est Libre.
8. archive.org clusters: CKRL, Podcast des Accros CC0, EDAPodcast CC0, Thomas Munier RPG actual plays (PDM, confirm).

**Spanish**
1. VoxPopuli ES — CC0; 166 h transcribed + ~4.4k h unlabelled.
2. YODAS es000 (+ Granary) — CC BY; 3,738 h manual-caption.
3. IA LatAm community radio: Una Radio Muchas Voces (~1,160 h) and Espika FM (~377 h), CC BY-SA/BY.
4. KDE España ecosystem: round-table podcast (~94 h), KDE Express (21 h, transcripts), Akademy-es / esLibre (~100–150 h).
5. Podcast Linux + Linux Express (~200 h, BY-SA) and Atareao (~313 h, confirm).
6. Wikimedia Commons ES: Jornadas WMES (244 h), WMES videos (37 h), Venezuelan PD block (2,838 h), Spoken Spanish Wikipedia (~110 h, exclude TTS).
7. Spanish PeerTube: fediverse.tv (~670 h), tube.undernet.uy (~250 h), FLISoL (~90 h).
8. CIEMPIESS (~90 h spontaneous Mexican radio), D-Strip-Ando BY-SA subset (~75 h), Common Voice 27 ES (598 h) + MSWC ES names, Google LatAm sets.

## (b) Internal-only worth having
- FR: Podcast Science (805 h), Tech Café (634 h), Les Cast Codeurs (310 h), LQDN (384 h unlicensed, ask), PyConFR (92 h, ask), Devoxx FR, Paris Web, Canal-U; Assemblée/Sénat; ORTOLANG family corpora (Diners familiaux, TCOF, ESLO, CLAPI, CID, CEFC, MPF, OFROM), NCCFr, CallFriend Québécois; Europarl-ST FR, mTEDx FR.
- ES: HistoCast (~1,480 h), Radio Skylab, Entre Dev y Ops, Compilando; TalkBank CallHome/CallFriend Spanish (best vocative source), AMERESCO, Val.Es.Co, ESLORA, COSER/PRESEEA, C-ORAL-ROM; mTEDx ES, TEDx Spanish, Europarl-ST ES; PyConES/T3chFest YouTube (unverified); PDM radio archives until CC0 confirmed. Radio Ambulante / El hilo off-limits.

## (c) Frank comparison with German
German has media.ccc.de (5,295 h CC BY with Q&A, licence per file), the Podlove/Metaebene culture with `<podcast:license>` tags on large multi-host shows, and Bundestag video. French reaches roughly a third of that by combining Libre à vous!/Cause Commune (~1,000+ h CC BY-SA radio talk, the one vocative-rich transcribed source) with libre-conference PeerTube (~2,000 h) and YODAS; both national parliaments forbid reuse, Canal-U is 0 % CC BY, and every French corpus designed around meals/family/meetings is CC BY-NC-SA. Spanish is thinner: best-verified redistributable conversational Spanish is a handful of libre-software podcasts (~370 h) plus Argentine/Uruguayan community radio (~1,500 h CC BY-SA, uneven metadata) and community PeerTube (~900 h); the large multi-host shows (HistoCast, Coffee Break) are NC-ND or unlicensed, no Spanish-speaking parliament publishes reusable video, all colloquial corpora are NC. For sheer negative hours both languages are fine (>5k h each under CC0/CC BY). For many-speaker, natural, first-name-vocative CC audio, **FR ≈ 25–35 % of DE, ES ≈ 10–20 %**. Expect to rely on internal-only NC corpora (TalkBank CallHome/CallFriend, AMERESCO, ORTOLANG family corpora, HistoCast, Podcast Science) for vocative measurements, and on relicensing requests (LQDN, AFPy, Thomas Munier, KDE España, Atareao, PDM radio uploaders) to grow the redistributable set.

Unverified / follow-up: Val.Es.Co licence, CORPES XXI, VoxLingua107, YouTube CC BY channels (BreizhCamp, Mix-IT, Commit Conf, PyConAr; re-run yt-dlp with cookies), Radio Skylab, UNAM/Tec de Monterrey/U. de Chile, Cause Commune per-show exceptions.
