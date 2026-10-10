"""Page styling and small HTML components.

Custom HTML inherits the active Streamlit theme through currentColor and
color-mix(), so the same markup reads correctly on the light and dark theme.
"""

from __future__ import annotations

from html import escape

from .config import CONDITIONS, SEDIMENT, WATER

CSS = f"""
<style>
:root {{
  --ld-water: {WATER};
  --ld-sediment: {SEDIMENT};
}}
[data-testid="stMainBlockContainer"] {{
  max-width: 1240px;
  padding-top: 2.4rem;
  padding-bottom: 4rem;
}}
/* Hydrographic name: lake names are set in italic serif, as on survey maps */
.ld-name {{
  font-family: 'Source Serif 4', 'Source Serif Pro', Georgia, serif;
  font-style: italic;
  font-weight: 500;
  font-size: clamp(2.3rem, 4.2vw, 3.4rem);
  line-height: 1.04;
  letter-spacing: -0.012em;
  color: color-mix(in oklab, var(--ld-water) 72%, currentColor);
  margin: 0 0 0.45rem 0;
}}
.ld-lede {{
  font-size: 1.06rem;
  line-height: 1.5;
  color: color-mix(in srgb, currentColor 74%, transparent);
  max-width: 64ch;
  margin: 0 0 0.2rem 0;
}}
.ld-when {{
  font-size: 1rem;
  color: color-mix(in srgb, currentColor 70%, transparent);
  margin: 0.2rem 0 0.1rem 0;
}}
.ld-figure {{
  font-size: 3.3rem;
  font-weight: 600;
  line-height: 1.02;
  letter-spacing: -0.015em;
  margin: 0;
}}
.ld-figure span {{
  font-size: 1.35rem;
  font-weight: 500;
  margin-left: 0.3rem;
  color: color-mix(in srgb, currentColor 72%, transparent);
}}
.ld-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 1rem;
  font-weight: 600;
  margin: 0.55rem 0 0.95rem 0;
}}
.ld-pill i {{
  width: 0.95rem;
  height: 0.95rem;
  border-radius: 3px;
  display: inline-block;
  background: var(--c);
  box-shadow: inset 0 0 0 1px color-mix(in srgb, currentColor 18%, transparent);
}}
.ld-headline {{
  font-size: 1.1rem;
  line-height: 1.58;
  max-width: 64ch;
  margin: 0 0 1.25rem 0;
}}
.ld-readings {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
  border-top: 1px solid color-mix(in srgb, currentColor 16%, transparent);
  border-bottom: 1px solid color-mix(in srgb, currentColor 10%, transparent);
  margin: 0 0 0.6rem 0;
  overflow: hidden;
}}
.ld-reading {{
  padding: 0.85rem 1rem 0.95rem 1rem;
  margin-left: -1px;
  border-left: 1px solid color-mix(in srgb, currentColor 12%, transparent);
}}
.ld-reading .l {{
  font-size: 0.88rem;
  line-height: 1.3;
  color: color-mix(in srgb, currentColor 68%, transparent);
}}
.ld-reading .v {{
  font-size: 1.6rem;
  font-weight: 600;
  line-height: 1.15;
  margin: 0.35rem 0 0.25rem 0;
  white-space: nowrap;
}}
.ld-reading .s {{
  font-size: 0.84rem;
  line-height: 1.35;
  color: color-mix(in srgb, currentColor 64%, transparent);
}}
.ld-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 1.1rem;
  font-size: 0.88rem;
  margin: 0.15rem 0 0.4rem 0;
  color: color-mix(in srgb, currentColor 80%, transparent);
}}
.ld-legend span {{ display: inline-flex; align-items: center; gap: 0.4rem; }}
.ld-legend i {{
  width: 0.8rem; height: 0.8rem; border-radius: 2px; display: inline-block; background: var(--c);
}}
.ld-legend i.line {{ height: 2px; width: 1.1rem; border-radius: 1px; }}
.ld-legend i.band {{ opacity: 0.55; }}
.ld-note {{
  font-size: 0.88rem;
  line-height: 1.5;
  color: color-mix(in srgb, currentColor 70%, transparent);
  max-width: 78ch;
}}
.ld-demo {{
  border-left: 3px solid var(--ld-sediment);
  padding: 0.6rem 0.95rem;
  margin: 0 0 1.4rem 0;
  background: color-mix(in srgb, var(--ld-sediment) 10%, transparent);
  font-size: 0.95rem;
  line-height: 1.45;
  max-width: 78ch;
}}
.ld-wordmark {{
  display: flex;
  align-items: center;
  gap: 0.55rem;
  font-family: 'Barlow Semi Condensed', Barlow, sans-serif;
  font-weight: 600;
  font-size: 1.32rem;
  letter-spacing: 0.01em;
  margin: 0 0 0.15rem 0;
}}
.ld-wordmark svg {{ flex: none; }}
.ld-tagline {{
  font-size: 0.86rem;
  color: color-mix(in srgb, currentColor 66%, transparent);
  margin: 0 0 1rem 0;
}}
.ld-steps {{ margin: 0.2rem 0 0 0; padding-left: 1.2rem; max-width: 70ch; }}
.ld-steps li {{ margin: 0.25rem 0; line-height: 1.45; }}
.ld-facts {{ margin: 0.1rem 0 0 0; padding-left: 1.1rem; max-width: 74ch; }}
.ld-facts li {{ margin: 0.3rem 0; line-height: 1.5; }}
button[role="tab"] p {{ font-size: 0.98rem; }}
[data-testid="stSidebar"] {{ min-width: 340px; max-width: 340px; }}
.ld-gauge-wrap {{ margin-top: 0.35rem; }}
@media (max-width: 640px) {{
  .ld-figure {{ font-size: 2.6rem; }}
  [data-testid="stSidebar"] {{ min-width: 300px; max-width: 300px; }}
}}
</style>
"""

WORDMARK_ICON = f"""<svg width="22" height="26" viewBox="0 0 22 26" aria-hidden="true">
<rect x="6.5" y="1" width="9" height="24" rx="2.5" fill="none" stroke="currentColor"
 stroke-opacity="0.55" stroke-width="1.2"/>
<rect x="7.6" y="13" width="6.8" height="10.9" rx="1.6" fill="{WATER}" fill-opacity="0.55"/>
<line x1="3" x2="19" y1="13" y2="13" stroke="{WATER}" stroke-width="1.8" stroke-linecap="round"/>
<line x1="3.5" x2="6.5" y1="6" y2="6" stroke="currentColor" stroke-opacity="0.5"/>
<line x1="3.5" x2="6.5" y1="19" y2="19" stroke="currentColor" stroke-opacity="0.5"/>
</svg>"""


def wordmark() -> str:
    return (f'<div class="ld-wordmark">{WORDMARK_ICON}<span>LakeDelta</span></div>'
            '<p class="ld-tagline">How full is the lake, compared with its own history?</p>')


def name_block(name: str, lede: str) -> str:
    return f'<h1 class="ld-name">{escape(name)}</h1><p class="ld-lede">{escape(lede)}</p>'


def status_block(when: str, value_km2: str, condition_label: str | None, color: str | None,
                 headline: str) -> str:
    pill = ""
    if condition_label:
        pill = (f'<div class="ld-pill"><i style="--c:{color}"></i>{escape(condition_label)}</div>')
    num, _, unit = value_km2.partition(" ")
    return (f'<p class="ld-when">{escape(when)}</p>'
            f'<p class="ld-figure">{escape(num)}<span>{escape(unit)}</span></p>'
            f'{pill}<p class="ld-headline">{escape(headline)}</p>')


def readings(items: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f'<div class="ld-reading"><div class="l">{escape(l)}</div>'
        f'<div class="v">{escape(v)}</div><div class="s">{escape(s)}</div></div>'
        for l, v, s in items)
    return f'<div class="ld-readings">{cells}</div>'


def condition_legend() -> str:
    spans = "".join(f'<span><i style="--c:{c.color}"></i>{escape(c.label)}</span>' for c in CONDITIONS)
    return f'<div class="ld-legend">{spans}</div>'


def legend(items: list[tuple[str, str, str]]) -> str:
    """items: (kind, color, label) with kind in {'box', 'line', 'band'}."""
    spans = "".join(
        f'<span><i class="{k}" style="--c:{c}"></i>{escape(label)}</span>' for k, c, label in items)
    return f'<div class="ld-legend">{spans}</div>'


def note(text: str) -> str:
    return f'<p class="ld-note">{escape(text)}</p>'


def facts(items: list[str]) -> str:
    return '<ul class="ld-facts">' + "".join(f"<li>{escape(i)}</li>" for i in items) + "</ul>"


def demo_banner() -> str:
    return ('<div class="ld-demo"><b>Demo data.</b> Earth Engine is not connected, so this page '
            'shows an invented reservoir to demonstrate the analysis. Add an Earth Engine '
            'service account to the app secrets to analyse real lakes.</div>')
