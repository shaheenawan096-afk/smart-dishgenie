"""
services/logo.py
----------------
The Smart DishGenie System logo as inline SVG (sharp at every size, no image file needed).
  LOGO_HEADER : small logo + name in one line   (top of the website)
  LOGO_STACK  : big logo with the name below     (loading page)
  LOGO_DARK   : logo + name for dark backgrounds (footer)
Every string is ONE line, so Streamlit's markdown never turns it into a code block.
"""


def _one_line(s: str) -> str:
    return " ".join(part.strip() for part in s.strip().splitlines())


# the drawing (chef hat, steam, leaves, spoon, bowl, swoosh, sparkles) in a 240 x 200 box
_HAT = ('<g fill="{edge}"><circle cx="100" cy="50" r="24"/><circle cx="126" cy="38" r="28"/>'
        '<circle cx="152" cy="52" r="23"/><rect x="88" y="54" width="76" height="34" rx="8"/></g>'
        '<g fill="#FFFFFF"><circle cx="100" cy="50" r="19.5"/><circle cx="126" cy="38" r="23.5"/>'
        '<circle cx="152" cy="52" r="18.5"/><rect x="92.5" y="54" width="67" height="29.5" rx="5"/></g>')


def _mark(uid: str, dark: bool = False) -> str:
    edge = "#0B241B" if dark else "#12372A"
    steam1, steam2 = ("#6CC08A", "#A8E0B4") if dark else ("#2E8B62", "#6CC08A")
    spoon = "#6CC08A" if dark else "#1D6B4B"
    bowl = "#2E8B62" if dark else f"url(#{uid}b)"
    soup = "#EDA81B" if dark else f"url(#{uid}s)"
    swoosh = "#8FD19E" if dark else f"url(#{uid}w)"
    defs = "" if dark else (
        f'<defs><linearGradient id="{uid}b" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#2E8B62"/>'
        f'<stop offset="1" stop-color="#12372A"/></linearGradient>'
        f'<linearGradient id="{uid}s" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#F2994A"/>'
        f'<stop offset="1" stop-color="#EDA81B"/></linearGradient>'
        f'<linearGradient id="{uid}w" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#2E8B62"/>'
        f'<stop offset="1" stop-color="#8FD19E"/></linearGradient></defs>')
    return (defs + _HAT.format(edge=edge) +
            f'<path d="M116 116c-16-14 12-24-4-42" stroke="{steam1}" stroke-width="8" fill="none" stroke-linecap="round"/>'
            f'<path d="M134 114c-12-12 12-20 0-36" stroke="{steam2}" stroke-width="5.5" fill="none" stroke-linecap="round"/>'
            '<path d="M60 106c-20-6-27-26-14-42 15 6 22 24 14 42z" fill="#43A35A"/>'
            '<path d="M70 104c-6-14 0-28 15-32 5 14-2 28-15 32z" fill="#6CC08A"/>'
            f'<path d="M176 112l30-32" stroke="{spoon}" stroke-width="7" stroke-linecap="round"/>'
            f'<ellipse cx="210" cy="75" rx="8" ry="12" transform="rotate(42 210 75)" fill="{spoon}"/>'
            f'<path d="M58 110h132c-2 38-30 60-66 60s-64-22-66-60z" fill="{bowl}"/>'
            f'<ellipse cx="124" cy="110" rx="66" ry="9.5" fill="{soup}" stroke="{edge}" stroke-width="5"/>'
            f'<path d="M50 160c50 26 140 26 176-34" stroke="{swoosh}" stroke-width="6.5" fill="none" stroke-linecap="round"/>'
            '<path d="M196 26l2.6 6.4 6.4 2.6-6.4 2.6-2.6 6.4-2.6-6.4-6.4-2.6 6.4-2.6z" fill="#EDA81B"/>'
            '<path d="M216 50l1.6 3.9 3.9 1.6-3.9 1.6-1.6 3.9-1.6-3.9-3.9-1.6 3.9-1.6z" fill="#EDA81B"/>')


LOGO_HEADER = _one_line(f"""
<svg class="dg-logo-header" viewBox="0 0 300 66" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Smart DishGenie System">
<g transform="translate(-6 -2) scale(0.36)">{_mark("dgh")}</g>
<text x="88" y="34" font-family="'Bricolage Grotesque', Figtree, sans-serif" font-weight="800" font-size="24" letter-spacing="-0.4" fill="#12372A">Smart Dish<tspan fill="#1D8A55">Genie</tspan></text>
<text x="89" y="53" font-family="Figtree, sans-serif" font-weight="800" font-size="9.5" letter-spacing="4.2" fill="#B7791F">SYSTEM</text>
</svg>""")

LOGO_STACK = _one_line(f"""
<svg class="dg-logo-stack" viewBox="0 0 240 250" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Smart DishGenie System">
{_mark("dgs")}
<text x="120" y="212" text-anchor="middle" font-family="'Bricolage Grotesque', Figtree, sans-serif" font-weight="800" font-size="27" letter-spacing="-0.5" fill="#12372A">Smart Dish<tspan fill="#1D8A55">Genie</tspan></text>
<path d="M50 232h34M156 232h34" stroke="#EDA81B" stroke-width="2.4" stroke-linecap="round"/>
<text x="120" y="236" text-anchor="middle" font-family="Figtree, sans-serif" font-weight="800" font-size="11" letter-spacing="5" fill="#1D6B4B">SYSTEM</text>
</svg>""")

LOGO_DARK = _one_line(f"""
<svg class="dg-logo-dark" viewBox="0 0 470 168" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Smart DishGenie System">
<g transform="translate(-34 -6) scale(0.86)">{_mark("dgd", dark=True)}</g>
<text x="196" y="84" font-family="'Bricolage Grotesque', Figtree, sans-serif" font-weight="800" font-size="31" letter-spacing="-0.5" fill="#FFFFFF">Smart Dish<tspan fill="#8FD19E">Genie</tspan></text>
<path d="M198 104h26" stroke="#EDA81B" stroke-width="2.4" stroke-linecap="round"/>
<text x="232" y="108" font-family="Figtree, sans-serif" font-weight="800" font-size="12" letter-spacing="5" fill="#EDA81B">SYSTEM</text>
</svg>""")
