"""
Banner SVG ureticisi (dark.svg / light.svg).
Girdi (portrait_dither.py ve logo_points.py'nin ciktilari, kaynak-of-truth):
  dark_dots.npy, light_dots.npy, mask_clean.npy
  pts_fastapi.npy, pts_nodejs.npy, pts_archlinux.npy
Cikti: dark.svg, light.svg + banner_metrics.json (olcum raporu)
"""
import json
import numpy as np
from scipy.cluster.vq import kmeans2

RNG = np.random.default_rng(7)

# ---------------------------------------------------------------- geometri
BANNER_W, BANNER_H = 1180, 610
TITLEBAR_H = 36
PAD = 16
GRID_W, GRID_H = 300, 340  # portre/logo veri uzayi (birim: hucre)

FRAME_X = PAD
FRAME_W = round(BANNER_W * 0.38) - PAD  # sol ~38%
FRAME_Y = TITLEBAR_H + PAD + 22  # 22px: VISUAL.MAP etiketi icin
FRAME_H = BANNER_H - FRAME_Y - PAD

CELL = min(FRAME_W / GRID_W, FRAME_H / GRID_H)
DRAW_W, DRAW_H = GRID_W * CELL, GRID_H * CELL
OFFSET_X = FRAME_X + (FRAME_W - DRAW_W) / 2
OFFSET_Y = FRAME_Y + (FRAME_H - DRAW_H) / 2

PANEL_LEFT = FRAME_X + FRAME_W + PAD + 8
PANEL_RIGHT = BANNER_W - PAD
PANEL_W = PANEL_RIGHT - PANEL_LEFT
PANEL_TOP = TITLEBAR_H + PAD

ROW_FONT = 14
HEADER_FONT = 13
LIVE_FONT = 12
PILL_FONT = 14
ROW_SPACING = 23
CHAR_W = ROW_FONT * 0.6      # monospace varsayimi (satir metni)
DOT_ADVANCE = 6.0            # noktali liderin karakter araligi

# ---------------------------------------------------------- zamanlama (sn)
INTRO_DUR = 3.2
INTRO_FADE = 2.0
LOOP_PERIOD = 14.2
PORTRAIT_DWELL = 3.0
LOGO_DWELL = 2.0
TRANS = 1.3
N_INTRO_GROUPS = 60
N_BANDS = 94
N_TRAVELLERS = 900
DRIFT_FRACTION = 0.42
NOISE_SIGMA = 4.0

# loop icindeki mutlak zamanlar (0 = loop basi, INTRO_DUR sonra baslar)
T0 = 0.0
T1 = PORTRAIT_DWELL                      # 3.0  portrait dwell biter
T2 = T1 + TRANS                          # 4.3  logo1 dwell baslar
T3 = T2 + LOGO_DWELL                     # 6.3  logo1 dwell biter
T4 = T3 + TRANS                          # 7.6  logo2 dwell baslar
T5 = T4 + LOGO_DWELL                     # 9.6  logo2 dwell biter
T6 = T5 + TRANS                          # 10.9 logo3 dwell baslar
T7 = T6 + LOGO_DWELL                     # 12.9 logo3 dwell biter
T8 = T7 + TRANS                          # 14.2 = LOOP_PERIOD, portrait'e donus biter
assert abs(T8 - LOOP_PERIOD) < 1e-9

BAND_KEYTIMES = [T0 / LOOP_PERIOD, T1 / LOOP_PERIOD, T2 / LOOP_PERIOD,
                 T7 / LOOP_PERIOD, T8 / LOOP_PERIOD]
TRAVEL_KEYTIMES = [T0 / LOOP_PERIOD, T1 / LOOP_PERIOD, T2 / LOOP_PERIOD,
                    T3 / LOOP_PERIOD, T4 / LOOP_PERIOD, T5 / LOOP_PERIOD,
                    T6 / LOOP_PERIOD, T7 / LOOP_PERIOD, T8 / LOOP_PERIOD]

PALETTE = {
    "dark": {
        "bg": "#0A101F", "panel_border": "#1C2333",
        "portrait": "#A78BFA", "chrome": "#22D3EE", "accent": "#10B981",
        "live": "#EF4444", "text_value": "#E5E7EB", "text_label": "#8B93A7",
        "text_dim": "#586178", "pill_text": "#06251C",
    },
    "light": {
        "bg": "#F6F8FA", "panel_border": "#D7DEE8",
        "portrait": "#7C3AED", "chrome": "#0891B2", "accent": "#10B981",
        "live": "#DC2626", "text_value": "#1F2937", "text_label": "#5B6472",
        "text_dim": "#94A0B4", "pill_text": "#06251C",
    },
}

ROWS = [
    ("Subject", "Ahmet Talha Yaman"),
    ("Role", "DevOps Engineer"),
    ("Origin", "Ankara, Turkiye"),
    ("Education", "Bachelor's Degree - Computer Engineering"),
    ("Status", "Building + Learning + Shipping"),
    ("ToolChain", "VS Code, Git, Android Studio, Figma, PyCharm"),
    ("Core.Lang", "Python, TypeScript, Kotlin, Swift, C, C++, Node.js"),
    ("Core.Frontend", "Node.js, HTML5, CSS3"),
    ("Core.Backend", "Django, Nginx, Docker, Kubernetes, FastAPI"),
    ("Core.Database", "PostgreSQL, Supabase, MongoDB, SQLite"),
    ("Core.Infra", "AWS, Google Cloud, Firebase, Cloudflare"),
    ("Grid.Mail", "ahmettalhaymn@gmail.com"),
    ("Grid.Portfolio", "talhaymn7.github.io"),
    ("Grid.LinkedIn", "linkedin.com/in/talhaymn"),
    ("Grid.GitHub", "github.com/talhaymn7"),
    ("Grid.Facebook", "-"),
]


# ------------------------------------------------------------------ helpers
def dots_to_path_runs(coords):
    """coords: iterable (x,y) int. Ayni satirdaki bitisik hucreleri tek 'rect'
    alt-yoluna birlestirir (yatay run-length). shape-rendering=crispEdges ile
    kullanilir; font glif YOK, sadece dikdortgen path komutlari."""
    by_row = {}
    for x, y in coords:
        by_row.setdefault(int(y), []).append(int(x))
    parts = []
    for y in sorted(by_row):
        xs = sorted(set(by_row[y]))
        start = prev = xs[0]
        for x in xs[1:]:
            if x == prev + 1:
                prev = x
                continue
            n = prev - start + 1
            parts.append(f"M{start},{y}h{n}v1h{-n}z")
            start = prev = x
        n = prev - start + 1
        parts.append(f"M{start},{y}h{n}v1h{-n}z")
    return "".join(parts)


def evenness_metric(dot_xy, group_ids, n_groups):
    global_c = dot_xy.mean(axis=0)
    dot_spread = np.sqrt(np.mean(np.sum((dot_xy - global_c) ** 2, axis=1)))
    centroids = []
    for g in range(n_groups):
        pts = dot_xy[group_ids == g]
        if len(pts):
            centroids.append(pts.mean(axis=0))
    centroids = np.array(centroids)
    group_spread = np.sqrt(np.mean(np.sum((centroids - global_c) ** 2, axis=1)))
    return float(group_spread / dot_spread)


def straight_boundary_metric(dot_xy, band_ids):
    """Duz-dikis (grid artifact) olcumu: iki farkli banda ait komsu hucreler
    arasindaki sinirin, ayni x (veya y) degerinde kac ARDISIK satir (sutun)
    boyunca surdugunu olcer. Uzun ardisik run = duz cizgi = grid gorunumu.
    Literal grid partition icin medyan ~14px / en uzun ~37px cikarken,
    gurultulu KMeans kumeleme medyan ~2px / en uzun ~11px cikariyor -- yani
    olceklenebilir organik sinir. (Oran-tabanli ilk denemem seyrek dither
    verisinde anlamsiz cikti: rastgele atama bile yuksek skor veriyordu,
    cunku o metrik sinir kimligini degil sadece 'bu satirda sinir var mi'yi
    olcuyordu. Bu versiyon ayni x/y'nin ardisik run uzunlugunu dogrudan
    olctugu icin grid ile organik arasinda gercek ayrim yapiyor.)"""
    lookup = {(int(x), int(y)): int(b) for (x, y), b in zip(dot_xy, band_ids)}
    v_bound = {}  # x -> set(y) dikey sinirin gorüldugu satirlar
    h_bound = {}  # y -> set(x) yatay sinirin gorüldugu sutunlar
    for (x, y), b in lookup.items():
        r = lookup.get((x + 1, y))
        if r is not None and r != b:
            v_bound.setdefault(x, set()).add(y)
        d = lookup.get((x, y + 1))
        if d is not None and d != b:
            h_bound.setdefault(y, set()).add(x)

    def max_runs(bound_map):
        runs = []
        for _, vals in bound_map.items():
            vs = sorted(vals)
            run = best = 1
            for i in range(1, len(vs)):
                if vs[i] == vs[i - 1] + 1:
                    run += 1
                    best = max(best, run)
                else:
                    run = 1
            runs.append(best)
        return runs

    runs = max_runs(v_bound) + max_runs(h_bound)
    runs = np.array(runs) if runs else np.array([0])
    return {
        "median_seam_px": float(np.median(runs)),
        "p90_seam_px": float(np.percentile(runs, 90)),
        "max_seam_px": int(runs.max()),
    }


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def text_el(x, y, s, size, color, anchor="start", weight="normal",
            text_length=None, extra=""):
    tl = f' textLength="{text_length:.2f}" lengthAdjust="spacingAndGlyphs"' if text_length else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="ui-monospace,SFMono-Regular,'
            f'Menlo,Consolas,monospace" font-size="{size}" fill="{color}" '
            f'text-anchor="{anchor}" font-weight="{weight}"{tl}{extra}>{esc(s)}</text>')


def build_row(label, value, y, colors):
    label_w = len(label) * CHAR_W
    value_w = len(value) * CHAR_W
    label_x = PANEL_LEFT
    value_x = PANEL_RIGHT - value_w
    leader_start = label_x + label_w + 6
    leader_end = value_x - 6
    leader_w = max(0.0, leader_end - leader_start)
    n_dots = max(3, int(leader_w // DOT_ADVANCE))
    parts = [text_el(label_x, y, label, ROW_FONT, colors["text_label"], text_length=label_w)]
    parts.append(text_el(leader_start, y, "·" * n_dots, ROW_FONT, colors["text_dim"],
                          text_length=n_dots * DOT_ADVANCE))
    parts.append(text_el(value_x, y, value, ROW_FONT, colors["text_value"], text_length=value_w))
    return "".join(parts), n_dots


# ---------------------------------------------------------------- katmanlar
def build_intro_layer(dot_xy, colors):
    n = len(dot_xy)
    perm = RNG.permutation(n)
    group_of = np.empty(n, dtype=int)
    for i, idx in enumerate(perm):
        group_of[idx] = i % N_INTRO_GROUPS
    evenness = evenness_metric(dot_xy, group_of, N_INTRO_GROUPS)

    begins = RNG.uniform(0.0, INTRO_FADE - 0.35, N_INTRO_GROUPS)
    parts = [f'<g id="portrait-intro" opacity="1">']
    for g in range(N_INTRO_GROUPS):
        coords = dot_xy[group_of == g]
        if len(coords) == 0:
            continue
        d = dots_to_path_runs(coords)
        b = begins[g]
        parts.append(
            f'<path d="{d}" fill="{colors["portrait"]}" opacity="0" '
            f'shape-rendering="crispEdges">'
            f'<animate attributeName="opacity" from="0" to="1" '
            f'begin="{b:.3f}s" dur="0.35s" fill="freeze"/>'
            f'</path>'
        )
    parts.append(
        f'<animate attributeName="opacity" from="1" to="0" '
        f'begin="{INTRO_DUR:.2f}s" dur="0.3s" fill="freeze"/>'
    )
    parts.append("</g>")
    return "".join(parts), evenness


def build_loop_layer(dot_xy, logo1_centroid, colors):
    noisy = dot_xy + RNG.normal(0, NOISE_SIGMA, dot_xy.shape)
    _, labels = kmeans2(noisy, N_BANDS, minit="++", seed=11)
    straight = straight_boundary_metric(dot_xy, labels)

    kt = ";".join(f"{t:.5f}" for t in BAND_KEYTIMES)
    parts = [f'<g id="portrait-loop">']
    for b in range(N_BANDS):
        coords = dot_xy[labels == b]
        if len(coords) == 0:
            continue
        centroid = coords.mean(axis=0)
        dx, dy = (logo1_centroid - centroid) * DRIFT_FRACTION
        d = dots_to_path_runs(coords)
        opacity_vals = "1;1;0.08;0.08;1"
        translate_vals = f"0,0;0,0;{dx:.2f},{dy:.2f};{dx:.2f},{dy:.2f};0,0"
        parts.append(
            f'<g opacity="0" transform="translate(0,0)">'
            f'<path d="{d}" fill="{colors["portrait"]}" shape-rendering="crispEdges"/>'
            f'<animate attributeName="opacity" begin="{INTRO_DUR:.2f}s" '
            f'dur="{LOOP_PERIOD:.2f}s" repeatCount="indefinite" '
            f'keyTimes="{kt}" values="{opacity_vals}" calcMode="linear"/>'
            f'<animateTransform attributeName="transform" type="translate" '
            f'begin="{INTRO_DUR:.2f}s" dur="{LOOP_PERIOD:.2f}s" '
            f'repeatCount="indefinite" keyTimes="{kt}" values="{translate_vals}" '
            f'calcMode="linear"/>'
            f'</g>'
        )
    parts.append("</g>")
    return "".join(parts), straight


def build_travellers_layer(pts_a, pts_b, pts_c, colors):
    kt = ";".join(f"{t:.5f}" for t in TRAVEL_KEYTIMES)
    r = 1.1
    parts = [f'<g id="travellers">']
    for i in range(N_TRAVELLERS):
        ax, ay = pts_a[i]
        bx, by = pts_b[i]
        cx_, cy_ = pts_c[i]
        cxs = f"{ax:.2f};{ax:.2f};{ax:.2f};{ax:.2f};{bx:.2f};{bx:.2f};{cx_:.2f};{cx_:.2f};{ax:.2f}"
        cys = f"{ay:.2f};{ay:.2f};{ay:.2f};{ay:.2f};{by:.2f};{by:.2f};{cy_:.2f};{cy_:.2f};{ay:.2f}"
        ops = "0;0;1;1;1;1;1;1;0"
        parts.append(
            f'<circle cx="{ax:.2f}" cy="{ay:.2f}" r="{r}" fill="{colors["chrome"]}" opacity="0">'
            f'<animate attributeName="cx" begin="{INTRO_DUR:.2f}s" dur="{LOOP_PERIOD:.2f}s" '
            f'repeatCount="indefinite" keyTimes="{kt}" values="{cxs}" calcMode="linear"/>'
            f'<animate attributeName="cy" begin="{INTRO_DUR:.2f}s" dur="{LOOP_PERIOD:.2f}s" '
            f'repeatCount="indefinite" keyTimes="{kt}" values="{cys}" calcMode="linear"/>'
            f'<animate attributeName="opacity" begin="{INTRO_DUR:.2f}s" dur="{LOOP_PERIOD:.2f}s" '
            f'repeatCount="indefinite" keyTimes="{kt}" values="{ops}" calcMode="linear"/>'
            f'</circle>'
        )
    parts.append("</g>")
    return "".join(parts)


def build_chrome(colors, theme):
    parts = []
    parts.append(f'<rect x="0" y="0" width="{BANNER_W}" height="{BANNER_H}" rx="10" '
                  f'fill="{colors["bg"]}" stroke="{colors["panel_border"]}"/>')
    parts.append(f'<rect x="0" y="0" width="{BANNER_W}" height="{TITLEBAR_H}" rx="10" '
                  f'fill="{colors["panel_border"]}" opacity="0.35"/>')
    parts.append(f'<rect x="0" y="{TITLEBAR_H/2}" width="{BANNER_W}" height="{TITLEBAR_H/2}" '
                  f'fill="{colors["panel_border"]}" opacity="0.35"/>')
    for i, c in enumerate(["#EF4444", "#F59E0B", "#10B981"]):
        parts.append(f'<circle cx="{18+i*16}" cy="{TITLEBAR_H/2}" r="5" fill="{c}" opacity="0.85"/>')
    parts.append(text_el(BANNER_W / 2, TITLEBAR_H / 2 + 4, "profile.sh --live", 12,
                          colors["text_dim"], anchor="middle"))
    parts.append(f'<rect x="{FRAME_X}" y="{FRAME_Y}" width="{FRAME_W}" height="{FRAME_H}" '
                  f'rx="6" fill="none" stroke="{colors["panel_border"]}"/>')
    parts.append(text_el(FRAME_X + 4, FRAME_Y - 8, "VISUAL.MAP", 11, colors["chrome"]))
    return "".join(parts)


def build_info_panel(colors):
    parts = []
    header_y = PANEL_TOP + 8
    parts.append(text_el(PANEL_LEFT, header_y, "SYSTEM.INFO", HEADER_FONT, colors["chrome"]))

    live_x = PANEL_RIGHT - 62
    parts.append(f'<circle cx="{live_x}" cy="{header_y-4}" r="4" fill="{colors["live"]}">'
                 f'<animate attributeName="opacity" values="1;0.25;1" dur="1.4s" '
                 f'repeatCount="indefinite"/></circle>')
    parts.append(text_el(live_x + 10, header_y, "LIVE", LIVE_FONT, colors["live"], weight="bold"))

    pill_y = header_y + 22
    pill_text = "@talhaymn7"
    pill_w = len(pill_text) * PILL_FONT * 0.62 + 20
    pill_x = PANEL_RIGHT - pill_w
    parts.append(f'<rect x="{pill_x:.2f}" y="{pill_y-15}" width="{pill_w:.2f}" height="22" '
                 f'rx="11" fill="{colors["accent"]}"/>')
    parts.append(text_el(pill_x + pill_w / 2, pill_y, pill_text, PILL_FONT, colors["pill_text"],
                          anchor="middle", weight="bold"))

    row_y0 = pill_y + 34
    for i, (label, value) in enumerate(ROWS):
        y = row_y0 + i * ROW_SPACING
        row_svg, _ = build_row(label, value, y, colors)
        parts.append(row_svg)
        if label in ("ToolChain", "Core.Infra"):
            parts.append(f'<line x1="{PANEL_LEFT}" y1="{y+10}" x2="{PANEL_RIGHT}" y2="{y+10}" '
                         f'stroke="{colors["panel_border"]}" stroke-width="1"/>')
    return "".join(parts)


def build_svg(theme, dot_bool_grid, logo_pts, metrics):
    colors = PALETTE[theme]
    ys, xs = np.where(dot_bool_grid)
    dot_xy = np.stack([xs, ys], axis=1).astype(float)

    logo1_centroid = logo_pts["fastapi"].mean(axis=0)

    intro_svg, evenness = build_intro_layer(dot_xy, colors)
    loop_svg, straight = build_loop_layer(dot_xy, logo1_centroid, colors)
    travellers_svg = build_travellers_layer(
        logo_pts["fastapi"], logo_pts["nodejs"], logo_pts["archlinux"], colors)

    portrait_group = (
        f'<g transform="translate({OFFSET_X:.2f},{OFFSET_Y:.2f}) scale({CELL:.4f})">'
        f'{intro_svg}{loop_svg}{travellers_svg}</g>'
    )

    chrome_svg = build_chrome(colors, theme)
    panel_svg = build_info_panel(colors)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{BANNER_W}" height="{BANNER_H}" '
        f'viewBox="0 0 {BANNER_W} {BANNER_H}">'
        f'<title>Ahmet Talha Yaman - profile.sh --live</title>'
        f'{chrome_svg}'
        f'<clipPath id="visualmap-clip-{theme}"><rect x="{FRAME_X}" y="{FRAME_Y}" '
        f'width="{FRAME_W}" height="{FRAME_H}"/></clipPath>'
        f'<g clip-path="url(#visualmap-clip-{theme})">{portrait_group}</g>'
        f'{panel_svg}'
        f'</svg>'
    )

    metrics[theme] = {
        "dot_count": int(len(dot_xy)),
        "intro_evenness": evenness,
        "loop_straight_boundary": straight,
        "svg_bytes": len(svg.encode("utf-8")),
    }
    return svg


def main():
    dark_dots = np.load("dark_dots.npy")
    light_dots = np.load("light_dots.npy")
    logo_pts = {
        "fastapi": np.load("pts_fastapi.npy"),
        "nodejs": np.load("pts_nodejs.npy"),
        "archlinux": np.load("pts_archlinux.npy"),
    }

    metrics = {}
    dark_svg = build_svg("dark", dark_dots, logo_pts, metrics)
    light_svg = build_svg("light", light_dots, logo_pts, metrics)

    with open("dark.svg", "w", encoding="utf-8") as f:
        f.write(dark_svg)
    with open("light.svg", "w", encoding="utf-8") as f:
        f.write(light_svg)

    with open("banner_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    for theme, m in metrics.items():
        sb = m["loop_straight_boundary"]
        print(f"[{theme}] dots={m['dot_count']} "
              f"evenness={m['intro_evenness']:.4f} (hedef ~0.05) "
              f"seam medyan={sb['median_seam_px']:.1f}px p90={sb['p90_seam_px']:.1f}px "
              f"max={sb['max_seam_px']}px (grid referansi: medyan~14 p90~29 max~37) "
              f"size={m['svg_bytes']/1024:.1f}KB")


if __name__ == "__main__":
    main()
