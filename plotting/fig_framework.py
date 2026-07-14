"""
ContextEcho framework figure — icon-rich version matching the reference layout.

5-layer structure:
  (1) Top: 4 design-principle pillars + tagline (with icons)
  (2) Left sidebar: 23 targets + 3 sessions (with icons)
  (3) Stimuli row: 4 cards (probes / stressors / SWE-Bench / A-anchor) each with a big icon
  (4) Primitive band: snapshot-then-probe formula
  (5) Outcomes strip: Q1-Q5 with metric icons

Output: ../paper/figures/fig_framework.{pdf,png}
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------- color palette ----------
COL_PROBES = "#16a34a"  # green
COL_STRESSORS = "#7c3aed"  # purple
COL_SWEBENCH = "#ea580c"  # orange
COL_ANCHOR = "#dc2626"  # red
COL_PRIMITIVE_BG = "#fef3c7"  # light yellow
COL_PRIMITIVE_EC = "#f59e0b"  # amber
COL_FRAME_BG = "#eff6ff"  # very light blue
COL_FRAME_EC = "#bfdbfe"  # light blue border
COL_SIDEBAR_BG = "#f9fafb"  # very light grey
COL_SIDEBAR_EC = "#d1d5db"
COL_HEADER = "#1e3a8a"  # dark blue
COL_TEXT = "#111827"
COL_MUTED = "#4b5563"


def rounded_box(ax, x, y, w, h, fc, ec="none", radius=0.012, lw=0.8, zorder=1):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(box)


# ---------- Icon primitives drawn with matplotlib shapes ----------
def icon_target(ax, cx, cy, r, color):
    for i, frac in enumerate([1.0, 0.66, 0.33]):
        ax.add_patch(plt.Circle((cx, cy), r * frac, fill=(i == 2), facecolor=color, edgecolor=color, lw=1.6, zorder=10))


def icon_chip(ax, cx, cy, r, color):
    # rounded square = "chip/protocol"
    rounded_box(ax, cx - r, cy - r, 2 * r, 2 * r, fc="white", ec=color, lw=1.6, radius=0.005, zorder=10)
    ax.text(cx, cy, "API", ha="center", va="center", fontsize=8, fontweight="bold", color=color, zorder=11)


def icon_puzzle(ax, cx, cy, r, color):
    rounded_box(ax, cx - r, cy - r * 0.7, 2 * r, 1.4 * r, fc=color, ec="none", radius=0.004, zorder=10)
    ax.add_patch(plt.Circle((cx + r * 0.55, cy + r * 0.3), r * 0.28, fc="white", ec=color, lw=1.0, zorder=11))
    ax.add_patch(plt.Circle((cx - r * 0.55, cy + r * 0.3), r * 0.28, fc="white", ec=color, lw=1.0, zorder=11))


def icon_lock(ax, cx, cy, r, color):
    # body
    rounded_box(ax, cx - r * 0.7, cy - r, 1.4 * r, 1.4 * r, fc=color, ec="none", radius=0.004, zorder=10)
    # shackle (semicircle)
    arc = plt.matplotlib.patches.Arc(
        (cx, cy + r * 0.3),
        width=r * 1.0,
        height=r * 1.2,
        angle=0,
        theta1=0,
        theta2=180,
        edgecolor=color,
        lw=2.0,
        zorder=10,
    )
    ax.add_patch(arc)


def icon_arrow(ax, cx, cy, r, color):
    arr = FancyArrowPatch(
        (cx - r, cy - r * 0.5),
        (cx + r, cy + r * 0.5),
        arrowstyle="-|>",
        mutation_scale=18,
        color=color,
        lw=2.2,
        zorder=10,
    )
    ax.add_patch(arr)


def icon_book(ax, cx, cy, r, color):
    # stylized "book": 3 stacked rectangles representing pages
    for i, dy in enumerate([-0.4, 0, 0.4]):
        rounded_box(
            ax,
            cx - r * 0.8,
            cy + dy * r - 0.04,
            1.6 * r,
            0.18,
            fc=color if i == 1 else "white",
            ec=color,
            lw=1.2,
            radius=0.003,
            zorder=10,
        )


def icon_balance(ax, cx, cy, r, color):
    # scale balance (simplified): horizontal bar + center post + 2 pans
    ax.plot([cx - r, cx + r], [cy + r * 0.4, cy + r * 0.4], color=color, lw=2.2, zorder=10)
    ax.plot([cx, cx], [cy + r * 0.4, cy - r * 0.8], color=color, lw=2.2, zorder=10)
    # pans
    for sx in [cx - r * 0.8, cx + r * 0.8]:
        ax.plot([sx, sx], [cy + r * 0.4, cy + r * 0.05], color=color, lw=1.2, zorder=10)
        arc = plt.matplotlib.patches.Arc(
            (sx, cy - r * 0.05),
            width=r * 0.9,
            height=r * 0.4,
            theta1=180,
            theta2=360,
            edgecolor=color,
            lw=1.8,
            zorder=10,
        )
        ax.add_patch(arc)


def icon_clipboard(ax, cx, cy, r, color):
    # clipboard outline + clip on top
    rounded_box(ax, cx - r * 0.7, cy - r, 1.4 * r, 1.8 * r, fc="white", ec=color, lw=1.6, radius=0.004, zorder=10)
    rounded_box(ax, cx - r * 0.3, cy + r * 0.65, 0.6 * r, 0.25 * r, fc=color, ec=color, lw=1.0, radius=0.002, zorder=11)
    # lines
    for dy in [0.3, 0.1, -0.1, -0.3]:
        ax.plot([cx - r * 0.4, cx + r * 0.4], [cy + dy * r, cy + dy * r], color=color, lw=1.2, zorder=11)


def icon_magnifier(ax, cx, cy, r, color):
    ax.add_patch(plt.Circle((cx - r * 0.2, cy + r * 0.2), r * 0.55, fc="white", ec=color, lw=2.0, zorder=10))
    ax.plot(
        [cx + r * 0.25, cx + r * 0.7],
        [cy - r * 0.25, cy - r * 0.7],
        color=color,
        lw=2.4,
        zorder=10,
    )


def icon_wrench(ax, cx, cy, r, color):
    # plus sign = "fix / wrench"
    ax.add_patch(plt.Circle((cx, cy), r * 0.85, fc=color, ec="none", zorder=10))
    ax.plot([cx - r * 0.45, cx + r * 0.45], [cy, cy], color="white", lw=2.6, zorder=11)
    ax.plot([cx, cx], [cy - r * 0.45, cy + r * 0.45], color="white", lw=2.6, zorder=11)


def icon_chart(ax, cx, cy, r, color):
    # 3 ascending bars
    for i, h in enumerate([0.4, 0.7, 1.0]):
        ax.add_patch(
            plt.Rectangle(
                (cx - r * 0.7 + i * r * 0.5, cy - r * 0.5),
                r * 0.35,
                r * h,
                fc=color,
                ec=color,
                zorder=10,
            )
        )


def icon_shield(ax, cx, cy, r, color):
    pts = [
        (cx, cy + r),
        (cx + r * 0.85, cy + r * 0.5),
        (cx + r * 0.7, cy - r * 0.7),
        (cx, cy - r),
        (cx - r * 0.7, cy - r * 0.7),
        (cx - r * 0.85, cy + r * 0.5),
    ]
    poly = plt.Polygon(pts, closed=True, fc=color, ec="none", zorder=10)
    ax.add_patch(poly)
    # check mark
    ax.plot(
        [cx - r * 0.35, cx - r * 0.1, cx + r * 0.45],
        [cy, cy - r * 0.25, cy + r * 0.35],
        color="white",
        lw=2.4,
        zorder=11,
    )


def icon_rocket(ax, cx, cy, r, color):
    # simple rocket: triangle + body
    body = plt.Polygon(
        [(cx, cy + r), (cx - r * 0.4, cy), (cx + r * 0.4, cy)],
        closed=True,
        fc=color,
        ec=color,
        zorder=10,
    )
    ax.add_patch(body)
    rounded_box(ax, cx - r * 0.4, cy - r * 0.8, 0.8 * r, 0.8 * r, fc=color, ec="none", radius=0.003, zorder=10)
    ax.add_patch(plt.Circle((cx, cy), r * 0.18, fc="white", ec="none", zorder=11))
    # flame
    ax.add_patch(plt.Polygon([(cx - r * 0.25, cy - r * 0.8), (cx, cy - r * 1.2), (cx + r * 0.25, cy - r * 0.8)], fc="#fbbf24", ec="none", zorder=10))


def icon_org(ax, cx, cy, r, color):
    # building-style: 3 stacked rectangles
    rounded_box(ax, cx - r * 0.7, cy - r, 1.4 * r, 1.6 * r, fc="white", ec=color, lw=1.4, radius=0.003, zorder=10)
    # windows
    for col in [-0.3, 0.3]:
        for row in [-0.3, 0.1, 0.5]:
            ax.add_patch(plt.Rectangle((cx + col * r - r * 0.12, cy + row * r), r * 0.24, r * 0.18, fc=color, ec="none", zorder=11))


# ---------- main figure ----------
def make_figure(out_path: Path):
    fig = plt.figure(figsize=(17, 9.8))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pad = 0.010
    side_w = 0.165
    main_x = side_w + pad
    main_w = 1 - main_x - pad

    # vertical bands
    top_h = 0.115
    stim_h = 0.36
    prim_h = 0.16
    out_h = 0.16
    gap = 0.010

    top_y = 1 - pad - top_h
    stim_y = top_y - gap - stim_h
    prim_y = stim_y - gap - prim_h
    out_y = prim_y - gap - out_h

    # ============== TOP STRIP: 4 principles + tagline ==============
    rounded_box(ax, pad, top_y, 1 - 2 * pad, top_h, fc=COL_FRAME_BG, ec=COL_FRAME_EC, lw=1.0, radius=0.008)

    # "Our Vision" cell on far left
    vision_w = 0.13
    vx = pad + 0.012
    ax.text(vx, top_y + top_h - 0.025, "Our Vision:", ha="left", va="top", fontsize=12, fontweight="bold", color=COL_HEADER)
    ax.text(
        vx, top_y + top_h - 0.05,
        "A reusable benchmark\nfor persona drift in long\nagentic-coding sessions",
        ha="left", va="top", fontsize=8.5, color=COL_MUTED, linespacing=1.3,
    )
    # divider
    ax.plot([vx + vision_w, vx + vision_w], [top_y + 0.012, top_y + top_h - 0.012], color="#9ca3af", lw=0.8)

    principles = [
        (icon_target, "Snapshot-then-probe", "Fork the session at turn t,\nprobe without perturbing"),
        (icon_balance, "Length-matched filler", "Lorem-ipsum control strips\nthe family signal"),
        (icon_chart, "Dual measurement", "Judge-scored probes plus\njudge-free regex compliance"),
        (icon_lock, "Pre-registered analysis", "SHA-256 hashed plans\nlocked before data lands"),
    ]
    n = len(principles)
    avail_w = 1 - 2 * pad - vision_w - 0.05 - 0.18  # leave room for "Our Goal" at right
    cell_w = avail_w / n
    cell_x0 = pad + vision_w + 0.025

    for i, (icon_fn, title, sub) in enumerate(principles):
        cx_text = cell_x0 + i * cell_w
        icon_cx = cx_text + 0.018
        icon_cy = top_y + top_h / 2
        icon_fn(ax, icon_cx, icon_cy, 0.020, COL_HEADER)
        ax.text(
            cx_text + 0.045, icon_cy + 0.020, title,
            ha="left", va="center", fontsize=10.5, fontweight="bold", color=COL_HEADER,
        )
        ax.text(
            cx_text + 0.045, icon_cy - 0.015, sub,
            ha="left", va="center", fontsize=8.0, color=COL_MUTED, linespacing=1.3,
        )

    # "Our Goal" cell on far right
    goal_x = 1 - pad - 0.16
    rounded_box(ax, goal_x, top_y + 0.008, 0.155, top_h - 0.016, fc="white", ec=COL_HEADER, lw=1.2, radius=0.006, zorder=3)
    # arrow
    arr = FancyArrowPatch(
        (goal_x - 0.018, top_y + top_h / 2),
        (goal_x - 0.003, top_y + top_h / 2),
        arrowstyle="-|>", mutation_scale=18, color=COL_HEADER, lw=2.0, zorder=4,
    )
    ax.add_patch(arr)
    ax.text(goal_x + 0.0775, top_y + top_h - 0.028, "Our Goal:", ha="center", va="center", fontsize=11, fontweight="bold", color=COL_HEADER, zorder=4)
    ax.text(
        goal_x + 0.0775, top_y + 0.038,
        "Audit whether the persona\na model ships with is the\npersona users encounter\nat session end.",
        ha="center", va="center", fontsize=7.8, color=COL_MUTED, linespacing=1.3, zorder=4,
    )

    # ============== LEFT SIDEBAR ==============
    side_top = stim_y + stim_h
    side_bottom = out_y
    side_h = side_top - side_bottom
    rounded_box(ax, pad, side_bottom, side_w - pad, side_h, fc=COL_SIDEBAR_BG, ec=COL_SIDEBAR_EC, lw=1.0, radius=0.008)

    cx_side = pad + (side_w - pad) / 2
    cy = side_top - 0.022
    ax.text(cx_side, cy, "Evaluation\nCoverage", ha="center", va="top", fontsize=11.5, fontweight="bold", color=COL_HEADER, linespacing=1.2)
    cy -= 0.052
    ax.text(cx_side, cy, "23 frontier targets\nfrom 10 organizations", ha="center", va="top", fontsize=8.2, color=COL_MUTED, linespacing=1.3)
    cy -= 0.032

    # Org rows
    orgs = [
        ("Anthropic", "Haiku, Sonnet 4.5/4.6, Opus 4.1"),
        ("OpenAI", "GPT-4.1, 4o, 5, 5-mini"),
        ("Google", "Gemini 2.5 Pro, Flash"),
        ("DeepSeek", "V3"),
        ("Mistral", "Small, Medium, Large"),
        ("Cohere", "Command R7B, Command A"),
        ("NVIDIA", "Nemotron Nano, 49B, 120B"),
        ("Alibaba", "Qwen3 235B, Next 80B"),
        ("Meta", "Llama 3.3 70B"),
        ("Moonshot", "Kimi K2.6"),
    ]
    for org, models in orgs:
        # small org icon
        icon_org(ax, pad + 0.012, cy - 0.006, 0.008, COL_HEADER)
        ax.text(pad + 0.026, cy, org, ha="left", va="top", fontsize=7.4, fontweight="bold", color=COL_TEXT)
        ax.text(pad + 0.026, cy - 0.014, models, ha="left", va="top", fontsize=6.8, color=COL_MUTED)
        cy -= 0.028

    # Separator
    sep_y = cy - 0.002
    ax.plot([pad + 0.012, side_w - pad - 0.005], [sep_y, sep_y], color="#9ca3af", lw=0.6)

    # Sessions block
    cy = sep_y - 0.018
    ax.text(cx_side, cy, "3 Donor Sessions", ha="center", va="top", fontsize=9.5, fontweight="bold", color=COL_HEADER)
    cy -= 0.022
    # `conv` = conversation-turn index (user+assistant); this is the position the
    # cross-compaction experiments map to transcript lines and MUST stay verbatim
    # for reproducibility (see experiments/e08_cross_compaction/run.py TOTAL_TURNS).
    # `user` = real human-prompt count (the dataset's "user turns").
    sessions = [
        ("Session 1", "9,643 conv. turns", "1,242 user turns", "agentic coding"),
        ("Session 2", "3,746 conv. turns", "445 user turns", "manuscript writing"),
        ("Session 3", "4,918 conv. turns", "458 user turns", "non-coding docs"),
    ]
    for s, conv_turns, user_turns, kind in sessions:
        ax.text(pad + 0.012, cy, s, fontsize=7.2, fontweight="bold", color=COL_TEXT, va="top")
        ax.text(pad + 0.055, cy, f"{conv_turns} · {user_turns}", fontsize=6.2, color=COL_MUTED, va="top")
        ax.text(pad + 0.012, cy - 0.012, kind, fontsize=6.4, color=COL_MUTED, va="top", style="italic")
        cy -= 0.030

    # Separator
    sep_y = cy + 0.005
    ax.plot([pad + 0.012, side_w - pad - 0.005], [sep_y, sep_y], color="#9ca3af", lw=0.6)
    cy = sep_y - 0.018
    ax.text(cx_side, cy, "12 Snapshot Positions", ha="center", va="top", fontsize=8.6, fontweight="bold", color=COL_HEADER)
    cy -= 0.022
    ax.text(cx_side, cy, "P0_start → P1..P5\n(pre-compaction)\n→ post-C1 .. post-C6", ha="center", va="top", fontsize=7.2, color=COL_MUTED, linespacing=1.4)

    # "Probe Invocation" tag between sidebar and stimuli
    tag_x = side_w - 0.002
    tag_y_top = stim_y + stim_h * 0.55
    tag_y_bot = stim_y + stim_h * 0.45
    arr = FancyArrowPatch(
        (tag_x, tag_y_top),
        (main_x - 0.001, tag_y_top),
        arrowstyle="-|>", mutation_scale=14, color="#4b5563", lw=1.5, zorder=5,
    )
    ax.add_patch(arr)
    arr = FancyArrowPatch(
        (main_x - 0.001, tag_y_bot),
        (tag_x, tag_y_bot),
        arrowstyle="-|>", mutation_scale=14, color="#4b5563", lw=1.5, zorder=5,
    )
    ax.add_patch(arr)
    ax.text(
        (tag_x + main_x) / 2, tag_y_top + 0.014,
        "Probe\nInvocation", ha="center", va="center", fontsize=7.5, color=COL_MUTED, linespacing=1.2,
    )
    ax.text(
        (tag_x + main_x) / 2, tag_y_bot - 0.014,
        "Structured\nResponse", ha="center", va="center", fontsize=7.5, color=COL_MUTED, linespacing=1.2,
    )

    # ============== STIMULI LAYER: 4 cards ==============
    # row label
    ax.text(
        main_x + 0.005, stim_y + stim_h + 0.018,
        "Measurement Stimuli (Stimuli Layer)",
        ha="left", va="center", fontsize=11.5, fontweight="bold", color=COL_HEADER,
    )
    ax.text(
        main_x + 0.30, stim_y + stim_h + 0.018,
        "— One Primitive, Many Stimuli",
        ha="left", va="center", fontsize=11, style="italic", color=COL_MUTED,
    )

    cards = [
        dict(
            color=COL_PROBES, num="1", title="Identity Probes", icon=icon_target,
            body_main="25-probe off-task battery scored\non a 4-pt assistant-register rubric.",
            kv=[("Categories:", "Identity 4 · Experience 8 · Preference 4\nRelational 4 · Coding-Self 5 (primary)"),
                ("Paraphrases:", "n=10 per cell")],
            metric="judge-scored, granular",
        ),
        dict(
            color=COL_STRESSORS, num="2", title="Format Stressors", icon=icon_balance,
            body_main="Four format-constraint instructions\nscored by deterministic regex.",
            kv=[("Scope:", "S1 byte-exact 1-word\nS2 no-preamble (primary)\nS3 1-sentence · S4 byte-exact JSON"),
                ("Metric:", "% compliance, length ratio")],
            metric="judge-free, deployment-relevant",
        ),
        dict(
            color=COL_SWEBENCH, num="3", title="SWE-Bench Continuation", icon=icon_arrow,
            body_main="Agent shown Claude- vs.\nGPT-flavored prefix → next tool call.",
            kv=[("Coverage:", "25 cutpoints × 3 targets:\nSonnet 4.6, Mistral Small, Kimi K2.6"),
                ("Metric:", "paired Δ argument fidelity")],
            metric="tool-using, paired",
        ),
        dict(
            color=COL_ANCHOR, num="4", title="A-anchor Mitigation", icon=icon_wrench,
            body_main="~80-token user-turn block\ninserted between prefix and probe.",
            kv=[("Recipes:", "V0 identity (30t) · V2 format demo\nV0+V2 combined · 200t large variant"),
                ("Persistence:", "≥20 unanchored turns immunized")],
            metric="single-shot, deployment-grade",
        ),
    ]

    n_cards = len(cards)
    card_gap = 0.012
    card_w = (main_w - card_gap * (n_cards - 1)) / n_cards
    card_centers_x = []

    for i, card in enumerate(cards):
        x = main_x + i * (card_w + card_gap)
        card_centers_x.append(x + card_w / 2)
        # body
        rounded_box(ax, x, stim_y, card_w, stim_h, fc="white", ec=card["color"], lw=1.6, radius=0.01, zorder=2)
        # numbered title row
        title_y = stim_y + stim_h - 0.028
        ax.text(x + 0.012, title_y, f"{card['num']}. {card['title']}", ha="left", va="center", fontsize=12, fontweight="bold", color=card["color"])
        # subtle subtitle (metric)
        ax.text(x + 0.012, title_y - 0.022, card["metric"], ha="left", va="center", fontsize=8.0, style="italic", color=COL_MUTED)

        # icon — large, centered horizontally below title
        icon_cy = stim_y + stim_h - 0.105
        card["icon"](ax, x + card_w / 2, icon_cy, 0.024, card["color"])

        # divider under icon
        ax.plot([x + 0.014, x + card_w - 0.014], [icon_cy - 0.038, icon_cy - 0.038], color=card["color"], lw=0.6, alpha=0.4)

        # main body description
        body_y = icon_cy - 0.055
        ax.text(x + card_w / 2, body_y, card["body_main"], ha="center", va="top", fontsize=8.6, color=COL_TEXT, linespacing=1.4)

        # key-value items
        kv_y = body_y - 0.065
        for k, v in card["kv"]:
            ax.text(x + 0.012, kv_y, k, ha="left", va="top", fontsize=7.8, fontweight="bold", color=card["color"])
            ax.text(x + 0.012, kv_y - 0.018, v, ha="left", va="top", fontsize=7.5, color=COL_TEXT, linespacing=1.4)
            kv_y -= 0.055

    # Down-arrows from each card → primitive layer (color-matched)
    arrow_top_y = stim_y - 0.001
    arrow_bot_y = prim_y + prim_h + 0.001
    for cx, card in zip(card_centers_x, cards):
        arr = FancyArrowPatch(
            (cx, arrow_top_y),
            (cx, arrow_bot_y),
            arrowstyle="-|>",
            mutation_scale=18,
            color=card["color"],
            lw=2.4,
            zorder=5,
        )
        ax.add_patch(arr)

    # ============== PRIMITIVE LAYER ==============
    rounded_box(ax, pad, prim_y, 1 - 2 * pad, prim_h, fc=COL_PRIMITIVE_BG, ec=COL_PRIMITIVE_EC, lw=1.2, radius=0.008)

    ax.text(
        pad + 0.012, prim_y + prim_h - 0.022,
        "Snapshot-then-Probe Primitive",
        ha="left", va="top", fontsize=11.5, fontweight="bold", color=COL_HEADER,
    )
    ax.text(
        pad + 0.30, prim_y + prim_h - 0.022,
        "— shared infrastructure: every stimulus flows through this primitive",
        ha="left", va="top", fontsize=10, style="italic", color=COL_MUTED,
    )

    # Formula block
    f_y = prim_y + prim_h / 2 + 0.012
    ax.text(
        0.5, f_y,
        r"$r^{(\mathrm{claude})}_{t,p,i} \;=\; \mathcal{M}\!\left(c^{(\mathrm{claude})}_{1:t} \,\oplus\, \mathtt{FRAME} \,\oplus\, p_i\right)$"
        r"$\qquad\qquad$"
        r"$r^{(\mathrm{filler})}_{t,p,i} \;=\; \mathcal{M}\!\left(c^{(\mathrm{filler})}_{1:t} \,\oplus\, \mathtt{FRAME} \,\oplus\, p_i\right)$",
        ha="center", va="center", fontsize=13, color=COL_TEXT,
    )
    ax.text(
        0.5, f_y - 0.045,
        r"$\Delta(t, \mathcal{M}) \;=\; \frac{1}{|\mathcal{P}|}\!\sum_{p \in \mathcal{P}}\!\left[\bar{J}(r^{(\mathrm{filler})}_{t,p}) - \bar{J}(r^{(\mathrm{claude})}_{t,p})\right]$"
        r"$\qquad\qquad$"
        r"$\Delta > 0 \;\Rightarrow\;$ persona drift",
        ha="center", va="center", fontsize=13, color=COL_TEXT,
    )
    ax.text(
        0.5, prim_y + 0.012,
        "Fork is discarded after sampling — the main session is never perturbed.",
        ha="center", va="center", fontsize=8.6, style="italic", color=COL_MUTED,
    )

    # ============== OUTCOMES STRIP ==============
    rounded_box(ax, pad, out_y, 1 - 2 * pad, out_h, fc=COL_FRAME_BG, ec=COL_FRAME_EC, lw=1.0, radius=0.008)
    ax.text(
        0.5, out_y + out_h - 0.022,
        "Outcomes — Five Research Questions Answered",
        ha="center", va="top", fontsize=11.5, fontweight="bold", color=COL_HEADER,
    )

    questions = [
        ("Q1", "Existence", "Does drift exist at\ndeployment scale?", icon_chart, COL_PROBES),
        ("Q2", "Cross-org Generality", "Family-specific, or\ngeneral across orgs?", icon_org, COL_PROBES),
        ("Q3", "Compaction Reset", "Does in-session\ncompaction reset drift?", icon_arrow, COL_STRESSORS),
        ("Q4", "Anchor Mitigation", "Can a single-shot\nanchor restore register?", icon_wrench, COL_ANCHOR),
        ("Q5", "Mode-dependent Cost", "Tool-free vs tool-using\ncost direction?", icon_shield, COL_SWEBENCH),
    ]
    qn = len(questions)
    q_gap = 0.010
    q_w = (1 - 2 * pad - 0.024 - q_gap * (qn - 1)) / qn
    q_y = out_y + 0.018
    q_h = out_h - 0.06
    for i, (qid, qname, qdesc, ic, qcol) in enumerate(questions):
        qx = pad + 0.012 + i * (q_w + q_gap)
        rounded_box(ax, qx, q_y, q_w, q_h, fc="white", ec=qcol, lw=1.4, radius=0.006, zorder=2)
        # icon left
        ic(ax, qx + 0.020, q_y + q_h - 0.030, 0.014, qcol)
        # Q-id
        ax.text(qx + 0.045, q_y + q_h - 0.018, qid, ha="left", va="top", fontsize=12, fontweight="bold", color=qcol, zorder=3)
        ax.text(qx + 0.066, q_y + q_h - 0.020, qname, ha="left", va="top", fontsize=9.5, fontweight="bold", color=COL_TEXT, zorder=3)
        ax.text(qx + 0.012, q_y + q_h - 0.060, qdesc, ha="left", va="top", fontsize=8.0, color=COL_MUTED, linespacing=1.4, zorder=3)

    # ============== Save ==============
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, dpi=300)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.05, dpi=200)
    print(f"wrote {out_path} and {out_path.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    here = Path(__file__).parent
    out = here.parent / "paper" / "figures" / "fig_framework.pdf"
    make_figure(out)
