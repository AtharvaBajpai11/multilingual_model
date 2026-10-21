"""
Generates the two exact user flowchart diagrams as high-resolution (300 DPI) images:
1. paper_figures/flowchart_model.png (Model Component Hierarchy & Freezing Strategy)
2. paper_figures/flowchart_system.png (End-to-End System Pipeline & Signal Processing Flow)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs("./paper_figures", exist_ok=True)


def draw_flowchart_1():
    """Draws Flowchart 1: IsoFlow-S2ST Model Component Hierarchy & Freezing Strategy."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.axis("off")

    # Colors matching original design
    green_bg = "#e6f4ea"
    green_border = "#34a853"
    purple_bg = "#f3e8ff"
    purple_border = "#9333ea"
    dark_text = "#0f172a"
    sub_text = "#475569"

    # Box dimensions
    bw, bh = 0.65, 0.12
    bx = 0.175

    boxes = [
        {"title": "Streaming S2ST encoder", "subtitle": "Extracts features and translation tokens", "y": 0.82, "bg": green_bg, "border": green_border},
        {"title": "IsoSync-Bridge (novel)", "subtitle": "Aligns speech duration with video tempo", "y": 0.62, "bg": purple_bg, "border": purple_border},
        {"title": "Speech synthesizer", "subtitle": "Renders target-language speech", "y": 0.42, "bg": green_bg, "border": green_border},
        {"title": "Flow-matching lip-sync decoder", "subtitle": "Renders synchronized facial motion", "y": 0.22, "bg": green_bg, "border": green_border},
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch(
            (bx, b["y"]), bw, bh,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=b["bg"], edgecolor=b["border"], linewidth=1.8
        )
        ax.add_patch(rect)
        ax.text(bx + bw/2, b["y"] + bh*0.68, b["title"], ha="center", va="center", fontsize=11, fontweight="bold", color=dark_text)
        ax.text(bx + bw/2, b["y"] + bh*0.32, b["subtitle"], ha="center", va="center", fontsize=9, color=sub_text)

    # Downward connecting arrows
    arrow_props = dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#64748b", lw=2)
    ax.annotate("", xy=(bx + bw/2, 0.74), xytext=(bx + bw/2, 0.82), arrowprops=arrow_props)
    ax.annotate("", xy=(bx + bw/2, 0.54), xytext=(bx + bw/2, 0.62), arrowprops=arrow_props)
    ax.annotate("", xy=(bx + bw/2, 0.34), xytext=(bx + bw/2, 0.42), arrowprops=arrow_props)

    # Source video frames side arrow
    ax.text(0.02, 0.35, "Source video frames", fontsize=9, color=sub_text)
    ax.annotate("", xy=(bx, 0.28), xytext=(0.02, 0.28), arrowprops=arrow_props)
    ax.plot([0.02, 0.02], [0.28, 0.33], color="#64748b", lw=2)

    # Legend at bottom
    # Frozen circle (green)
    ax.add_patch(plt.Circle((0.28, 0.08), 0.015, facecolor="none", edgecolor=green_border, linewidth=2))
    ax.text(0.31, 0.08, "Frozen (pretrained)", va="center", fontsize=10, color=dark_text)

    # Trainable circle (purple)
    ax.add_patch(plt.Circle((0.58, 0.08), 0.015, facecolor="none", edgecolor=purple_border, linewidth=2))
    ax.text(0.61, 0.08, "Trainable (novel)", va="center", fontsize=10, color=dark_text)

    plt.tight_layout()
    out_path = "./paper_figures/flowchart_model.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Flowchart 1] Saved {out_path}")


def draw_flowchart_2():
    """Draws Flowchart 2: End-to-End System Pipeline & Signal Processing Flow."""
    fig, ax = plt.subplots(figsize=(8, 8.5), dpi=300)
    ax.axis("off")

    beige_bg = "#f7f5ef"
    beige_border = "#d6cfc4"
    green_bg = "#e6f4ea"
    green_border = "#34a853"
    purple_bg = "#f3e8ff"
    purple_border = "#9333ea"
    dark_text = "#0f172a"
    sub_text = "#475569"

    bw, bh = 0.65, 0.11
    bx = 0.175

    boxes = [
        {"title": "User input & web UI", "subtitle": "Upload video, select target language", "y": 0.85, "bg": beige_bg, "border": beige_border},
        {"title": "Audio & video demuxer", "subtitle": "Splits source audio and frame sequence", "y": 0.68, "bg": green_bg, "border": green_border},
        {"title": "IsoFlow-S2ST neural model", "subtitle": "Encoder, bridge, synthesizer, decoder", "y": 0.51, "bg": purple_bg, "border": purple_border},
        {"title": "Video/audio muxer", "subtitle": "Merges synced frames with translated audio", "y": 0.34, "bg": green_bg, "border": green_border},
        {"title": "Output & downloadable result", "subtitle": "Playback, comparison, alignment metrics", "y": 0.17, "bg": beige_bg, "border": beige_border},
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch(
            (bx, b["y"]), bw, bh,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor=b["bg"], edgecolor=b["border"], linewidth=1.8
        )
        ax.add_patch(rect)
        ax.text(bx + bw/2, b["y"] + bh*0.68, b["title"], ha="center", va="center", fontsize=11, fontweight="bold", color=dark_text)
        ax.text(bx + bw/2, b["y"] + bh*0.32, b["subtitle"], ha="center", va="center", fontsize=9, color=sub_text)

    arrow_props = dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#64748b", lw=2)
    ax.annotate("", xy=(bx + bw/2, 0.79), xytext=(bx + bw/2, 0.85), arrowprops=arrow_props)
    ax.annotate("", xy=(bx + bw/2, 0.62), xytext=(bx + bw/2, 0.68), arrowprops=arrow_props)
    ax.annotate("", xy=(bx + bw/2, 0.45), xytext=(bx + bw/2, 0.51), arrowprops=arrow_props)
    ax.annotate("", xy=(bx + bw/2, 0.28), xytext=(bx + bw/2, 0.34), arrowprops=arrow_props)

    # Legend at bottom
    ax.add_patch(plt.Circle((0.15, 0.05), 0.012, facecolor="none", edgecolor=beige_border, linewidth=2))
    ax.text(0.18, 0.05, "Interface / I/O", va="center", fontsize=9, color=dark_text)

    ax.add_patch(plt.Circle((0.45, 0.05), 0.012, facecolor="none", edgecolor=green_border, linewidth=2))
    ax.text(0.48, 0.05, "Signal processing", va="center", fontsize=9, color=dark_text)

    ax.add_patch(plt.Circle((0.75, 0.05), 0.012, facecolor="none", edgecolor=purple_border, linewidth=2))
    ax.text(0.78, 0.05, "Neural translation model", va="center", fontsize=9, color=dark_text)

    plt.tight_layout()
    out_path = "./paper_figures/flowchart_system.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Flowchart 2] Saved {out_path}")


if __name__ == "__main__":
    draw_flowchart_1()
    draw_flowchart_2()
