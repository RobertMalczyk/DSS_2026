"""
make_embedded_compute_slide_pptx.py

Builds embedded_compute_slide.pptx -- a single-slide draft using the
DSS 2026 conference template, designed to be copy-pasted between slide
18 and slide 19 of Robert_Malczyk.pptx without restyling.

Mirrors slide 18's layout grammar (TWO_OBJECTS):
- title across the top
- left column = three bullets with green bold lead-ins
- right column = figure (presentation_embedded_compute_slide.png)
- bottom punchline text under the figure
- footnote in italic grey under the bullets

Run after make_embedded_compute_figure.py --for-slide has produced the
slide variant of the figure.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

# ---- palette matching the existing deck ----
NAVY        = RGBColor(0x1A, 0x1A, 0x1A)
GREEN_DARK  = RGBColor(0x2E, 0x8B, 0x57)   # the lead-in / callout green
GREEN_BOLD  = RGBColor(0x1F, 0x6E, 0x3A)
BLACK_BODY  = RGBColor(0x1A, 0x1A, 0x1A)
GREY_FOOT   = RGBColor(0x60, 0x60, 0x60)
GREY_LIGHT  = RGBColor(0x7F, 0x8C, 0x8D)

TEMPLATE = r"C:\Robak\DSS2026\Claude\DSSAI26-Presentation_Template_EN.pptx"
FIGURE   = r"C:\Robak\DSS2026\Claude\presentation_embedded_compute_slide.png"
OUT      = r"C:\Robak\DSS2026\Claude\embedded_compute_slide.pptx"

p = Presentation(TEMPLATE)

# strip the three template slides cleanly: drop the slide parts via the
# package relationships, then remove the sldId entries from the deck XML.
# Otherwise the orphaned slide*.xml parts remain in the zip and cause
# "Duplicate name" warnings on save and corruption warnings on open.
sldIdLst = p.slides._sldIdLst
for sldId in list(sldIdLst):
    rId = sldId.attrib[qn("r:id")]
    p.part.drop_rel(rId)
    sldIdLst.remove(sldId)

# ---- BLANK layout (idx 6): no inherited placeholders to fight with ----
slide = p.slides.add_slide(p.slide_layouts[6])

# the BLANK layout still pulls in slide-number / footer placeholders from the
# master; drop any empty placeholder so they don't render as ghost text boxes.
for shape in list(slide.shapes):
    if shape.has_text_frame and not shape.text_frame.text.strip():
        if shape.is_placeholder:
            sp = shape._element
            sp.getparent().remove(sp)

# ============== TITLE ==============
# slide 18 reference: L=0.917, T=0.399, W=11.5, H=1.45, sz=30 bold
title_box = slide.shapes.add_textbox(Inches(0.917), Inches(0.40),
                                     Inches(11.5), Inches(1.0))
tf = title_box.text_frame
tf.word_wrap = True
para = tf.paragraphs[0]
run = para.add_run()
run.text = "Smarter beats bigger — and fits in flash"
run.font.size = Pt(30)
run.font.bold = True
run.font.color.rgb = NAVY

# subtitle line
sub = title_box.text_frame.add_paragraph()
sub.space_before = Pt(4)
r = sub.add_run()
r.text = ("What 70% recovery from 173 hand-picked features actually buys "
          "you on the device.")
r.font.size = Pt(13)
r.font.italic = True
r.font.color.rgb = GREY_FOOT

# ============== LEFT COLUMN: BULLETS ==============
# slide 18 reference: L=0.917, T=2.0, W=5.67, H=4.76
left_box = slide.shapes.add_textbox(Inches(0.917), Inches(2.0),
                                    Inches(5.67), Inches(4.5))
ltf = left_box.text_frame
ltf.word_wrap = True

BULLETS = [
    ("Compute.",
     "~13 MMACs per 3 s window. ~30 ms on a Cortex-M7 @ 400 MHz, FP32 — "
     "real-time at 50% overlap with 15× headroom for the rest of the firmware."),
    ("Memory.",
     "~250 KB total: support vectors, mel filterbank, streaming STFT buffer. "
     "Fits a $5 MCU's flash and SRAM with room for the application above it."),
    ("Hardware class.",
     "Stays one tier below the cheapest CNN baseline. The v1 small CNN already "
     "needs an MCU+NPU SoC ($30); v3 ResNet-18 and v4 transformer need a "
     "Jetson-class board ($100+)."),
]

first = True
for lead, body in BULLETS:
    if first:
        para = ltf.paragraphs[0]
        first = False
    else:
        para = ltf.add_paragraph()
        para.space_before = Pt(10)
    r = para.add_run()
    r.text = lead + " "
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = GREEN_DARK

    r2 = para.add_run()
    r2.text = body
    r2.font.size = Pt(14)
    r2.font.color.rgb = BLACK_BODY

# ---- footnote under the bullets ----
foot_box = slide.shapes.add_textbox(Inches(0.917), Inches(6.45),
                                    Inches(5.67), Inches(0.7))
ft = foot_box.text_frame
ft.word_wrap = True
fp = ft.paragraphs[0]
fr = fp.add_run()
fr.text = ("Order-of-magnitude estimates from spec-derived MAC counts, "
           "not measured cycles. v2 vs naive CNN baselines.")
fr.font.size = Pt(10)
fr.font.italic = True
fr.font.color.rgb = GREY_LIGHT

# ============== RIGHT COLUMN: FIGURE ==============
# figure aspect ~2.04:1; place at L=6.67, W=5.83, scale H accordingly
img_left   = Inches(6.55)
img_top    = Inches(2.0)
img_width  = Inches(6.20)
img_height = Inches(3.04)   # 6.20 / 2.04
slide.shapes.add_picture(FIGURE, img_left, img_top,
                         width=img_width, height=img_height)

# ============== BOTTOM PUNCHLINE (under figure) ==============
punch_box = slide.shapes.add_textbox(Inches(6.55), Inches(5.20),
                                     Inches(6.20), Inches(0.55))
pt_ = punch_box.text_frame
pt_.word_wrap = True
pp = pt_.paragraphs[0]
pr = pp.add_run()
pr.text = "“Smarter beats bigger” — the same lesson on the hardware side as on the model side."
pr.font.size = Pt(13)
pr.font.bold = True
pr.font.color.rgb = GREEN_BOLD

# small stat row just under the punchline
stat_box = slide.shapes.add_textbox(Inches(6.55), Inches(5.75),
                                    Inches(6.20), Inches(0.5))
st = stat_box.text_frame
st.word_wrap = True
sp = st.paragraphs[0]
sp.alignment = PP_ALIGN.CENTER
for label, sep in [("~30 ms latency", "  ·  "),
                   ("250 KB memory", "  ·  "),
                   ("$5 MCU", "")]:
    r = sp.add_run()
    r.text = label
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = NAVY
    if sep:
        r2 = sp.add_run()
        r2.text = sep
        r2.font.size = Pt(13)
        r2.font.color.rgb = GREY_LIGHT

# ============== SLOT NOTE (speaker reference, off-slide-ish corner) ==============
note_box = slide.shapes.add_textbox(Inches(6.55), Inches(6.40),
                                    Inches(6.20), Inches(0.7))
nt = note_box.text_frame
nt.word_wrap = True
np_ = nt.paragraphs[0]
np_.alignment = PP_ALIGN.CENTER
nr = np_.add_run()
nr.text = ("Slot between current slides 18 (\"Rebuild the features\") "
           "and 19 (\"What to take home\").")
nr.font.size = Pt(10)
nr.font.italic = True
nr.font.color.rgb = GREY_LIGHT

p.save(OUT)
print(f"Saved {OUT}")
