"""
Pipeline PDF raporu - Arial unicode font, sadece sonuclar
"""
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph,
    Spacer, PageBreak, HRFlowable, KeepTogether
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

FONTS_DIR = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Arial", FONTS_DIR + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", FONTS_DIR + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", FONTS_DIR + "ariali.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic")

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"

C_HEADER  = colors.HexColor("#1a3a5c")
C_SUBHDR  = colors.HexColor("#2e6da4")
C_GREEN   = colors.HexColor("#c8e6c9")
C_YELLOW  = colors.HexColor("#fff9c4")
C_RED     = colors.HexColor("#ffcdd2")
C_ALT     = colors.HexColor("#f0f4f8")
C_BLUE    = colors.HexColor("#e3f2fd")
C_WHITE   = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
CW = PAGE_W - 2 * MARGIN


def s(name, **kw):
    kw.setdefault("fontName", "Arial")
    kw.setdefault("fontSize", 8)
    kw.setdefault("leading", 11)
    return ParagraphStyle(name, **kw)


H1    = s("H1", fontName="Arial-Bold", fontSize=16, textColor=C_HEADER, spaceAfter=8, spaceBefore=14)
H2    = s("H2", fontName="Arial-Bold", fontSize=11, textColor=C_SUBHDR, spaceAfter=5, spaceBefore=10)
H3    = s("H3", fontName="Arial-Bold", fontSize=9,  textColor=C_HEADER, spaceAfter=4, spaceBefore=7)
BODY  = s("BODY", fontSize=8.5, spaceAfter=4, leading=13)
SMALL = s("SMALL", fontSize=7.5, spaceAfter=3, leading=11, textColor=colors.HexColor("#555"))
CAP   = s("CAP", fontSize=7.5, alignment=TA_CENTER, fontName="Arial-Italic",
          textColor=colors.HexColor("#666"), spaceAfter=5)
TITLE = s("TITLE", fontName="Arial-Bold", fontSize=24, textColor=C_HEADER,
          alignment=TA_CENTER, spaceAfter=10)
SUBT  = s("SUBT", fontSize=11, textColor=C_SUBHDR, alignment=TA_CENTER, spaceAfter=6)


def load(fname):
    with open(RESULTS_DIR / fname) as f:
        return json.load(f)


def H(text, bold=True, size=8, align=TA_CENTER, color=C_WHITE):
    fn = "Arial-Bold" if bold else "Arial"
    return Paragraph(text, s("_h", fontName=fn, fontSize=size, alignment=align,
                              textColor=color, leading=size + 3))


def C(text, bold=False, size=8, align=TA_LEFT, color=colors.black):
    fn = "Arial-Bold" if bold else "Arial"
    return Paragraph(str(text), s("_c", fontName=fn, fontSize=size, alignment=align,
                                   textColor=color, leading=size + 3))


def tbl(data, widths, style_cmds=()):
    base = [
        ("BACKGROUND", (0, 0), (-1, 0), C_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 8),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("LINEBELOW", (0, 0), (-1, 0), 1.0, C_SUBHDR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_WHITE, C_ALT]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    base.extend(style_cmds)
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle(base))
    return t


def f1color(v):
    if v >= 0.93: return C_GREEN
    if v >= 0.85: return C_YELLOW
    return C_RED


def mcolor(m):
    return {"FULL": C_GREEN, "PARTIAL": C_YELLOW, "NONE": C_RED}.get(m, C_WHITE)


# ── KAPAK ─────────────────────────────────────────────────────────────────────
def cover():
    rd = load("eval_realdata.json")
    et = load("eval_train.json")
    ts = load("threshold_sweep.json")
    final_full = ts["final"]["FULL"]
    total = et["total"]
    elems = [Spacer(1, 0.2*cm)]
    elems.append(Paragraph("Zaman Serisi Sınıflandırma Pipeline", TITLE))
    elems.append(Paragraph("Kapsamlı Sonuç Raporu", SUBT))
    elems.append(HRFlowable(width="100%", thickness=2, color=C_SUBHDR))
    elems.append(Spacer(1, 0.3*cm))

    rows = [
        [H("Metrik"), H("Değer")],
        [C("Pipeline katmanları"), C("Gate → Base → Single/Multi Anom → Meta", bold=True)],
        [C("Toplam model"), C("16  (1 gate + 3 base + 5 single + 5 multi + meta)", bold=True)],
        [C("Sınıf taksonomisi"), C("34 grup (G01–G34)", bold=True)],
        [C("Sentetik veri — FULL doğruluk"), C(f"{final_full/total*100:.1f}%   ({final_full}/{total})", bold=True)],
        [C("Gerçek veri — FULL"), C(f"{rd['FULL']}/{rd['total']}  ({rd['FULL']/rd['total']*100:.1f}%)", bold=True)],
        [C("Gerçek veri — PARTIAL"), C(f"{rd['PARTIAL']}/{rd['total']}", bold=True)],
        [C("Gerçek veri — NONE"), C(f"{rd['NONE']}/{rd['total']}", bold=True)],
    ]
    elems.append(tbl(rows, [9*cm, 8.5*cm]))

    # Tüm model özet tablosunu da kapağa ekle — sayfayı doldurur
    gd = load("gate_training.json")
    bd = load("base_training.json")
    sd = load("single_anom_training.json")
    md = load("multi_anom_training.json")
    mta = load("meta_training.json")["anom_results"]

    sum_rows = [[H("Model"), H("Katman"), H("Kazanan"), H("Test F1"), H("Test Acc")]]

    def add(name, layer, d):
        sum_rows.append([
            C(name), C(layer), C(d["best_clf"], bold=True),
            C(f"{d['test_f1']:.4f}", bold=True),
            C(f"{d['test_acc']:.4f}"),
        ])

    add("stationarity_gate", "Gate", gd)
    for k, d in bd.items():
        add(k.replace("_", " "), "Base", d)
    for k, d in sd.items():
        add(k.replace("_", " "), "Single-Anom", d)
    for k, d in md.items():
        add(k.replace("_", " "), "Multi-Anom", d)
    for k, d in mta.items():
        sum_rows.append([
            C(k.replace("_", " ")), C("Meta"), C("blend"),
            C(f"{d['f1']:.4f}", bold=True), C(f"{d['acc']:.4f}"),
        ])

    scmds = []
    for i, row in enumerate(sum_rows[1:], start=1):
        try:
            val = float(row[3].text)
            scmds.append(("BACKGROUND", (3, i), (4, i), f1color(val)))
        except Exception:
            pass

    elems.append(Spacer(1, 0.4*cm))
    elems.append(Paragraph("Tüm Modeller — Test Sonuçları", H2))
    elems.append(Paragraph("Renk: yeşil ≥ 0.93 | sarı ≥ 0.85 | kırmızı < 0.85", SMALL))
    elems.append(tbl(sum_rows, [5.0*cm, 3.2*cm, 3.2*cm, 2.5*cm, 2.5*cm], scmds))
    elems.append(PageBreak())
    return elems


# ── TÜM MODELLER ÖZET ─────────────────────────────────────────────────────────
def all_models_summary():
    gd = load("gate_training.json")
    bd = load("base_training.json")
    sd = load("single_anom_training.json")
    md = load("multi_anom_training.json")
    mta = load("meta_training.json")["anom_results"]

    rows = [[H("Model"), H("Katman"), H("Kazanan"),
             H("LGB Val F1"), H("XGB Val F1"), H("MLP Val F1"),
             H("Test F1"), H("Test Acc")]]

    def add(name, layer, d):
        rows.append([
            C(name), C(layer), C(d["best_clf"], bold=True),
            C(f"{d['val_scores']['LightGBM']:.4f}"),
            C(f"{d['val_scores']['XGBoost']:.4f}"),
            C(f"{d['val_scores']['MLP']:.4f}"),
            C(f"{d['test_f1']:.4f}", bold=True),
            C(f"{d['test_acc']:.4f}"),
        ])

    add("stationarity_gate", "Gate", gd)
    for k, d in bd.items():
        add(k.replace("_"," "), "Base", d)
    for k, d in sd.items():
        add(k.replace("_"," "), "Single-Anom", d)
    for k, d in md.items():
        add(k.replace("_"," "), "Multi-Anom", d)

    cws = [4.0*cm, 2.8*cm, 2.4*cm, 2.2*cm, 2.2*cm, 2.2*cm, 1.9*cm, 1.9*cm]
    cmds = []
    for i, row in enumerate(rows[1:], start=1):
        val = float(row[6].text)
        cmds.append(("BACKGROUND", (6, i), (7, i), f1color(val)))

    mrows = [[H("Anomali"), H("Meta F1"), H("Meta Acc")]]
    mcmds = []
    for i, (k, d) in enumerate(mta.items(), start=1):
        mrows.append([C(k.replace("_"," ")), C(f"{d['f1']:.4f}", bold=True), C(f"{d['acc']:.4f}")])
        mcmds.append(("BACKGROUND", (1, i), (2, i), f1color(d["f1"])))

    elems = [
        Paragraph("1. Tüm Model Eğitim Sonuçları", H1),
        Paragraph("Renk: yeşil ≥ 0.93 | sarı ≥ 0.85 | kırmızı < 0.85", SMALL),
        tbl(rows, cws, cmds),
        Spacer(1, 0.4*cm),
        KeepTogether([
            Paragraph("Meta-Learner Blend Sonuçları", H2),
            tbl(mrows, [6*cm, 4*cm, 4*cm], mcmds),
        ]),
        PageBreak(),
    ]
    return elems


# ── GATE + BASE DETAY ─────────────────────────────────────────────────────────
def gate_base_detail():
    gd = load("gate_training.json")
    bd = load("base_training.json")

    cws = [3.8*cm, 2.3*cm, 2.1*cm, 2.1*cm, 2.1*cm, 1.8*cm, 1.8*cm, 1.3*cm, 1.3*cm]

    grow = [[H("Model"), H("Kazanan"), H("LGB"), H("XGB"), H("MLP"),
             H("Test F1"), H("Test Acc"), H("Pos"), H("Neg")]]
    grow.append([
        C("stationarity_gate"), C(gd["best_clf"], bold=True),
        C(f"{gd['val_scores']['LightGBM']:.4f}"),
        C(f"{gd['val_scores']['XGBoost']:.4f}"),
        C(f"{gd['val_scores']['MLP']:.4f}"),
        C(f"{gd['test_f1']:.4f}", bold=True),
        C(f"{gd['test_acc']:.4f}", bold=True),
        C(str(gd["pos"]), align=TA_CENTER),
        C(str(gd["neg"]), align=TA_CENTER),
    ])

    brows = [[H("Model"), H("Kazanan"), H("LGB Val F1"), H("XGB Val F1"), H("MLP Val F1"),
              H("Test F1"), H("Test Acc"), H("Pos"), H("Neg")]]
    bcmds = []
    for i, (k, d) in enumerate(bd.items(), start=1):
        brows.append([
            C(k.replace("_"," ")), C(d["best_clf"], bold=True),
            C(f"{d['val_scores']['LightGBM']:.4f}"),
            C(f"{d['val_scores']['XGBoost']:.4f}"),
            C(f"{d['val_scores']['MLP']:.4f}"),
            C(f"{d['test_f1']:.4f}", bold=True),
            C(f"{d['test_acc']:.4f}"),
            C(str(d["pos"]), align=TA_CENTER),
            C(str(d["neg"]), align=TA_CENTER),
        ])
        bcmds.append(("BACKGROUND", (5, i), (6, i), f1color(d["test_f1"])))

    return [
        KeepTogether([Paragraph("1.1 Gate", H2), tbl(grow, cws, [("BACKGROUND", (5, 1), (6, 1), f1color(gd["test_f1"]))])]),
        Spacer(1, 0.3*cm),
        KeepTogether([Paragraph("1.2 Base Modeller", H2), tbl(brows, cws, bcmds)]),
    ]


# ── SINGLE + MULTI DETAY ─────────────────────────────────────────────────────
def anom_detail():
    sd = load("single_anom_training.json")
    md = load("multi_anom_training.json")

    cws = [3.8*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.0*cm, 2.0*cm]
    hdr = [H("Anomali"), H("Kazanan"), H("LGB Val F1"), H("XGB Val F1"),
           H("MLP Val F1"), H("Test F1"), H("Test Acc")]

    srows = [hdr[:]]
    scmds = []
    for i, (k, d) in enumerate(sd.items(), start=1):
        srows.append([
            C(k.replace("_"," ")), C(d["best_clf"], bold=True),
            C(f"{d['val_scores']['LightGBM']:.4f}"),
            C(f"{d['val_scores']['XGBoost']:.4f}"),
            C(f"{d['val_scores']['MLP']:.4f}"),
            C(f"{d['test_f1']:.4f}", bold=True),
            C(f"{d['test_acc']:.4f}"),
        ])
        scmds.append(("BACKGROUND", (5, i), (6, i), f1color(d["test_f1"])))

    mrows = [hdr[:]]
    mcmds = []
    for i, (k, d) in enumerate(md.items(), start=1):
        mrows.append([
            C(k.replace("_"," ")), C(d["best_clf"], bold=True),
            C(f"{d['val_scores']['LightGBM']:.4f}"),
            C(f"{d['val_scores']['XGBoost']:.4f}"),
            C(f"{d['val_scores']['MLP']:.4f}"),
            C(f"{d['test_f1']:.4f}", bold=True),
            C(f"{d['test_acc']:.4f}"),
        ])
        mcmds.append(("BACKGROUND", (5, i), (6, i), f1color(d["test_f1"])))

    return [
        Spacer(1, 0.3*cm),
        KeepTogether([Paragraph("1.3 Single-Anomaly Modeller", H2), tbl(srows, cws, scmds)]),
        Spacer(1, 0.3*cm),
        KeepTogether([Paragraph("1.4 Multi-Anomaly Modeller", H2), tbl(mrows, cws, mcmds)]),
    ]


# ── 34-SINIF SONUÇLARI ────────────────────────────────────────────────────────
def section_34():
    et = load("eval_train.json")
    ts = load("threshold_sweep.json")
    final_full = ts["final"]["FULL"]
    total = et["total"]
    elems = [Paragraph("2. 34-Sınıf Sentetik Veri Doğrulama", H1)]

    elems.append(Paragraph(
        f"FULL: {final_full}/{total} ({final_full/total*100:.1f}%)  |  "
        f"PARTIAL: {ts['final'].get('PARTIAL', et['PARTIAL'])}  |  "
        f"NONE: {ts['final'].get('NONE', et['NONE'])}", BODY))

    pg = et["per_group"]
    rows = [[H("GID"), H("Grup Adı"), H("FULL"), H("PARTIAL"), H("NONE"),
             H("Toplam"), H("Başarı %"), H("Durum")]]
    cmds = []
    for i, (gid_str, g) in enumerate(pg.items(), start=1):
        pct = g["FULL"] / g["total"] * 100
        durum = "✓ İyi" if pct >= 80 else ("~ Orta" if pct >= 60 else "✗ Zayıf")
        # grup adını temizle
        name = g["name"]
        for prefix in [f"G{int(gid_str):02d}_", f"G0{gid_str}_", f"G{gid_str}_"]:
            name = name.replace(prefix, "")
        name = name.replace("_", " ")
        rows.append([
            C(f"G{int(gid_str):02d}", align=TA_CENTER),
            C(name),
            C(str(g["FULL"]), align=TA_CENTER, bold=True),
            C(str(g["PARTIAL"]), align=TA_CENTER),
            C(str(g["NONE"]), align=TA_CENTER),
            C(str(g["total"]), align=TA_CENTER),
            C(f"{pct:.0f}%", align=TA_CENTER, bold=True),
            C(durum, align=TA_CENTER, bold=(pct < 60)),
        ])
        cmds.append(("BACKGROUND", (6, i), (6, i),
                     C_GREEN if pct >= 80 else (C_YELLOW if pct >= 60 else C_RED)))
        if g["FULL"] > 0:
            cmds.append(("BACKGROUND", (2, i), (2, i), C_GREEN))
        if g["NONE"] > 0:
            cmds.append(("BACKGROUND", (4, i), (4, i), C_RED))

    cws = [1.5*cm, 5.5*cm, 1.6*cm, 1.8*cm, 1.5*cm, 1.8*cm, 2.0*cm, 2.0*cm]
    elems.append(tbl(rows, cws, cmds))
    elems.append(Paragraph(
        "FULL = base + tüm anomaliler doğru  |  PARTIAL = kısmen doğru  |  NONE = hiç doğru değil  "
        "|  Renk: yeşil ≥ 80% | sarı ≥ 60% | kırmızı < 60%", CAP))
    elems.append(PageBreak())
    return elems


def KT(*items):
    """KeepTogether + öncesinde küçük boşluk."""
    return [Spacer(1, 0.25*cm), KeepTogether(list(items))]


# ── GERÇEK VERİ ANA TABLO ────────────────────────────────────────────────────
def section_realdata():
    rd = load("eval_realdata.json")
    rows_data = rd["rows"]
    elems = [Paragraph("3. Gerçek Veri Değerlendirmesi — 39 Seri", H1)]

    full_pct = rd["FULL"] / rd["total"] * 100
    elems.append(Paragraph(
        f"FULL: {rd['FULL']}/{rd['total']} ({full_pct:.1f}%)  |  "
        f"PARTIAL: {rd['PARTIAL']}  |  NONE: {rd['NONE']}", BODY))

    # Ozet kutu
    oz = [
        [H("Etiket"), H("Adet"), H("Oran")],
        [C("FULL — base + tüm anomaliler doğru"), C(str(rd["FULL"]), bold=True, align=TA_CENTER),
         C(f"{rd['FULL']/rd['total']*100:.1f}%", bold=True, align=TA_CENTER)],
        [C("PARTIAL — kısmen doğru"), C(str(rd["PARTIAL"]), align=TA_CENTER),
         C(f"{rd['PARTIAL']/rd['total']*100:.1f}%", align=TA_CENTER)],
        [C("NONE — hiç doğru değil"), C(str(rd["NONE"]), align=TA_CENTER),
         C(f"{rd['NONE']/rd['total']*100:.1f}%", align=TA_CENTER)],
    ]
    t = Table(oz, colWidths=[9*cm, 3*cm, 3*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_HEADER),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 1), (-1, 1), C_GREEN),
        ("BACKGROUND", (0, 2), (-1, 2), C_YELLOW),
        ("BACKGROUND", (0, 3), (-1, 3), C_RED),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 0.3*cm))

    # Ana detay tablosu
    elems.append(Paragraph("Seri Bazında Tahmin Karşılaştırması", H2))
    hdr = [H("CSV"), H("n"), H("GT Base"), H("GT Anom"),
           H("Pred Base"), H("Pred Anom"),
           H("Yol"), H("Match"), H("P_gate"), H("p_combo")]
    det = [hdr]
    cmds = []
    for i, r in enumerate(rows_data, start=1):
        gt_a = ", ".join(a.replace("_"," ") for a in r["exp_anoms"]) if r["exp_anoms"] else "—"
        pr_a = ", ".join(a.replace("_"," ") for a in r["pred_anoms"]) if r["pred_anoms"] else "—"
        base_ok = r["pred_base"] == r["exp_base"]
        det.append([
            C(r["csv"].replace(".csv",""), size=7),
            C(str(r["n"]), size=7, align=TA_CENTER),
            C(r["exp_base"].replace("_"," "), size=7),
            C(gt_a, size=7),
            C(r["pred_base"].replace("_"," "), size=7, bold=base_ok),
            C(pr_a, size=7),
            C(r["path"], size=6.5),
            C(r["match"], size=7, bold=True, align=TA_CENTER),
            C(f"{r['P_gate']:.4f}", size=7, align=TA_CENTER),
            C(f"{r['p_combo']:.4f}", size=7, align=TA_CENTER),
        ])
        cmds.append(("BACKGROUND", (7, i), (7, i), mcolor(r["match"])))
        cmds.append(("BACKGROUND", (4, i), (4, i), C_GREEN if base_ok else C_RED))

    # toplam = 17.4cm (A4 - 2*1.8cm margin)
    cws2 = [2.4*cm, 0.7*cm, 2.2*cm, 2.3*cm, 2.2*cm, 2.3*cm, 1.9*cm, 1.5*cm, 1.0*cm, 0.9*cm]
    elems.append(tbl(det, cws2, cmds))
    elems.append(Paragraph(
        "Match: yeşil=FULL | sarı=PARTIAL | kırmızı=NONE  |  Pred Base: yeşil=doğru | kırmızı=yanlış", CAP))

    # GT base bazında ozet
    by_base = {}
    for r in rows_data:
        b = r["exp_base"]
        if b not in by_base:
            by_base[b] = {"FULL": 0, "PARTIAL": 0, "NONE": 0, "total": 0}
        by_base[b][r["match"]] += 1
        by_base[b]["total"] += 1
    brows = [[H("GT Base"), H("FULL"), H("PARTIAL"), H("NONE"), H("Toplam"), H("FULL%")]]
    bcmds = []
    for i, (b, d) in enumerate(by_base.items(), start=1):
        pct = d["FULL"] / d["total"] * 100
        brows.append([
            C(b.replace("_"," ")),
            C(str(d["FULL"]), bold=True, align=TA_CENTER),
            C(str(d["PARTIAL"]), align=TA_CENTER),
            C(str(d["NONE"]), align=TA_CENTER),
            C(str(d["total"]), align=TA_CENTER),
            C(f"{pct:.0f}%", bold=True, align=TA_CENTER),
        ])
        bcmds.append(("BACKGROUND", (5, i), (5, i),
                      C_GREEN if pct >= 70 else (C_YELLOW if pct >= 40 else C_RED)))
    elems += KT(Paragraph("GT Base Türüne Göre Özet", H2),
                tbl(brows, [5*cm, 2*cm, 2.5*cm, 2*cm, 2.5*cm, 2.5*cm], bcmds))

    elems.append(PageBreak())
    return elems


# ── BASELINE KARSILASTIRMA ────────────────────────────────────────────────────
def section_baseline():
    bd = load("baselines.json")
    elems = [Paragraph("4. Klasik Yöntem Baseline Karşılaştırması", H1)]

    rp = bd["ruptures_anom_presence"]
    sum_rows = [
        [H("Yöntem"), H("Metrik"), H("Değer")],
        [C("ADF + KPSS"), C("Stationarity Accuracy"),
         C(f"{bd['adf_kpss_stationarity_acc']:.1%}", bold=True)],
        [C("ruptures"), C("Anomali Varlığı Precision"), C(f"{rp['precision']:.3f}", bold=True)],
        [C("ruptures"), C("Anomali Varlığı Recall"), C(f"{rp['recall']:.3f}", bold=True)],
        [C("ruptures"), C("Anomali Varlığı F1"), C(f"{rp['f1']:.3f}", bold=True)],
        [C("ruptures"), C("TP / FP / FN"),
         C(f"TP={rp['tp']}  FP={rp['fp']}  FN={rp['fn']}")],
    ]
    cmds = [
        ("BACKGROUND", (2, 1), (2, 1), f1color(bd["adf_kpss_stationarity_acc"])),
        ("BACKGROUND", (2, 4), (2, 4), f1color(rp["f1"])),
    ]
    elems.append(tbl(sum_rows, [4*cm, 7*cm, 6.5*cm], cmds))

    elems.append(Spacer(1, 0.3*cm))
    elems.append(Paragraph("Baseline — Pipeline Karşılaştırma", H2))
    comp = [
        [H("Yöntem"), H("Stationarity Acc"), H("Anom F1"), H("Gerçek Veri FULL%")],
        [C("ADF + KPSS + ruptures (baseline)"),
         C("87.2%", align=TA_CENTER), C("76.9%", align=TA_CENTER), C("—", align=TA_CENTER)],
        [C("Bu Pipeline (ensemble)", bold=True),
         C("97.3%  (gate F1)", align=TA_CENTER, bold=True),
         C("~91%  (model ort.)", align=TA_CENTER, bold=True),
         C(f"70.6%  (480/680) sentetik", align=TA_CENTER, bold=True)],
    ]
    cmds2 = [
        ("BACKGROUND", (0, 2), (-1, 2), C_BLUE),
        ("FONTNAME", (0, 2), (-1, 2), "Arial-Bold"),
    ]
    elems.append(tbl(comp, [6.5*cm, 3.5*cm, 3.5*cm, 4*cm], cmds2))

    elems.append(Spacer(1, 0.3*cm))
    elems.append(Paragraph("Baseline Seri Detayı", H2))
    drows = [[H("CSV"), H("GT Base"), H("GT Anom?"), H("ADF-p"), H("KPSS-p"),
              H("Kırılma"), H("Pred Stat?"), H("Pred Anom?")]]
    dcmds = []
    for i, r in enumerate(bd["rows"], start=1):
        gt_stat = r["gt_base"] == "stationary"
        ok_stat = gt_stat == r["pred_stationary"]
        ok_anom = r["gt_has_anom"] == r["pred_has_anom"]
        drows.append([
            C(r["csv"].replace(".csv",""), size=7),
            C(r["gt_base"].replace("_"," "), size=7),
            C("Evet" if r["gt_has_anom"] else "Hayır", size=7, align=TA_CENTER),
            C(f"{r['adf_p']:.3f}", size=7, align=TA_CENTER),
            C(f"{r['kpss_p']:.3f}", size=7, align=TA_CENTER),
            C(str(r["n_breaks"]), size=7, align=TA_CENTER),
            C("Evet" if r["pred_stationary"] else "Hayır", size=7, align=TA_CENTER),
            C("Evet" if r["pred_has_anom"] else "Hayır", size=7, align=TA_CENTER),
        ])
        dcmds.append(("BACKGROUND", (6, i), (6, i), C_GREEN if ok_stat else C_RED))
        dcmds.append(("BACKGROUND", (7, i), (7, i), C_GREEN if ok_anom else C_RED))

    elems.append(tbl(drows, [3.2*cm, 3.2*cm, 1.8*cm, 2.0*cm, 2.0*cm, 1.8*cm, 2.0*cm, 2.0*cm], dcmds))
    elems.append(PageBreak())
    return elems




# ── THRESHOLD PARAMETRELERİ ───────────────────────────────────────────────────
def section_thresholds():
    ts = load("threshold_sweep.json")
    bl = ts["blend"]

    # Formül açıklaması
    formula_rows = [
        [H("Değişken"), H("Hesaplanış"), H("Not")],
        [C("p_single", bold=True),
         C("kazanan_model.predict_proba(X_tsfresh)[:,1]"),
         C("Tek anomali modeli — validation F1'e göre seçilen tek model (LGB veya XGB).")],
        [C("p_multi", bold=True),
         C("kazanan_model.predict_proba(X_tsfresh)[:,1]"),
         C("Çift anomali modeli — yine tek kazanan model (LGB veya XGB).")],
        [C("meta_X", bold=True),
         C("[P_gate | p_base×3 | p_single×5 | p_multi×5 | derived×14 | tsfresh×777]"),
         C("805 boyutlu vektör. Tüm ara model çıktıları + orijinal tsfresh.")],
        [C("p_meta", bold=True),
         C("0.5 × XGB_anom(meta_X) + 0.5 × LGB_anom(meta_X)"),
         C("Meta katmanı her zaman XGB+LGB çiftini kaydeder ve blend kullanır.")],
        [C("p_raw", bold=True),
         C("alpha × p_single + (1−alpha) × p_multi"),
         C("alpha=0 → sadece p_multi; alpha=1 → sadece p_single.")],
        [C("p_final", bold=True),
         C("0.5 × p_meta + 0.5 × p_raw"),
         C("Meta ile ham blend eşit ağırlıkla birleştirilir.")],
        [C("Karar", bold=True),
         C("p_final >= thr  =>  anomali çıktıya eklenir"),
         C("thr = thr_anom_only (anomaly_only yolu) veya thr_combo (diğer yollar).")],
    ]

    prows = [[H("Anomali"), H("Alpha  (0=multi, 1=single)"),
              H("thr_anom_only"), H("thr_combo")]]
    for anom in ["collective_anomaly", "mean_shift", "point_anomaly", "trend_shift", "variance_shift"]:
        a = bl["alpha"][anom]
        note = "→ pure multi" if a == 0.0 else ("→ pure single" if a == 1.0 else f"→ {int((1-a)*100)}% multi / {int(a*100)}% single")
        prows.append([
            C(anom.replace("_", " ")),
            C(f"{a}  ({note})", align=TA_CENTER, bold=True),
            C(str(bl["thr_anom_only"][anom]), align=TA_CENTER),
            C(str(bl["thr_combo"][anom]), align=TA_CENTER),
        ])

    global_rows = [
        [H("Parametre"), H("Değer"), H("Ne zaman devreye girer")],
        [C("thr_stat_hard"), C(str(bl["thr_stat_hard"]), bold=True, align=TA_CENTER),
         C("P_gate ≥ bu değer → seri stationary kabul edilir, alt katmanlar atlanır")],
        [C("thr_router"), C(str(bl["thr_router"]), bold=True, align=TA_CENTER),
         C("Router skoru ≥ bu değer → multi-anom yoluna yönlendirilir")],
    ]

    return [
        Paragraph("4. Final Karar Eşikleri ve Blend Formülü", H1),
        KeepTogether([
            Paragraph("Blend Hesaplama Adımları", H2),
            tbl(formula_rows, [3.5*cm, 6.5*cm, 7.4*cm]),
        ]),
        Spacer(1, 0.3*cm),
        KeepTogether([
            Paragraph("Anomali Başına Parametreler", H2),
            tbl(prows, [3.5*cm, 5.5*cm, 4.2*cm, 4.2*cm]),
        ]),
        Spacer(1, 0.3*cm),
        KeepTogether([
            Paragraph("Global Eşikler", H2),
            tbl(global_rows, [3.5*cm, 2.5*cm, 11.4*cm]),
        ]),
    ]


# ── MAIN ─────────────────────────────────────────────────────────────────────
def build(out="pipeline_rapor.pdf"):
    doc = SimpleDocTemplate(
        out, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="Pipeline Sonuç Raporu",
    )
    story = []
    story += cover()
    story += gate_base_detail()
    story += anom_detail()
    story += [PageBreak()]
    story += section_34()
    story += section_realdata()
    story += section_thresholds()
    doc.build(story)
    print(f"PDF: {out}")


if __name__ == "__main__":
    build(str(BASE_DIR / "pipeline_rapor.pdf"))
