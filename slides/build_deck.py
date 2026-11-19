import math
import os
import subprocess

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RENDERS = os.path.join(HERE, "imagens", "renders")
STAGES = os.path.join(HERE, "imagens", "etapas")
OUT = os.path.join(HERE, "apresentacao.pptx")
SLIDE_W = 13.333

BG = RGBColor(0x14, 0x16, 0x1F)
PANEL = RGBColor(0x23, 0x27, 0x36)
ACCENT = RGBColor(0x4F, 0xC3, 0xF7)
WARN = RGBColor(0xFF, 0xB7, 0x4D)
TEXT = RGBColor(0xEE, 0xEE, 0xEE)
MUTED = RGBColor(0xA0, 0xA6, 0xB8)

prs = Presentation()
prs.slide_width = Inches(SLIDE_W)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(s, x, y, w, h, lines, size=20, color=TEXT, bold=False, bullets=False,
         align=None, font=None):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        level = 0
        if isinstance(line, tuple):
            line, level = line
        if bullets:
            line = ("\u2022 " if level == 0 else "\u2013 ") + line
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(8)
        if align is not None:
            p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size - 2 * level)
        r.font.bold = bold
        r.font.color.rgb = color
        if font:
            r.font.name = font
    return box


def new_slide(title, notes=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    text(s, 0.6, 0.3, 12.1, 0.9, [title], size=30, color=ACCENT, bold=True)
    rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), Inches(1.15),
                              Inches(1.6), Inches(0.05))
    rule.fill.solid()
    rule.fill.fore_color.rgb = ACCENT
    rule.line.fill.background()
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def picture(s, path, x, y, max_w, max_h, caption=None):
    with Image.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    pw, ph = w * scale, h * scale
    s.shapes.add_picture(path, Inches(x + (max_w - pw) / 2), Inches(y),
                         Inches(pw), Inches(ph))
    if caption:
        text(s, x, y + ph + 0.05, max_w, 0.6, [caption], size=14, color=MUTED,
             align=PP_ALIGN.CENTER)


def missing(s, x, y, w, h, name):
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                             Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = PANEL
    box.line.color.rgb = WARN
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "ainda não gerada: " + name
    r.font.size = Pt(14)
    r.font.color.rgb = WARN


def table(s, rows, x, y, col_widths, size=16, row_h=0.55):
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                               Inches(sum(col_widths)),
                               Inches(row_h * len(rows)))
    tbl = shape.table
    for j, w in enumerate(col_widths):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT if i == 0 else PANEL
            cell.text_frame.word_wrap = True
            r = cell.text_frame.paragraphs[0].add_run()
            r.text = val
            r.font.size = Pt(size)
            r.font.bold = i == 0
            r.font.color.rgb = BG if i == 0 else TEXT
    return shape


def flow_boxes(s, labels, y):
    n = len(labels)
    bw, gap = 1.75, 0.4
    x0 = (SLIDE_W - (n * bw + (n - 1) * gap)) / 2
    for i, lines in enumerate(labels):
        x = x0 + i * (bw + gap)
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                                 Inches(y), Inches(bw), Inches(1.4))
        box.fill.solid()
        box.fill.fore_color.rgb = PANEL
        box.line.color.rgb = ACCENT
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        for k, line in enumerate(lines):
            p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = line
            r.font.size = Pt(16 if k == 0 else 13)
            r.font.bold = k == 0
            r.font.color.rgb = TEXT
        if i < n - 1:
            arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                       Inches(x + bw + 0.06), Inches(y + 0.55),
                                       Inches(gap - 0.12), Inches(0.3))
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = ACCENT
            arrow.line.fill.background()


def explain(title, items, img=None, caption=None, notes=None, size=18):
    s = new_slide(title, notes)
    if img:
        text(s, 0.6, 1.45, 6.4, 5.6, items, size=size, bullets=True)
        picture(s, img, 7.3, 1.45, 5.4, 5.0, caption)
    else:
        text(s, 0.6, 1.45, 12.1, 5.6, items, size=size, bullets=True)
    return s


def stage_slide(title, items, notes=None):
    s = new_slide(title, notes)
    n = len(items)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    cell_w = 12.1 / cols
    cell_h = 5.6 / rows
    for k, (name, caption) in enumerate(items):
        r, c = divmod(k, cols)
        x = 0.6 + c * cell_w
        y = 1.45 + r * cell_h
        path = os.path.join(STAGES, name + ".png")
        if os.path.exists(path):
            picture(s, path, x, y, cell_w - 0.3, cell_h - 0.8, caption)
        else:
            missing(s, x + 0.1, y, cell_w - 0.5, cell_h - 0.8, name)
            text(s, x, y + cell_h - 0.75, cell_w - 0.3, 0.6, [caption], size=14,
                 color=MUTED, align=PP_ALIGN.CENTER)
    return s


def real(name):
    return os.path.join(RENDERS, name + ".png")


def ensure_renders():
    os.makedirs(RENDERS, exist_ok=True)
    for name in ("none", "heat", "flow"):
        for t in (0, 6, 9):
            out = real(f"{name}_t{t}")
            if os.path.exists(out):
                continue
            src = os.path.join(ROOT, f"procimg_{name}.mp4")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(t),
                            "-i", src, "-frames:v", "1", "-vf", "scale=600:600",
                            out], check=True)


ensure_renders()

# 1. Título
s = prs.slides.add_slide(BLANK)
s.background.fill.solid()
s.background.fill.fore_color.rgb = BG
text(s, 0.8, 2.3, 11.7, 1.2, ["Path-finding com operações de imagem"], size=44,
     color=ACCENT, bold=True)
text(s, 0.8, 3.5, 11.7, 0.8,
     ["Experimento 1: 100 unidades, um objetivo, somente OpenCV"], size=24)
text(s, 0.8, 4.4, 11.7, 1.4,
     ["Grade 400 x 400  |  112.627 células caminháveis  |  47.373 paredes",
      "Renders: procimg_*.mp4 (14 nov)  |  Imagens das etapas: slides 6, 11 e 14"],
     size=18, color=MUTED)

# 2. A ideia
s = new_slide("A ideia")
text(s, 0.6, 1.45, 12.1, 5.6, [
    "Unidades e objetivo são imagens (grades), não listas de posições",
    "O objetivo é espalhado como um campo de calor por difusão; o gradiente do "
    "calor aponta cada célula na direção do objetivo",
    "Uma unidade se move lendo o campo vetorial sob ela: esse movimento é uma "
    "advecção (cv2.remap)",
    "Os buffers são alocados uma vez no reset e reescritos a cada quadro",
    "Especificação: somente OpenCV",
], size=22, bullets=True)

# 3. Um quadro
s = new_slide("Um quadro = cinco operações de imagem")
flow_boxes(s, [
    ["Board", "imagem de rótulos"],
    ["1  Limiar", "máscaras"],
    ["2  Calor", "difusão"],
    ["3  Campo vetorial", "Sobel X / Y"],
    ["4  Máscara de parede", "zera nas paredes"],
    ["5  Remap", "move as unidades"],
], 1.9)
text(s, 0.6, 3.9, 12.1, 3.0, [
    "Cada caixa é uma sequência curta de chamadas OpenCV em imagens 400 x 400",
    "A máscara de parede (4) roda dentro da etapa do campo vetorial, antes do remap",
    "O remap (5) escreve a imagem das unidades; o próximo quadro lê essa imagem",
], size=20, bullets=True)

# 4. Entrada
s = new_slide("Entrada: uma imagem de rótulos")
table(s, [
    ["Valor", "Significado", "Definido a partir de"],
    ["0", "parede (não caminhável)", "navmap"],
    ["1", "célula livre caminhável", "navmap"],
    ["2", "unidade", "imagem de unidades > 0"],
    ["3", "objetivo", "imagem do objetivo > 0"],
], 0.6, 1.5, [1.6, 5.0, 5.5], size=18, row_h=0.6)
text(s, 0.6, 4.9, 12.1, 2.0, [
    "Prioridade: objetivo sobre unidade sobre célula livre",
    "Unidades e objetivo compartilham essa imagem, então um único limiar pode "
    "separá-los (etapa 1 da especificação)",
], size=20, bullets=True)

# 5. Op 1 tabela
s = new_slide("Op 1: máscaras de limiar a partir de uma imagem")
table(s, [
    ["Máscara", "Chamada OpenCV", "Seleciona"],
    ["walkable", "threshold(board, 0, BINARY)", "board > 0"],
    ["beyond", "threshold(board, 1, BINARY)", "board > 1: unidade ou objetivo"],
    ["goal_mask", "threshold(board, 2, BINARY)", "board > 2: objetivo"],
    ["goal_inv", "threshold(board, 2, BINARY_INV)", "board <= 2: não é objetivo"],
    ["wall_mask", "threshold(board, 0, BINARY_INV)", "board <= 0: parede"],
    ["units_mask", "bitwise_and(beyond, goal_inv)",
     "células de unidade que não são o objetivo"],
], 0.6, 1.5, [2.4, 4.6, 5.1], size=16, row_h=0.6)
text(s, 0.6, 6.0, 12.1, 1.0, [
    "BINARY_INV retorna maxval onde src <= thresh. bitwise_and é seguro aqui "
    "porque as máscaras são de 8 bits.",
], size=16, color=MUTED)

# 6. Op 1 etapas
stage_slide("Op 1: como ficam as máscaras", [
    ("01_board", "board: 0 parede, 1 livre, 2 unidade, 3 objetivo (escalado)"),
    ("02_walkable", "walkable"),
    ("03_beyond", "beyond"),
    ("04_goal_mask", "goal_mask"),
    ("05_goal_inv", "goal_inv"),
    ("06_wall_mask", "wall_mask"),
    ("07_units_mask", "units_mask"),
])

# 7. Op 2
explain("Op 2: difusão de calor", [
    "field = filter2D(field, kernel): difusão 3 x 3 (taxa 0,5; decaimento 2e-6), no mesmo buffer",
    "field /= cobertura: divide pela fração caminhável de cada vizinhança, para o "
    "calor não vazar para paredes ou borda",
    "field *= navmap: paredes forçadas a zero",
    "field += fonte do objetivo: o objetivo reinjeta calor a cada substep",
    "2 substeps por quadro; o campo começa só com a semente analítica, sem pré-aquecimento",
], img=real("heat_t0"),
    caption="campo de calor no quadro 1 (procimg_heat.mp4)",
    notes="Em cada substep: filter2D espalha o calor para os 8 vizinhos; divide "
          "corrige as paredes; a máscara zera as paredes; o objetivo soma calor "
          "de volta.")

# 8. Calor ao longo do tempo
s = new_slide("Calor ao longo do tempo (render real)")
text(s, 0.6, 1.35, 12.1, 0.8, [
    "As unidades começam à esquerda, onde o campo é fraco (azul escuro); o "
    "objetivo é o ponto brilhante à direita",
], size=18, color=TEXT)
for k, (name, cap) in enumerate([("heat_t0", "0 s"), ("heat_t6", "6 s"),
                                  ("heat_t9", "9 s")]):
    picture(s, real(name), 0.6 + k * 4.1, 2.3, 3.8, 4.6, cap)

# 10. Op 3
explain("Op 3: campo vetorial (Sobel X e Y)", [
    "avg = filter2D(field, cross) / wall_coverage: média dos vizinhos, para que "
    "as células de parede recebem um valor suave",
    "copy_to_masked: essa média é escrita só nas células de parede",
    "gx = Sobel X, gy = Sobel Y: 3 x 3, borda replicada",
    "gx e gy são zerados nas paredes",
    "mag = magnitude(gx, gy); max(mag, 1e-30); divide: vetores unitários",
    "merge(gx, gy): campo de 2 canais, CV_32FC2",
], img=real("flow_t0"),
    caption="setas = campo vetorial, a cada 8 células; preto = paredes; azul = objetivo",
    notes="O comprimento unitário importa: a velocidade de uma unidade não depende "
          "de quão íngreme é o calor, só da direção.")

# 11. Op 3 etapas
stage_slide("Op 3: Sobel X, Sobel Y e magnitude", [
    ("09_flow_gx", "Sobel X (escalado -1..1)"),
    ("10_flow_gy", "Sobel Y (escalado -1..1)"),
    ("11_flow_mag", "magnitude (escalado 0..máx)"),
])

# 12. Op 4
s = new_slide("Op 4: máscara de paredes")
text(s, 0.6, 1.45, 12.1, 5.6, [
    "setTo(0, wall_mask): células de parede recebem vetores zero",
    "Efeito: paredes não carregam movimento, e o remap nunca escolhe uma parede "
    "como destino, então nenhuma unidade entra numa parede",
], size=20, bullets=True)

# 13. Op 5
explain("Op 5: o remap move as unidades (cv2.remap)", [
    "map_x = -1 em todo lugar; a célula de cada unidade recebe as próprias "
    "coordenadas",
    "Até 8 rodadas de reivindicação: uma unidade que ainda não se moveu lê o "
    "vetor sob ela e tenta (dx, dy), depois (dx, 0), depois (0, dy); se falhar, "
    "o vizinho livre mais alinhado com o vetor",
    "O destino precisa ser caminhável e não reivindicado; a unidade libera sua "
    "célula e o destino puxa da célula antiga",
    "Células que ninguém reivindica puxam de uma célula de parede (zero), então "
    "ficam vazias",
    "cv2.remap com INTER_NEAREST: dst(x, y) = src(map_x, map_y); depois copiado "
    "de volta para units",
    "O vizinho mais próximo mantém cada unidade em uma célula de massa 1",
    "A massa é exata: 100 unidades entram, 100 saem",
], notes="As rodadas de reivindicação evitam que duas unidades caiam na mesma "
         "célula: a primeira unidade a reivindicar um destino fica com ele naquele "
         "quadro.")

# 14. Op 5 etapas
stage_slide("Op 5: unidades antes e depois do remap", [
    ("12_units_before", "unidades antes do remap"),
    ("13_map_dx", "deslocamento map_x (cinza = sem movimento)"),
    ("14_map_dy", "deslocamento map_y (cinza = sem movimento)"),
    ("15_units_after", "unidades depois do remap"),
])

# 15. Tabela de efeitos
s = new_slide("Como cada operação aparece na simulação")
table(s, [
    ["Operação", "Imagem que produz", "O que as unidades fazem com ela"],
    ["1 Limiar", "máscaras: caminhável, objetivo, parede, unidade",
     "unidades nunca contam como objetivo; a máscara de parede alimenta a etapa 4"],
    ["2 Difusão de calor", "campo alto perto do objetivo, zero nas paredes",
     "cada célula recebe um valor que cai com a distância do objetivo"],
    ["3 Sobel X / Y", "vetor de comprimento unitário por célula",
     "cada unidade conhece a direção de subida, rumo ao objetivo"],
    ["4 Máscara de parede", "vetores zero nas paredes",
     "nenhuma unidade é movida para dentro de uma parede"],
    ["5 Remap", "novas posições para cada unidade",
     "unidades se movem no máximo uma célula por eixo por quadro, ao longo dos vetores"],
], 0.6, 1.5, [2.6, 4.4, 5.1], size=16, row_h=0.8)

# 16. Render da simulação
s = new_slide("O que a simulação mostra (render real)")
text(s, 0.6, 1.35, 12.1, 0.8, [
    "Unidades (vermelho) começam à esquerda e vão em direção ao objetivo "
    "(quadrado azul, borda direita). Em 9 s (quadro 217) elas estão numa faixa no "
    "centro-inferior; provavelmente presas em um mínimo local",
], size=18)
for k, (name, cap) in enumerate([("none_t0", "0 s"), ("none_t6", "6 s"),
                                  ("none_t9", "9 s")]):
    picture(s, real(name), 0.6 + k * 4.1, 2.45, 3.8, 4.4, cap)

prs.save(OUT)
print(OUT, len(prs.slides))
