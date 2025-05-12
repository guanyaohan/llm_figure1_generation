import os, csv, base64, json, tempfile
from pathlib import Path
from typing import Dict, List

from PIL import Image
from pdf2image import convert_from_path
import cairosvg
from openai import OpenAI
from tqdm import tqdm


def pdf_to_png(pdf_path: Path) -> Path:
    pages = convert_from_path(str(pdf_path), dpi=300, first_page=1, last_page=1)
    tmp = Path(tempfile.gettempdir()) / (pdf_path.stem + "_page1.png")
    pages[0].save(tmp, "PNG")
    return tmp

def svg_to_png(svg_path: Path, dpi=300) -> Path:
    tmp = Path(tempfile.gettempdir()) / (svg_path.stem + ".png")
    cairosvg.svg2png(url=str(svg_path), write_to=str(tmp), dpi=dpi)
    return tmp

def ensure_png(img_path: str) -> Path:
    p = Path(img_path)
    suf = p.suffix.lower()
    if suf in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        return p
    if suf == ".pdf":
        return pdf_to_png(p)
    if suf == ".svg":
        return svg_to_png(p)

    img = Image.open(p)
    tmp = Path(tempfile.gettempdir()) / (p.stem + ".png")
    img.save(tmp, "PNG")
    return tmp


# Set the environment variable directly in the Python script
os.environ["OPENAI_API_KEY"] = ""
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)  

SYSTEM_PROMPT = (
"Given the paper's context and figure image, return a JSON object with these 6 scores (0-100): "
"**SAS**, **IDS**, **AS**, **LS**, **CPC**, **Overall**. Follow these guidelines:\n\n"

"**Semantic Alignment Score (SAS)** - "
"• Does the example CLEARLY demonstrate the paper's core technical mechanism? "
"• Is it realistic (not misleading) yet memorable/interesting? "
"• Does it show input/output relationships or key components?\n\n"

"**Information Density Score (IDS)** - "
"• Substance per square inch (avoid sparse/wasted space) "
"• Does it hint at methodological scope without overwhelming? "
"• Can caption explain key elements without requiring paper context?\n\n"

"**Aesthetic Score (AS)** - "
"• First-glance visual appeal (layout balance, modern styling) "
"• Effective use of design principles (contrast, alignment, hierarchy) "
"• Purposeful whitespace and typography choices\n\n"

"**Legibility Score (LS)** - "
"• Readability at 100% zoom (min 8pt text for print) "
"• Clear visual hierarchy for guided viewing "
"• Label precision (no typos, unambiguous annotations)\n\n"

"**Color-Palette Consistency (CPC)** - "
"• Intentional color semantics (e.g., blue=input, red=output) "
"• Accessibility compliance (contrast ratios >4.5:1) "
"• Avoids rainbow palettes without narrative purpose\n\n"

"**Overall Score** - "
"• Self-contained explanation (figure + caption = standalone understanding) "
"• Copyright compliance (emoji/icon licensing, vector exports) "
"• Novelty factor for conference poster potential "
"• Iteration polish (avoids first-draft artifacts)\n\n"

"**Critical Red Flags**: "
"- Overly cherry-picked examples undermining credibility "
"- Meaningless decorative elements distracting from substance "
"- Unlabeled abbreviations/domain-specific jargon "
"- Crowded layouts requiring zoomed inspection\n\n"

"Return ONLY valid JSON with scores and NO commentary."
)

def score_figure(paper: Dict, png_path: Path, model="gpt-4o-mini") -> Dict[str, int]:
  

    b64 = base64.b64encode(png_path.read_bytes()).decode()
    data_uri = f"data:image/png;base64,{b64}"


    user_content = [
        {
            "type": "text",
            "text": (
                f"Paper Title: {paper['paper_title']}\n"
                f"Abstract: {paper['abstract']}\n"
                f"Figure 1 Caption: {paper['fig1_caption']}\n\n"
                "Please evaluate the figure on the 5 criteria."
            ),
        },
        {"type": "image_url", "image_url": {"url": data_uri}},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=100,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def read_csv(csv_path: str, max_rows=None, remove= None) -> List[Dict]:


    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            if remove is not None and i == remove - 1:
                continue
            rows.append(row)
    return rows


def main():
    papers = read_csv("./csv/sample_200_even.csv", max_rows=10, remove =4)  
    base_dir = "/home/yguan19/llm_fig1_with_all_avenues/llm_fig1_generation/out"
    variants = ["original", "zero_shot", "improved", "zero_shot_svg", "cot_v2"]

    for p in papers:
        fname   = os.path.basename(p["fig1_file_path"])
        dirname = os.path.splitext(fname)[0]

        print(f"\n=== {fname} ===")
        for v in variants:
            if v == "original":
                img_path = Path(base_dir) / v / fname
            else:
                subdir   = Path(base_dir) / v / dirname
              
                cand = next((subdir / f for f in os.listdir(subdir)
                             if f.lower().endswith((".png",".jpg",".jpeg",".svg",".pdf"))),
                            None)
                img_path = cand or (Path(base_dir) / v / fname)

            try:
                png_path = ensure_png(img_path)
                scores   = score_figure(p, png_path)
                print(f"{v:12s} | {scores}")
            except Exception as e:
                print(f"{v:12s} | ✖ {e}")

if __name__ == "__main__":
    main()
