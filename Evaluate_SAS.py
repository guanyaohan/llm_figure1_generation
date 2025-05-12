import os
import csv
import torch
import clip     
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
from pdf2image import convert_from_path
import tempfile
import cairosvg   
import base64, json
from openai import OpenAI
import os


os.environ["OPENAI_API_KEY"] = ""
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)        

def ensure_png(img_path: str) -> Path:
    p = Path(img_path)
    suf = p.suffix.lower()
    if suf in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
        return p
    if suf == ".pdf":
        return pdf_to_png(p)
    if suf == ".svg":
        return svg_to_png(p)

    img = Image.open(p)
    tmp = Path(tempfile.gettempdir()) / (p.stem + ".png")
    img.save(tmp, "PNG");  return tmp


def describe_image_with_gpt(img_path: str,
                            prompt="Describe the image in one sentence.") -> str:

    b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
    data_uri = f"data:image/png;base64,{b64}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ]
        }],
        temperature=0.2,
        max_tokens=80
    )
    return resp.choices[0].message.content.strip()


def get_desc_embed(model, desc: str, device: str):
    tokens = clip.tokenize([desc]).to(device)
    with torch.no_grad():
        e = model.encode_text(tokens)[0]
    return F.normalize(e, dim=-1)

def pdf_to_png(first_page_pdf: Path) -> Path:
    pages = convert_from_path(str(first_page_pdf), dpi=300, first_page=1, last_page=1)
    tmp = Path(tempfile.gettempdir()) / (first_page_pdf.stem + "_page1.png")
    pages[0].save(tmp, "PNG")
    return tmp
def load_data(csv_path, max_rows=None, remove = None):

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

def find_image_file(dir_path, exts=('.png','.jpg','.jpeg','.svg','.pdf')):

    if not os.path.isdir(dir_path):
        return None
    for fn in sorted(os.listdir(dir_path)):
        if fn.lower().endswith(exts):
            return os.path.join(dir_path, fn)
    return None

def svg_to_png(svg_path: Path, dpi: int = 300) -> Path:

    tmp = Path(tempfile.gettempdir()) / (svg_path.stem + ".png")
    # CairoSVG 接收 dpi=96 基准；换算一下
    cairosvg.svg2png(url=str(svg_path),
                     write_to=str(tmp),
                     dpi=dpi)
    return tmp


def get_image_embed(model, preprocess, img_path: str, device: str):
    p = Path(img_path)
    if not p.exists():
        raise FileNotFoundError(f"{img_path} not found")


    if p.suffix.lower() == ".svg":
        png_path = svg_to_png(p)           # SVG → PNG
        img = Image.open(png_path)


    elif p.suffix.lower() == ".pdf":
        png_path = pdf_to_png(p)           # PDF → PNG
        img = Image.open(png_path)

    else:                              
        img = Image.open(p)

    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embed = model.encode_image(img_tensor)
    return F.normalize(embed, dim=-1)


def get_text_embeds(model, caption, title, abstract, device, max_len=200):

    cap_txt = caption[:max_len]
    ta_txt  = (title + " " + abstract)[:max_len]
    tokens = clip.tokenize([cap_txt, ta_txt]).to(device)
    with torch.no_grad():
        cap_e, ta_e = model.encode_text(tokens).chunk(2, dim=0)
    return F.normalize(cap_e, dim=-1), F.normalize(ta_e, dim=-1)

def compute_sas(sim_cap, sim_ta, w_ta=0.6, w_cap=0.4):

    return w_ta * sim_ta + w_cap * sim_cap

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)

    data = load_data("./csv/sample_200_even.csv", max_rows=10, remove = 4)

    base_dir = "/home/yguan19/llm_fig1_with_all_avenues/llm_fig1_generation/out"
    variants = ["original", "zero_shot", "improved", "zero_shot_svg", "cot_v2"]

    for row in data:
        fname   = os.path.basename(row["fig1_file_path"])
        dirname = os.path.splitext(fname)[0]


        cap_e, ta_e = get_text_embeds(
            model, row["fig1_caption"], row["paper_title"], row["abstract"], device
        )

        print(f"\n=== {fname} ===")
        for v in variants:
            if v == "original":
                img_path = os.path.join(base_dir, v, fname)
            else:
   
                subdir = os.path.join(base_dir, v, dirname)
                img_path = find_image_file(subdir) or os.path.join(base_dir, v, fname)

            img_e = get_image_embed(model, preprocess, img_path, device)
            if img_e is None:
                continue

            vis_path   = ensure_png(img_path)
            description = describe_image_with_gpt(vis_path)   
            desc_e      = get_desc_embed(model, description, device)


            sim_cap_GPT = (desc_e @ cap_e.T).item()     
            sim_ta_GPT  = (desc_e @ ta_e.T).item()   
            sim_cap = (img_e @ cap_e.T).item()
            sim_ta  = (img_e @ ta_e.T).item()
            sas     = compute_sas(sim_cap, sim_ta)
            sas_GPTdescrib = compute_sas(sim_cap_GPT, sim_ta_GPT)

            print(f"{v:12s} | sim_cap: {sim_cap:.4f} | sim_ta: {sim_ta:.4f} "
                f"| SAS: {sas:.4f} | desc: {description[:60]}…")



            print(f"{v:12s} | sim_cap_GPT: {sim_cap_GPT:.4f} | sim_ta_GPT: {sim_ta_GPT:.4f} | sas_GPTdescrib: {sas_GPTdescrib:.4f}")

if __name__ == "__main__":
    main()
