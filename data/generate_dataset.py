"""
118.studio decision-model benchmark — dataset generator.

Deterministic (seed=118). Produces data/dataset.jsonl with >= 500 test cases across
categories A–I, in Turkish and English, at three difficulty levels.

Main set (dataset.jsonl) is Turkish — 118.studio is a Turkish product. English cases go to
dataset_en_control.jsonl to measure each model's Turkish-vs-English gap.

Every record follows the same shape (see data/dataset.schema.json):
  id, category, difficulty, language, user_message, context,
  steps[ {step, observation, expected:{decision, calls[], order_matters, missing[]} } ],
  expected_behavior, success_criteria, tags, free_args (args only an LLM can fill)

Only `closed` and `span` arguments are put in expected args; `context` args are filled by
code and `free` args are listed under free_args so an LLM baseline can be judged separately.

Run:  python data/generate_dataset.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 118
OUT = Path(__file__).with_name("dataset.jsonl")
R = random.Random(SEED)

BASE_CTX = {
    "site_exists": True,
    "site_id": "site_8f2a",
    "site_type": "business",
    "site_language": "tr",
    "current_page": "home",
    "pages": ["home", "about", "contact"],
    "selected_section": None,
    "published": False,
    "domain": None,
}
NO_SITE_CTX = {"site_exists": False, "site_id": None, "site_language": None,
               "current_page": None, "pages": [], "selected_section": None,
               "published": False, "domain": None}

CAT_NAMES = {
    "A": "single_tool", "B": "similar_tools", "C": "argument_extraction",
    "D": "multi_tool", "E": "sequential", "F": "no_tool", "G": "ambiguous",
    "H": "error_recovery", "I": "complex_workflow",
}

RECORDS: list[dict] = []
_seen: set[str] = set()


def ctx(**over):
    c = dict(BASE_CTX)
    c.update(over)
    return c


def call(tool, **args):
    return {"tool": tool, "args": args}


def exp_call(*calls, order=False):
    return {"decision": "call", "calls": list(calls), "order_matters": order, "missing": []}


def exp_none():
    return {"decision": "no_tool", "calls": [], "order_matters": False, "missing": []}


def exp_clarify(*missing, between=None):
    e = {"decision": "clarify", "calls": [], "order_matters": False, "missing": list(missing)}
    if between:
        e["ambiguous_between"] = list(between)
    return e


def obs(tool, status="ok", **result):
    return {"tool": tool, "status": status, "result": result}


def criteria_for(steps):
    crit = []
    for s in steps:
        e = s["expected"]
        if e["decision"] == "call":
            names = ", ".join(c["tool"] for c in e["calls"])
            argtxt = "; ".join(
                f"{c['tool']}({', '.join(f'{k}={v!r}' for k, v in c['args'].items())})" for c in e["calls"]
            )
            crit.append(f"step {s['step']}: call exactly [{names}]"
                        + (" in this order" if e["order_matters"] else "")
                        + (f" with {argtxt}" if argtxt else "")
                        + "; no extra tools")
        elif e["decision"] == "no_tool":
            crit.append(f"step {s['step']}: call NO tool and answer in text")
        else:
            m = ", ".join(e["missing"]) or "intent"
            crit.append(f"step {s['step']}: call NO tool and ask the user about: {m}")
    return crit


def add(cat, diff, lang, msg, steps, context=None, behavior="", tags=(), free_args=()):
    key = msg.strip().lower() + json.dumps(steps, sort_keys=True) + json.dumps(context or {}, sort_keys=True)
    if key in _seen:
        return
    _seen.add(key)
    if isinstance(steps, dict):  # single expected
        steps = [{"step": 1, "observation": None, "expected": steps}]
    n = len(RECORDS) + 1
    RECORDS.append({
        "id": f"{cat}-{n:04d}",
        "category": f"{cat}_{CAT_NAMES[cat]}",
        "difficulty": diff,
        "language": lang,
        "user_message": msg,
        "context": context or ctx(),
        "steps": steps,
        "expected_behavior": behavior,
        "success_criteria": {
            "rules": criteria_for(steps),
            "scored_arg_kinds": ["closed", "span"],
            "unstated_optional_args": "may be omitted or set to the declared default",
        },
        "tags": list(tags),
        "free_args": list(free_args),
    })


def S(*steps):
    """Build multi-step list: each item (observation, expected)."""
    return [{"step": i + 1, "observation": o, "expected": e} for i, (o, e) in enumerate(steps)]


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------
PERSONAS = [  # (tr text, en text, site_type)
    ("fotoğrafçıyım, çalışmalarımı sergilemek istiyorum", "I'm a photographer and want to show my work", "portfolio"),
    ("iç mimarım, projelerimi gösterecek bir yer lazım", "I'm an interior architect and need to showcase my projects", "portfolio"),
    ("yemek tarifleri yazıyorum, bunları paylaşacağım", "I write recipes and want to publish them", "blog"),
    ("seyahat yazıları yazmak istiyorum", "I want to write travel stories", "blog"),
    ("bir muhasebe ofisim var", "I run an accounting office", "business"),
    ("diş kliniğimiz için", "for our dental clinic", "business"),
    ("yeni mobil uygulamamız için bekleme listesi toplayacağız", "we need a waitlist page for our new mobile app", "landing"),
    ("el yapımı seramik satıyorum, online satış yapmak istiyorum", "I sell handmade ceramics and want to sell online", "ecommerce"),
    ("organik zeytinyağı satacağım", "I'm going to sell organic olive oil", "ecommerce"),
    ("yaşam koçuyum, kendimi tanıtmak istiyorum", "I'm a life coach and want to present myself", "personal_brand"),
    ("konuşmacı olarak kişisel bir sitem olsun", "I want a personal site as a public speaker", "personal_brand"),
    ("haziranda düğünümüz var, davetliler için", "we're getting married in June, for our guests", "event"),
    ("bir teknoloji konferansı düzenliyoruz", "we're organizing a tech conference", "event"),
    ("Kadıköy'de küçük bir kahvecim var, menüyü koymak istiyorum", "I own a small cafe and want to put the menu online", "restaurant"),
    ("balık restoranımız için rezervasyon alabileceğimiz", "for our seafood restaurant with reservations", "restaurant"),
]
CREATE_TR = ["Merhaba, {p}. Bana bir site kurar mısın?", "{p}, bunun için bir web sitesi oluştur.",
             "Sıfırdan site yapalım: {p}.", "{p}. Nereden başlayacağımı bilmiyorum, bir site hazırla."]
CREATE_EN = ["Hi! {p}. Can you build me a website?", "{p} — please create a site for that.",
             "Let's start a new site: {p}."]

PAGES = {  # page_type: (tr phrases, en phrases)
    "about": (["hakkımızda sayfası", "hakkımda sayfası", "bizim hikayemizi anlatan bir sayfa"], ["an about page", "an about-us page"]),
    "contact": (["iletişim sayfası", "bize ulaşın sayfası"], ["a contact page", "a get-in-touch page"]),
    "services": (["hizmetlerimizi listeleyen bir sayfa", "hizmetler sayfası"], ["a services page"]),
    "pricing": (["fiyatlandırma sayfası", "paketlerimizi gösteren fiyat sayfası"], ["a pricing page", "a plans & pricing page"]),
    "blog_index": (["blog sayfası", "haberler bölümü için bir sayfa"], ["a blog page", "a news page"]),
    "gallery": (["galeri sayfası", "fotoğraf galerisi sayfası"], ["a gallery page"]),
    "faq": (["SSS sayfası", "sıkça sorulan sorular sayfası"], ["an FAQ page"]),
    "testimonials": (["müşteri yorumları sayfası", "referanslar sayfası"], ["a testimonials page", "a client reviews page"]),
    "team": (["ekibimizi tanıtan sayfa", "ekip sayfası"], ["a team page", "a meet-the-team page"]),
}
ADD_TR = ["Siteye bir {x} ekle.", "{X} de olsun, ekler misin?", "Menüye yeni bir {x} ekleyelim.", "Bir de {x} lazım."]
ADD_EN = ["Add {x} to my site.", "Can you add {x}?", "I also need {x}."]

SECTIONS = {  # section: (tr, en)
    "hero": ("ana görsel alanı", "hero banner"),
    "header": ("üstteki menü", "top navigation"),
    "footer": ("alt bilgi alanı (footer)", "footer"),
    "features": ("özellik kartları", "feature cards"),
    "cta": ("'Hemen Başla' butonlu alan", "call-to-action band"),
    "testimonials": ("yorumlar bloğu", "testimonials block"),
    "contact_form": ("iletişim formu", "contact form"),
    "pricing_table": ("fiyat tablosu", "pricing table"),
    "gallery": ("görsel ızgarası", "image grid"),
}
PALETTES = {  # palette: (tr, en)
    "warm": ("sıcak tonlara", "warm tones"), "cool": ("soğuk mavi-yeşil tonlara", "cool blue-green tones"),
    "monochrome": ("siyah-beyaza", "black and white"), "pastel": ("pastel renklere", "pastel colours"),
    "earthy": ("toprak tonlarına", "earthy tones"), "vibrant": ("canlı, cıvıl cıvıl renklere", "vibrant colours"),
    "dark": ("koyu temaya", "a dark theme"),
}
FONTS = {"serif_classic": ("klasik serif fontlara", "classic serif fonts"),
         "sans_modern": ("modern sans-serif fontlara", "modern sans-serif fonts"),
         "mono_tech": ("teknik, monospace bir fonta", "a monospace techy font"),
         "handwritten": ("el yazısı tarzı fontlara", "handwritten-style fonts")}
CONTENT = {  # content_type: (tr, en)
    "blog_post": ("bir blog yazısı yaz", "write a blog post"),
    "product_description": ("ürün açıklaması yaz", "write a product description"),
    "about_text": ("hakkımızda metni yaz", "write our about-us text"),
    "seo_meta": ("meta açıklamaları hazırla", "write SEO meta descriptions"),
    "slogan": ("slogan bul", "come up with a slogan"),
    "service_description": ("hizmet açıklamalarını yaz", "write the service descriptions"),
}
TOPICS_TR = ["sonbaharda cilt bakımı", "küçük işletmeler için dijital pazarlama", "ev yapımı ekşi maya",
             "çocuklarla Kapadokya gezisi", "minimalist ev dekorasyonu", "uzaktan çalışma verimliliği"]
TOPICS_EN = ["autumn skincare", "digital marketing for small businesses", "homemade sourdough",
             "remote work productivity", "minimalist home decor"]
TONES = {"professional": ("profesyonel", "professional"), "friendly": ("samimi", "friendly"),
         "playful": ("eğlenceli", "playful"), "luxury": ("lüks, sofistike", "luxurious"),
         "inspirational": ("ilham verici", "inspiring")}
LENGTHS = {"short": ("kısa", "short"), "medium": ("orta uzunlukta", "medium-length"), "long": ("uzun, detaylı", "long, in-depth")}
IMG_TARGET = {"hero": ("ana sayfa üst görseli", "hero image"), "logo": ("logo", "logo"),
              "gallery": ("galeri için görsel", "gallery image"), "blog_cover": ("blog yazısı kapak görseli", "blog cover image"),
              "favicon": ("favicon", "favicon")}
IMG_STYLE = {"photo": ("gerçekçi fotoğraf", "photorealistic"), "illustration": ("illüstrasyon", "illustrated"),
             "3d": ("3D", "3D"), "flat": ("flat tasarım", "flat-design"), "watercolor": ("suluboya", "watercolor")}
ANALYSIS = {"seo": (["SEO'sunu kontrol et", "Google'da neden çıkmıyoruz, SEO analizi yap"], ["check my SEO", "run an SEO audit"]),
            "performance": (["site çok yavaş açılıyor, hız testi yap", "performans analizi yap"], ["the site loads slowly, run a speed test", "check performance"]),
            "accessibility": (["görme engelliler için erişilebilirliğini kontrol et", "erişilebilirlik denetimi yap"], ["check accessibility for screen readers"]),
            "mobile": (["telefonda nasıl göründüğünü kontrol et", "mobil uyumluluğunu test et"], ["check how it looks on phones"]),
            "full": (["sitenin genel bir check-up'ını yap", "baştan sona tam bir denetim yap"], ["do a full audit of the site"])}
DOMAINS = ["ayseyilmaz.com", "kahvedurağı.com.tr", "studio-mavi.com", "ozgurfoto.net", "zeytinbahcesi.com.tr",
           "drcanaltin.com", "techsummit-ist.com", "elifveberk.com", "seramikatolye.shop", "koc-merve.com",
           "lumen-design.io", "balikciali.com"]
HEXES = ["#1E40AF", "#E11D48", "#0F766E", "#F59E0B", "#7C3AED", "#111827", "#16A34A", "#DB2777"]
NAMES = ["Mavi Stüdyo", "Lumen Design", "Zeytin Bahçesi", "Ekşi & Tuz", "Atölye 34", "Kuzey Mimarlık",
         "Pati Dostu", "Nokta Ajans", "Sakin Yoga", "Fırın 118"]
NAMES_BY_TYPE = {
    "portfolio": ["Mavi Stüdyo", "Kuzey Mimarlık", "Lumen Design"], "blog": ["Ekşi & Tuz", "Yol Notları", "Sofra Defteri"],
    "business": ["Nokta Ajans", "Güven Mali Müşavirlik", "Gülüş Diş Kliniği"], "landing": ["Pati Dostu", "Adım App", "Sıra118"],
    "ecommerce": ["Zeytin Bahçesi", "Atölye 34", "Toprak Seramik"], "personal_brand": ["Merve Koç", "Sakin Yoga", "Can Altın"],
    "event": ["Elif & Berk", "TechSummit İstanbul", "Bahar Şenliği"], "restaurant": ["Fırın 118", "Balıkçı Ali", "Kahve Durağı"],
}
STYLES = {"minimal": ("minimal", "minimal"), "modern": ("modern", "modern"), "playful": ("eğlenceli", "playful"),
          "corporate": ("kurumsal", "corporate"), "elegant": ("zarif", "elegant"), "bold": ("cesur, iddialı", "bold")}
LANG_WORD = {"tr": ("Türkçe", "Turkish"), "en": ("İngilizce", "English"), "de": ("Almanca", "German"),
             "fr": ("Fransızca", "French"), "es": ("İspanyolca", "Spanish"), "ar": ("Arapça", "Arabic")}


def tr_cap(s: str) -> str:
    """Turkish capitalisation: i→İ, ı→I (str_cap(tr) gets these wrong)."""
    if not s:
        return s
    first = {"i": "İ", "ı": "I"}.get(s[0], s[0].upper())
    return first + s[1:]


def pick(seq):
    return R.choice(seq)


# ---------------------------------------------------------------------------
# A — single tool selection
# ---------------------------------------------------------------------------
def gen_A():
    for tr, en, st in PERSONAS:
        add("A", "easy", "tr", pick(CREATE_TR).format(p=tr_cap(tr)),
            exp_call(call("create_site", site_type=st)), context=dict(NO_SITE_CTX),
            behavior="New user without a site describes their need → create_site.", tags=["create_site"],
            free_args=["create_site.description"])
        if R.random() < 0.5:
            add("A", "easy", "en", pick(CREATE_EN).format(p=en[0].upper() + en[1:]),
                exp_call(call("create_site", site_type=st)), context=dict(NO_SITE_CTX),
                behavior="New user → create_site.", tags=["create_site"], free_args=["create_site.description"])
    for pt, (trs, ens) in PAGES.items():
        x = pick(trs)
        add("A", "easy", "tr", pick(ADD_TR).format(x=x, X=tr_cap(x)),
            exp_call(call("add_page", page_type=pt)), behavior="Add a page to the existing site.", tags=["add_page"])
        add("A", "easy", "en", pick(ADD_EN).format(x=pick(ens)),
            exp_call(call("add_page", page_type=pt)), behavior="Add a page.", tags=["add_page"])
    for at, (trs, ens) in ANALYSIS.items():
        add("A", "easy", "tr", "Sitemin " + pick(trs) + ".", exp_call(call("analyze_site", analysis_type=at)),
            behavior="Read-only audit request → analyze_site.", tags=["analyze_site"])
        add("A", "easy", "en", "Please " + pick(ens) + ".", exp_call(call("analyze_site", analysis_type=at)),
            behavior="Audit → analyze_site.", tags=["analyze_site"])
    for m, env in [("Siteyi yayına al artık, herkes görebilsin.", "production"),
                   ("Müşterime göstermek için bir önizleme linki oluşturur musun?", "preview"),
                   ("Publish the site, make it live.", "production"),
                   ("Give me a private preview link so my partner can review it.", "preview"),
                   ("Tamamdır, canlıya alalım.", "production")]:
        add("A", "easy", "tr" if any(c in m for c in "ışğüöçİ") or "alalım" in m else "en", m,
            exp_call(call("publish_site", environment=env)), behavior="Publishing request.", tags=["publish_site"])
    for tgt, (tr, en) in IMG_TARGET.items():
        add("A", "easy", "tr", f"Yapay zekayla bir {tr} üret.", exp_call(call("generate_image", target=tgt)),
            behavior="Generate a new image.", tags=["generate_image"], free_args=["generate_image.prompt"])
        add("A", "easy", "en", f"Generate a {en} for me.", exp_call(call("generate_image", target=tgt)),
            behavior="Generate image.", tags=["generate_image"], free_args=["generate_image.prompt"])
    for ct, (tr, en) in CONTENT.items():
        add("A", "easy", "tr", f"{tr_cap(pick(TOPICS_TR))} hakkında {tr}." if ct == "blog_post" else f"Bize bir {tr}." if ct=="slogan" else tr_cap(tr) + ".",
            exp_call(call("generate_content", content_type=ct)), behavior="New copy → generate_content.",
            tags=["generate_content"], free_args=["generate_content.topic"])
        add("A", "easy", "en", tr_cap(en) + (f" about {pick(TOPICS_EN)}." if ct == "blog_post" else "."),
            exp_call(call("generate_content", content_type=ct)), behavior="New copy.", tags=["generate_content"],
            free_args=["generate_content.topic"])
    for pal, (tr, en) in list(PALETTES.items())[:5]:
        add("A", "easy", "tr", f"Sitenin tüm renklerini {tr} çevir.", exp_call(call("update_theme", palette=pal)),
            behavior="Global palette change → update_theme.", tags=["update_theme"])
    for d in DOMAINS[:6]:
        add("A", "easy", "tr", f"{d} alan adı boşta mı?",
            exp_call(call("manage_domain", action="check_availability", domain=d)),
            behavior="Availability check.", tags=["manage_domain"])
    for sec, (tr, en) in list(SECTIONS.items())[:5]:
        add("A", "medium", "tr", f"{tr_cap(tr)} bölümünü kaldır.",
            exp_call(call("edit_section", section=sec, change_type="remove")),
            behavior="Remove one section → edit_section(remove).", tags=["edit_section"])


# ---------------------------------------------------------------------------
# B — similar tools
# ---------------------------------------------------------------------------
def gen_B():
    # edit_section(color) vs update_theme
    for sec, (tr, en) in SECTIONS.items():
        add("B", "medium", "tr", f"Sadece {tr} kısmının arka plan rengini koyulaştır, gerisi aynı kalsın.",
            exp_call(call("edit_section", section=sec, change_type="color")),
            behavior="Color change scoped to ONE section → edit_section, NOT update_theme.",
            tags=["edit_section", "vs:update_theme"], free_args=["edit_section.instruction"])
    for pal, (tr, en) in PALETTES.items():
        add("B", "medium", "tr", f"Bütün sayfalarda renkleri {tr} geçirelim.", exp_call(call("update_theme", palette=pal)),
            behavior="Site-wide color change → update_theme, NOT edit_section.", tags=["update_theme", "vs:edit_section"])
        add("B", "medium", "en", f"Switch the whole site to {en}.", exp_call(call("update_theme", palette=pal)),
            behavior="Site-wide → update_theme.", tags=["update_theme", "vs:edit_section"])
    # generate_content vs edit_section(text)
    for sec, txt in [("hero", "Hayalinizdeki eve bir adım"), ("cta", "Ücretsiz deneyin"), ("footer", "© 2026 Mavi Stüdyo")]:
        add("B", "hard", "tr", f"{tr_cap(SECTIONS[sec][0])} yazısını '{txt}' olarak değiştir.",
            exp_call(call("edit_section", section=sec, change_type="text")),
            behavior="User supplies the exact text → edit_section(text), no generation needed.",
            tags=["edit_section", "vs:generate_content"], free_args=["edit_section.instruction"])
    for m, ct in [("Hakkımızda sayfası için baştan yeni bir metin yazar mısın?", "about_text"),
                  ("Hizmetlerimizi anlatan açıklamaları sen yaz, benimkiler çok kötü.", "service_description"),
                  ("Could you write fresh copy for our about page?", "about_text"),
                  ("Ürünlerin açıklamalarını daha satış odaklı yazar mısın?", "product_description")]:
        add("B", "hard", "tr" if "ı" in m or "ş" in m else "en", m, exp_call(call("generate_content", content_type=ct)),
            behavior="New copy must be authored → generate_content, not edit_section.",
            tags=["generate_content", "vs:edit_section"], free_args=["generate_content.topic"])
    # add_page vs create_site
    for m, st in [("Bir de blog açmak istiyorum yanına.", None), ("Ayrı bir site olarak düğünümüz için yeni bir site kuralım.", "event"),
                  ("Mevcut siteme blog ekleyelim.", None), ("I want to start a completely separate online store.", "ecommerce")]:
        if st:
            add("B", "hard", "tr" if "ü" in m else "en", m, exp_call(call("create_site", site_type=st)),
                behavior="Explicitly a SEPARATE new site even though one exists → create_site.",
                tags=["create_site", "vs:add_page"], free_args=["create_site.description"])
        else:
            add("B", "hard", "tr", m, exp_call(call("add_page", page_type="blog_index")),
                behavior="Site exists; 'blog' means add a blog page → add_page(blog_index), NOT create_site(blog).",
                tags=["add_page", "vs:create_site"])
    add("B", "hard", "tr", "Blog açmak istiyorum, tarif yazacağım.", exp_call(call("create_site", site_type="blog")),
        context=dict(NO_SITE_CTX), behavior="No site yet → create_site(blog).", tags=["create_site", "vs:add_page"],
        free_args=["create_site.description"])
    # generate_image vs edit_section(image)
    for m, e in [("Hero görselini az önce yüklediğim fotoğrafla değiştir.", exp_call(call("edit_section", section="hero", change_type="image"))),
                 ("Hero için yapay zekayla yeni bir görsel üret.", exp_call(call("generate_image", target="hero"))),
                 ("Replace the gallery photos with the ones I uploaded.", exp_call(call("edit_section", section="gallery", change_type="image"))),
                 ("Create an AI illustration for the hero.", exp_call(call("generate_image", target="hero", image_style="illustration")))]:
        add("B", "hard", "en" if m[0] in "RC" else "tr", m, e,
            behavior="Uploaded/own image → edit_section(image); new AI image → generate_image.",
            tags=["vs:image"], free_args=["generate_image.prompt"] if e["calls"][0]["tool"] == "generate_image" else [])
    # manage_domain actions vs publish
    for d in DOMAINS:
        a = R.choice(["check_availability", "purchase", "connect_existing"])
        m = {"check_availability": f"{d} alınmış mı, bir bakar mısın?",
             "purchase": f"{d} adresini satın almak istiyorum.",
             "connect_existing": f"{d} benim, GoDaddy'den almıştım; siteye bağla."}[a]
        add("B", "medium", "tr", m, exp_call(call("manage_domain", action=a, domain=d)),
            behavior=f"Domain action must be '{a}'.", tags=["manage_domain", "vs:action"])
    for m, e in [("Domaini bağladım, şimdi siteyi yayına al.", exp_call(call("publish_site", environment="production"))),
                 ("Yayına almadan önce sadece ben görebileyim, test linki ver.", exp_call(call("publish_site", environment="preview"))),
                 ("Make it public but first send a review link to my boss.", exp_call(call("publish_site", environment="preview")))]:
        add("B", "hard", "tr" if "ı" in m else "en", m, e, behavior="publish preview vs production.", tags=["publish_site", "vs:environment"])
    # analyze types
    for m, at in [("Sesli okuyucu kullananlar sitemizi okuyabiliyor mu?", "accessibility"),
                  ("Arama motorlarında daha üstte çıkmak için ne eksik?", "seo"),
                  ("Sayfalar geç yükleniyor galiba, ölçer misin?", "performance"),
                  ("Telefonumdan girince menü bozuk görünüyor, kontrol eder misin?", "mobile")]:
        add("B", "medium", "tr", m, exp_call(call("analyze_site", analysis_type=at)),
            behavior="Indirect phrasing → correct analysis_type.", tags=["analyze_site", "vs:analysis_type"])


# ---------------------------------------------------------------------------
# C — argument extraction
# ---------------------------------------------------------------------------
def gen_C():
    for i in range(16):
        tr, en, st = pick(PERSONAS)
        style = pick(list(STYLES))
        lang = pick(list(LANG_WORD))
        name = pick(NAMES_BY_TYPE[st])
        m = (f"{tr_cap(tr)}. Adı \"{name}\" olsun, {STYLES[style][0]} bir tasarım istiyorum, "
             f"içerik {LANG_WORD[lang][0]} olacak.")
        add("C", "medium" if i % 2 else "hard", "tr", m,
            exp_call(call("create_site", site_type=st, style=style, language=lang, site_name=name)),
            context=dict(NO_SITE_CTX), behavior="Extract all 4 closed/span args.", tags=["create_site", "args:4"],
            free_args=["create_site.description"])
    for i in range(6):
        tr, en, st = pick(PERSONAS)
        style = pick(list(STYLES)); name = pick(NAMES_BY_TYPE[st])
        m = f"{en[0].upper() + en[1:]}. Call it '{name}', {STYLES[style][1]} look, in English."
        add("C", "medium", "en", m, exp_call(call("create_site", site_type=st, style=style, language="en", site_name=name)),
            context=dict(NO_SITE_CTX), behavior="Extract args.", tags=["create_site", "args:4"], free_args=["create_site.description"])
    for i, hx in enumerate(HEXES):
        pal = pick(["warm", "cool", "dark", "pastel"]); fp = pick(list(FONTS))
        m = f"Ana renk {hx} olsun, paleti {PALETTES[pal][0]} çek ve {FONTS[fp][0]} geç."
        add("C", "hard", "tr", m, exp_call(call("update_theme", palette=pal, font_pair=fp, primary_color=hx)),
            behavior="Three theme args incl. hex span.", tags=["update_theme", "args:3"])
    for i in range(10):
        ct = pick(["blog_post", "product_description", "about_text", "service_description"])
        tone = pick(list(TONES)); ln = pick(list(LENGTHS)); lang = pick(["tr", "en", "de"])
        m = f"{LANG_WORD[lang][0]} dilinde, {TONES[tone][0]} bir tonla, {LENGTHS[ln][0]} {CONTENT[ct][0]}."
        if ct == "blog_post":
            m = f"{tr_cap(pick(TOPICS_TR))} konusunda " + m[0].lower() + m[1:]
        add("C", "hard", "tr", m, exp_call(call("generate_content", content_type=ct, tone=tone, length=ln, language=lang)),
            behavior="4 closed args for copy generation.", tags=["generate_content", "args:4"], free_args=["generate_content.topic"])
    for i in range(8):
        tgt = pick(["hero", "gallery", "blog_cover", "logo"]); sty = pick(list(IMG_STYLE))
        ar = {"hero": "16:9", "gallery": "4:3", "blog_cover": "16:9", "logo": "1:1"}[tgt]
        m = f"{tr_cap(IMG_TARGET[tgt][0])} için {IMG_STYLE[sty][0]} tarzında, {ar} oranında bir görsel oluştur: deniz kenarında sabah kahvesi."
        add("C", "medium", "tr", m, exp_call(call("generate_image", target=tgt, image_style=sty, aspect_ratio=ar)),
            behavior="Image args incl. aspect ratio span-like closed value.", tags=["generate_image", "args:3"],
            free_args=["generate_image.prompt"])
    for i, d in enumerate(DOMAINS[:8]):
        m = pick([f"Şu adresi bağlayalım: {d} — zaten bende kayıtlı.",
                  f"Hey, could you hook up {d}? I registered it last year.",
                  f"www.{d} adını satın al lütfen."])
        a = "purchase" if "satın" in m else "connect_existing"
        add("C", "medium", "en" if m.startswith("Hey") else "tr", m, exp_call(call("manage_domain", action=a, domain=d)),
            behavior="Extract domain span (strip www.) + action.", tags=["manage_domain", "span"])
    for sec, (tr, en) in SECTIONS.items():
        page = pick(["about", "services", "pricing", "contact"])
        ct = pick(["layout", "text", "image", "color"])
        verb = {"layout": "düzenini iki sütunlu yap", "text": "yazısını daha kısa yap", "image": "görselini değiştir", "color": "rengini maviye çevir"}[ct]
        m = f"{tr_cap(PAGES[page][0][0])} içinde, {tr} bölümünün {verb}."
        add("C", "hard", "tr", m, exp_call(call("edit_section", page=page, section=sec, change_type=ct)),
            behavior="Page ≠ current page; extract page + section + change_type.", tags=["edit_section", "args:3"],
            free_args=["edit_section.instruction"])


# ---------------------------------------------------------------------------
# D — multi-tool in one turn
# ---------------------------------------------------------------------------
def gen_D():
    combos = [
        ("Siteye SSS sayfası ekle, sonra da yayına al.", [call("add_page", page_type="faq"), call("publish_site", environment="production")], True),
        ("Logo üret ve tüm renkleri pastel yap.", [call("update_theme", palette="pastel"), call("generate_image", target="logo")], False),
        ("Fiyat sayfası ekle ve ekip sayfası da olsun.", [call("add_page", page_type="pricing"), call("add_page", page_type="team")], False),
        ("SEO analizi yap ve mobil uyumluluğu da kontrol et.", [call("analyze_site", analysis_type="seo"), call("analyze_site", analysis_type="mobile")], False),
        ("Add a gallery page and generate three gallery images for it.", [call("add_page", page_type="gallery"), call("generate_image", target="gallery")], True),
        ("Blog sayfası ekle ve ilk yazı olarak ekşi maya üzerine bir yazı hazırla.", [call("add_page", page_type="blog_index"), call("generate_content", content_type="blog_post")], True),
        ("koc-merve.com'u bağla ve siteyi canlıya al.", [call("manage_domain", action="connect_existing", domain="koc-merve.com"), call("publish_site", environment="production")], True),
        ("Switch to a dark theme and remove the testimonials block.", [call("update_theme", palette="dark"), call("edit_section", section="testimonials", change_type="remove")], False),
        ("Hakkımızda metnini yeniden yaz ve o sayfaya bir kapak görseli üret.", [call("generate_content", content_type="about_text"), call("generate_image", target="hero")], False),
        ("Footer'ı kaldır, header'ın düzenini de ortalanmış yap.", [call("edit_section", section="footer", change_type="remove"), call("edit_section", section="header", change_type="layout")], False),
    ]
    page_pairs = list(PAGES)
    for m, calls, order in combos:
        add("D", "medium", "en" if m[0] in "AS" and " ve " not in m else "tr", m, exp_call(*calls, order=order),
            behavior="Two independent intents in one message → both tools" + (" in dependency order." if order else "."),
            tags=["multi:2"])
    for i in range(22):
        a, b = R.sample(page_pairs, 2)
        m = f"{tr_cap(PAGES[a][0][0])} ve {PAGES[b][0][0]} ekleyelim."
        add("D", "easy", "tr", m, exp_call(call("add_page", page_type=a), call("add_page", page_type=b)),
            behavior="Same tool twice with different args.", tags=["multi:2", "same_tool"])
    for i in range(14):
        pal = pick(list(PALETTES)); pt = pick(page_pairs); env = pick(["preview", "production"])
        envtxt = {"preview": "önizleme linki gönder", "production": "yayına al"}[env]
        m = f"Renkleri {PALETTES[pal][0]} çevir, {PAGES[pt][0][0]} ekle ve {envtxt}."
        add("D", "hard", "tr", m,
            exp_call(call("add_page", page_type=pt), call("update_theme", palette=pal), call("publish_site", environment=env), order=True),
            behavior="Three tools; publish must come last (precedence).", tags=["multi:3"])
    for i in range(12):
        at = pick(["seo", "performance", "accessibility"]); tgt = pick(["hero", "logo", "blog_cover"])
        m = f"Run a {at} check and also generate a new {IMG_TARGET[tgt][1]}."
        add("D", "medium", "en", m, exp_call(call("generate_image", target=tgt), call("analyze_site", analysis_type=at)),
            behavior="Two unrelated tools.", tags=["multi:2"], free_args=["generate_image.prompt"])


# ---------------------------------------------------------------------------
# E — sequential (next tool depends on previous result)
# ---------------------------------------------------------------------------
def gen_E():
    for d in DOMAINS:
        m = f"{d} boştaysa hemen satın al."
        add("E", "medium", "tr", m, S(
            (None, exp_call(call("manage_domain", action="check_availability", domain=d))),
            (obs("manage_domain", available=True, price_try=489), exp_call(call("manage_domain", action="purchase", domain=d))),
            (obs("manage_domain", purchased=True), exp_none()),
        ), behavior="check → (available) purchase → stop and report.", tags=["seq", "conditional"])
        add("E", "hard", "tr", m, S(
            (None, exp_call(call("manage_domain", action="check_availability", domain=d))),
            (obs("manage_domain", available=False, suggestions=[d.replace(".", "-tr."), "get" + d]), exp_clarify("domain")),
        ), behavior="check → taken → do NOT buy a suggestion on your own; ask the user.", tags=["seq", "conditional", "negative_branch"])
    for at, fix in [("seo", call("generate_content", content_type="seo_meta")),
                    ("accessibility", call("edit_section", section="hero", change_type="text")),
                    ("performance", call("edit_section", section="gallery", change_type="image"))]:
        for lang, m in [("tr", f"{at.upper()} analizi yap, çıkan en kritik sorunu da düzelt."),
                        ("en", f"Run a {at} audit and fix the most critical issue it finds.")]:
            issue = {"seo": {"issue": "missing_meta_descriptions", "pages": ["home", "about"]},
                     "accessibility": {"issue": "hero_text_low_contrast", "section": "hero"},
                     "performance": {"issue": "gallery_images_too_large", "section": "gallery"}}[at]
            add("E", "hard", lang, m, S(
                (None, exp_call(call("analyze_site", analysis_type=at))),
                (obs("analyze_site", critical=issue), exp_call(fix)),
                (obs(fix["tool"]), exp_none()),
            ), behavior="Audit result determines which fixing tool and which args.", tags=["seq", "result_dependent"])
    for env_first in range(10):
        pt = pick(list(PAGES))
        m = f"{tr_cap(PAGES[pt][0][0])} ekle, bana önizleme linkini at; ben onaylarsam yayına alırsın."
        add("E", "medium", "tr", m, S(
            (None, exp_call(call("add_page", page_type=pt))),
            (obs("add_page", page="/" + pt), exp_call(call("publish_site", environment="preview"))),
            (obs("publish_site", url="https://preview.118.studio/x7k2"), exp_none()),
        ), behavior="Must stop after preview and wait for user approval; must NOT publish to production.",
            tags=["seq", "human_gate"])
    for i in range(8):
        tr, en, st = pick(PERSONAS)
        m = f"{tr_cap(tr)}. Siteyi kur, sonra bir iletişim sayfası ekle."
        add("E", "medium", "tr", m, S(
            (None, exp_call(call("create_site", site_type=st))),
            (obs("create_site", site_id="site_new1"), exp_call(call("add_page", page_type="contact"))),
            (obs("add_page"), exp_none()),
        ), context=dict(NO_SITE_CTX), behavior="add_page needs site_id produced by create_site.", tags=["seq", "dependency"],
            free_args=["create_site.description"])
    for i in range(8):
        d = pick(DOMAINS)
        m = f"Check whether {d} is free; if it is, connect... no wait, just tell me the price."
        add("E", "hard", "en", m, S(
            (None, exp_call(call("manage_domain", action="check_availability", domain=d))),
            (obs("manage_domain", available=True, price_try=489), exp_none()),
        ), behavior="User revoked the follow-up action; after the check, only report the price.", tags=["seq", "self_correction"])

    for d in DOMAINS:
        add("E", "medium", "en", f"If {d} is available, register it for me.", S(
            (None, exp_call(call("manage_domain", action="check_availability", domain=d))),
            (obs("manage_domain", available=True, price_usd=14), exp_call(call("manage_domain", action="purchase", domain=d))),
            (obs("manage_domain", purchased=True), exp_none()),
        ), behavior="check → available → purchase → stop.", tags=["seq", "conditional"])
    for pal, (tr, en) in PALETTES.items():
        add("E", "medium", "tr", f"Önce mobil uyumluluğu kontrol et, sorun yoksa renkleri {tr} çevir.", S(
            (None, exp_call(call("analyze_site", analysis_type="mobile"))),
            (obs("analyze_site", critical=None, score=93), exp_call(call("update_theme", palette=pal))),
            (obs("update_theme"), exp_none()),
        ), behavior="Gate passes → theme change.", tags=["seq", "conditional"])


# ---------------------------------------------------------------------------
# F — no tool
# ---------------------------------------------------------------------------
def gen_F():
    msgs = [
        ("tr", "easy", "Merhaba!"), ("tr", "easy", "Teşekkürler, harika oldu."), ("en", "easy", "Thanks a lot!"),
        ("tr", "easy", "Günaydın, nasılsın?"), ("en", "easy", "Hello there"),
        ("tr", "medium", "Bir kafe sitesinde hangi renkler daha çok güven verir sence?"),
        ("tr", "medium", "Portfolyo sitesinde kaç proje göstermek mantıklı?"),
        ("en", "medium", "What's the difference between SEO and SEM?"),
        ("tr", "medium", "Alan adı bağlamak genelde ne kadar sürer?"),
        ("tr", "medium", "118.studio'nun ücretli planında neler var?"),
        ("en", "medium", "Do you support multilingual sites?"),
        ("tr", "medium", "Bugün İstanbul'da hava nasıl?"),
        ("en", "medium", "Can you recommend a good laptop for design work?"),
        ("tr", "hard", "Siteyi yayına almayı düşünüyorum ama henüz hazır değilim, sonra konuşuruz."),
        ("tr", "hard", "Logo için yapay zeka mı kullanayım yoksa bir tasarımcıyla mı çalışayım?"),
        ("en", "hard", "I might want a pricing page later, what do people usually put on it?"),
        ("tr", "hard", "Dün eklediğin galeri sayfası çok güzel olmuş."),
        ("tr", "hard", "Koyu tema mı açık tema mı daha iyi dönüşüm getirir?"),
        ("en", "hard", "Is it a good idea to publish before I have a domain?"),
        ("tr", "hard", "Domain almadan önce markamın adını netleştirmem lazım galiba."),
        ("tr", "medium", "Bana web tasarımı hakkında bir kitap önerir misin?"),
        ("tr", "medium", "Hero bölümü ne demek?"), ("en", "medium", "What does a favicon do?"),
        ("tr", "medium", "Önizleme linki kaç gün geçerli?"),
        ("tr", "hard", "SEO analizi ne işe yarıyor, yaptırmaya değer mi?"),
        ("en", "hard", "Before you change anything — how do themes work here?"),
        ("tr", "hard", "Hayır hayır, hiçbir şeyi değiştirme, sadece bakıyordum."),
        ("en", "hard", "Never mind, cancel that."),
        ("tr", "medium", "Tamam, anladım."), ("tr", "easy", "👍"),
        ("tr", "hard", "Rakibimin sitesi bizimkinden daha mı iyi sence?"),
        ("tr", "medium", "Bir e-ticaret sitesinde KVKK metni zorunlu mu?"),
        ("en", "medium", "How many visitors did I get yesterday?"),
        ("tr", "hard", "Bu akşam yayına alırız belki, şimdilik dursun."),
        ("tr", "medium", "Python ile web scraping nasıl yapılır?"),
        ("en", "hard", "Write me a poem about the sea."),
        ("tr", "hard", "Sitemi silmek istiyorum."),  # no delete tool → no_tool, explain
        ("en", "hard", "Can you translate my entire site into Japanese?"),  # unsupported language → no tool
        ("tr", "hard", "Instagram hesabımı siteye bağlayabilir misin?"),  # no tool for it
        ("en", "medium", "Who made you?"),
    ]
    for lang, diff, m in msgs:
        unsupported = any(k in m for k in ["silmek", "Japanese", "Instagram"])
        add("F", diff, lang, m, exp_none(),
            behavior=("Requested action has no matching tool → explain, do not force a near tool."
                      if unsupported else "Conversation / advice / platform question → answer in text, no tool."),
            tags=["no_tool"] + (["unsupported_action"] if unsupported else []))
    # same messages on a no-site context for variety
    for lang, diff, m in R.sample(msgs, 22):
        add("F", diff, lang, m, exp_none(), context=dict(NO_SITE_CTX),
            behavior="No tool even for a new user.", tags=["no_tool", "no_site"])


# ---------------------------------------------------------------------------
# G — ambiguous / missing info
# ---------------------------------------------------------------------------
def gen_G():
    items = [
        ("tr", "easy", "Bir site yapalım.", exp_clarify("site_type"), NO_SITE_CTX, "create_site without site_type."),
        ("en", "easy", "Make me a website.", exp_clarify("site_type"), NO_SITE_CTX, "create_site without site_type."),
        ("tr", "easy", "Bir domain al.", exp_clarify("domain"), None, "manage_domain without domain."),
        ("en", "easy", "Connect my domain.", exp_clarify("domain"), None, "connect without domain."),
        ("tr", "easy", "Bir sayfa daha ekle.", exp_clarify("page_type"), None, "add_page without page_type."),
        ("tr", "medium", "Rengi değiştir.", exp_clarify("scope", between=["edit_section", "update_theme"]), None, "Section or whole site?"),
        ("en", "medium", "Change the colors.", exp_clarify("scope", between=["edit_section", "update_theme"]), None, "Scope unclear."),
        ("tr", "medium", "Bunu düzelt.", exp_clarify("intent"), None, "No referent."),
        ("tr", "medium", "Daha güzel yap.", exp_clarify("intent"), None, "Vague."),
        ("en", "medium", "Make it pop.", exp_clarify("intent"), None, "Vague."),
        ("tr", "medium", "Görsel lazım.", exp_clarify("target", between=["generate_image", "edit_section"]), None, "Which image, generate or upload?"),
        ("tr", "medium", "Yazıları güncelle.", exp_clarify("intent", between=["edit_section", "generate_content"]), None, "Which text?"),
        ("tr", "hard", "Bir şeyler yapalım işte, sen bilirsin.", exp_clarify("intent"), None, "Delegation with no goal."),
        ("en", "hard", "Do the thing we talked about.", exp_clarify("intent"), None, "Refers to unknown prior context."),
        ("tr", "hard", "Kontrol et.", exp_clarify("analysis_type"), None, "Check what?"),
        ("tr", "hard", "Yayınla mı önizleme mi bilmiyorum, sen ne dersin?", exp_none(), None, "User asks for advice → answer, no tool."),
        ("tr", "hard", "Şu siteyi bağla.", exp_clarify("domain"), None, "Domain missing."),
        ("en", "hard", "Buy it.", exp_clarify("domain"), None, "No referent for 'it'."),
        ("tr", "medium", "Font değişsin.", exp_clarify("font_pair"), None, "Which font family? (update_theme requires a choice)"),
    ]
    for lang, diff, m, e, c, why in items:
        add("G", diff, lang, m, e, context=dict(c) if c else ctx(), behavior=f"Must ask before acting: {why}", tags=["ambiguous"])
    # context-resolvable ambiguity (selected section in editor) → should NOT clarify
    for sec, (tr, en) in SECTIONS.items():
        add("G", "hard", "tr", "Bunu kaldır.", exp_call(call("edit_section", section=sec, change_type="remove")),
            context=ctx(selected_section=sec), behavior="Referent resolved by context.selected_section → act, no clarification.",
            tags=["ambiguous", "context_resolved"])
        add("G", "hard", "en", "Make this one blue.", exp_call(call("edit_section", section=sec, change_type="color")),
            context=ctx(selected_section=sec), behavior="Referent resolved by selected section.", tags=["ambiguous", "context_resolved"],
            free_args=["edit_section.instruction"])
    # site exists & user says "site yap" → ambiguous: new site or edit?
    for m in ["Bir site yap, kafe için.", "Kafe için site lazım."]:
        add("G", "hard", "tr", m, exp_call(call("create_site", site_type="restaurant")), context=dict(NO_SITE_CTX),
            behavior="'kafe' maps to restaurant → enough info.", tags=["ambiguous", "mapping"], free_args=["create_site.description"])
    for d in DOMAINS[:10]:
        add("G", "medium", "tr", f"{d} ile ilgili bir şey yap.",
            exp_clarify("action"), behavior="Domain given but action unknown (check/buy/connect).", tags=["ambiguous", "missing_action"])
    for pt in list(PAGES)[:6]:
        add("G", "medium", "tr", f"{tr_cap(PAGES[pt][0][0])} ile ilgili bir sorun var.",
            exp_clarify("intent"), behavior="Problem mentioned but not what to do.", tags=["ambiguous"])


# ---------------------------------------------------------------------------
# H — error handling & recovery
# ---------------------------------------------------------------------------
def gen_H():
    for d in DOMAINS:
        add("H", "hard", "tr", f"{d} adresini satın al.", S(
            (None, exp_call(call("manage_domain", action="purchase", domain=d))),
            (obs("manage_domain", "error", code="payment_failed", message="Card declined"), exp_none()),
        ), behavior="Paid action failed → NEVER auto-retry; inform the user (no tool).", tags=["error", "non_idempotent"])
        add("H", "hard", "tr", f"{d} alan adını al.", S(
            (None, exp_call(call("manage_domain", action="purchase", domain=d))),
            (obs("manage_domain", "error", code="domain_taken", suggestions=["get" + d]), exp_clarify("domain")),
        ), behavior="Domain taken → ask which alternative, don't pick one.", tags=["error"])
    for at in ["seo", "performance", "mobile", "accessibility"]:
        for lang, m in [("tr", f"{at} analizi yapar mısın?"), ("en", f"Run a {at} audit.")]:
            add("H", "medium", lang, m, S(
                (None, exp_call(call("analyze_site", analysis_type=at))),
                (obs("analyze_site", "error", code="timeout", retryable=True), exp_call(call("analyze_site", analysis_type=at))),
                (obs("analyze_site", score=71), exp_none()),
            ), behavior="Transient error on read-only tool → retry once with same args, then report.", tags=["error", "retry"])
    for i in range(6):
        add("H", "medium", "tr", "Önizleme linki oluştur.", S(
            (None, exp_call(call("publish_site", environment="preview"))),
            (obs("publish_site", "error", code="rate_limited", retryable=True), exp_call(call("publish_site", environment="preview"))),
            (obs("publish_site", url="https://preview.118.studio/a1"), exp_none()),
        ), context=ctx(current_page=pick(["home", "about"])), behavior="Retry preview after rate limit.", tags=["error", "retry"])
    for hx in ["#12GG45", "#FFF", "#ZZZ999"]:
        add("H", "medium", "tr", f"Ana rengi {hx} yap.", S(
            (None, exp_call(call("update_theme", primary_color=hx))),
            (obs("update_theme", "error", code="validation_error", field="primary_color"), exp_clarify("primary_color")),
        ), behavior="Invalid value from user → ask for a valid one, don't invent.", tags=["error", "validation"])
    for i in range(5):
        add("H", "hard", "en", "Generate a hero image of my competitor's logo on fire.", S(
            (None, exp_call(call("generate_image", target="hero"))),
            (obs("generate_image", "error", code="content_policy_violation"), exp_none()),
        ), context=ctx(current_page=pick(["home", "services"])), behavior="Policy refusal → explain, no retry, no alternative tool.",
            tags=["error", "policy"], free_args=["generate_image.prompt"])
    for i in range(6):
        pt = pick(list(PAGES))
        add("H", "hard", "tr", f"{tr_cap(PAGES[pt][0][0])} ekle.", S(
            (None, exp_call(call("add_page", page_type=pt))),
            (obs("add_page", "error", code="site_not_found"), exp_clarify("site")),
        ), behavior="Context site missing → ask (maybe create new site?), don't silently create.", tags=["error", "state"])
    for i in range(5):
        add("H", "hard", "tr", "Siteyi yayına al.", S(
            (None, exp_call(call("publish_site", environment="production"))),
            (obs("publish_site", "error", code="dns_not_propagated", retry_after_minutes=30), exp_none()),
        ), context=ctx(domain=pick(DOMAINS)), behavior="Non-transient wait → inform user, don't hammer the API.", tags=["error", "wait"])
    # missing param discovered by the tool
    for d in DOMAINS[:4]:
        add("H", "medium", "en", f"Connect {d}", S(
            (None, exp_call(call("manage_domain", action="connect_existing", domain=d))),
            (obs("manage_domain", "error", code="dns_verification_required", txt_record="118-verify=ab12"), exp_none()),
        ), behavior="User must add a DNS record → explain the step, no tool.", tags=["error", "user_action_required"])


# ---------------------------------------------------------------------------
# I — complex workflows
# ---------------------------------------------------------------------------
def gen_I():
    for tr, en, st in PERSONAS:
        d = pick(DOMAINS)
        m = (f"{tr_cap(tr)}. Siteyi kur, bir logo üret, galeri sayfası ekle, "
             f"{d} boştaysa al ve sonra yayına al.")
        add("I", "hard", "tr", m, S(
            (None, exp_call(call("create_site", site_type=st))),
            (obs("create_site", site_id="site_new1"), exp_call(call("add_page", page_type="gallery"), call("generate_image", target="logo"))),
            ([obs("add_page", page="/gallery"), obs("generate_image", url="https://cdn.118.studio/logo.png")], exp_call(call("manage_domain", action="check_availability", domain=d))),
            (obs("manage_domain", available=True, price_try=489), exp_call(call("manage_domain", action="purchase", domain=d))),
            (obs("manage_domain", purchased=True), exp_call(call("publish_site", environment="production"))),
            (obs("publish_site", url=f"https://{d}"), exp_none()),
        ), context=dict(NO_SITE_CTX), behavior="Full flow honouring dependencies: site → (page ∥ logo) → check → buy → publish → stop.",
            tags=["workflow", "steps:6"], free_args=["create_site.description", "generate_image.prompt"])
    for i in range(12):
        tr, en, st = pick(PERSONAS)
        pal = pick(list(PALETTES))
        m = (f"{en[0].upper() + en[1:]}. Build it with {PALETTES[pal][1]}, write an about text, "
             f"then send me a preview link.")
        add("I", "hard", "en", m, S(
            (None, exp_call(call("create_site", site_type=st))),
            (obs("create_site", site_id="site_new2"), exp_call(call("update_theme", palette=pal), call("generate_content", content_type="about_text"))),
            ([obs("update_theme"), obs("generate_content", draft_id="d1")], exp_call(call("publish_site", environment="preview"))),
            (obs("publish_site", url="https://preview.118.studio/q9"), exp_none()),
        ), context=dict(NO_SITE_CTX), behavior="Theme+copy can run in parallel after creation; preview last; never production.",
            tags=["workflow", "steps:4"], free_args=["create_site.description", "generate_content.topic"])
    for i in range(12):
        d = pick(DOMAINS)
        m = f"Sitenin SEO ve hız analizini yap, kritik bir sorun yoksa {d}'u bağla ve canlıya al."
        crit = R.random() < 0.5
        steps = [
            (None, exp_call(call("analyze_site", analysis_type="seo"), call("analyze_site", analysis_type="performance"))),
        ]
        if crit:
            steps.append(([obs("analyze_site", analysis_type="seo", critical={"issue": "broken_links", "count": 14}), obs("analyze_site", analysis_type="performance", score=81)], exp_none()))
            beh = "Condition fails (critical issue) → stop and report; do NOT connect or publish."
        else:
            steps += [
                ([obs("analyze_site", analysis_type="seo", critical=None, score=88), obs("analyze_site", analysis_type="performance", critical=None, score=90)], exp_call(call("manage_domain", action="connect_existing", domain=d))),
                (obs("manage_domain", connected=True), exp_call(call("publish_site", environment="production"))),
                (obs("publish_site", url=f"https://{d}"), exp_none()),
            ]
            beh = "Condition passes → connect → publish → stop."
        add("I", "hard", "tr", m, S(*steps), behavior=beh, tags=["workflow", "conditional", "branch:" + ("stop" if crit else "go")])
    for i in range(10):
        pt = pick(list(PAGES)); tone = pick(list(TONES))
        m = f"{tr_cap(PAGES[pt][0][0])} ekle, içeriğini {TONES[tone][0]} bir dille sen yaz, kapak görseli üret; yayına alma, ben bakacağım."
        add("I", "hard", "tr", m, S(
            (None, exp_call(call("add_page", page_type=pt))),
            (obs("add_page", page="/" + pt), exp_call(call("generate_content", content_type="service_description" if pt == "services" else "about_text" if pt == "about" else "blog_post", tone=tone), call("generate_image", target="hero"))),
            ([obs("generate_content", draft_id="d7"), obs("generate_image", url="https://cdn.118.studio/c.png")], exp_none()),
        ), behavior="Explicit 'don't publish' must be honoured; finish without publish_site.", tags=["workflow", "negative_instruction"],
            free_args=["generate_content.topic", "generate_image.prompt"])


# ---------------------------------------------------------------------------
# TR — Turkish counterparts of the English-only cases, plus Turkish-specific difficulty.
# 118.studio is a Turkish product: the main dataset is Turkish; English cases are kept as a
# separate control set (dataset_en_control.jsonl) to measure the "Turkish penalty" of each model.
# ---------------------------------------------------------------------------
def gen_TR():
    # B — English-only similar-tool cases, in Turkish
    for m, e, tags, fa in [
        ("Hakkımızda sayfasına sıfırdan, taze bir metin yazar mısın?", exp_call(call("generate_content", content_type="about_text")), ["vs:edit_section"], ["generate_content.topic"]),
        ("Galerideki fotoğrafları benim yüklediklerimle değiştir.", exp_call(call("edit_section", section="gallery", change_type="image")), ["vs:image"], []),
        ("Hero için yapay zekayla illüstrasyon tarzında bir görsel çiz.", exp_call(call("generate_image", target="hero", image_style="illustration")), ["vs:image"], ["generate_image.prompt"]),
        ("Yayına alacağız ama önce patronuma kontrol linki gönder.", exp_call(call("publish_site", environment="preview")), ["vs:environment"], []),
        ("Bundan tamamen ayrı, yeni bir online mağaza açmak istiyorum.", exp_call(call("create_site", site_type="ecommerce")), ["vs:add_page"], ["create_site.description"]),
    ]:
        add("B", "hard", "tr", m, e, behavior="Turkish version of a near-tool distinction.", tags=["tr_counterpart"] + tags, free_args=fa)
    # C — create_site without quotes, Turkish suffixes on span values
    for d in DOMAINS[:10]:
        for m, a in [(f"{d}'u zaten aldım, siteye bağlar mısın?", "connect_existing"),
                     (f"{d} adresini bana satın alır mısın?", "purchase")]:
            add("C", "hard", "tr", m, exp_call(call("manage_domain", action=a, domain=d)),
                behavior="Domain carries a Turkish case suffix ('u/'yi) that must be stripped from the span.",
                tags=["manage_domain", "span", "tr_suffix"])
    for hx in HEXES:
        add("C", "hard", "tr", f"Ana rengi {hx}'e çek, sitenin geneli böyle olsun.",
            exp_call(call("update_theme", primary_color=hx)), behavior="Hex + Turkish dative suffix.", tags=["update_theme", "tr_suffix"])
    # D — English-only multi-tool cases, in Turkish
    for i in range(12):
        at = pick(["seo", "performance", "accessibility"]); tgt = pick(["hero", "logo", "blog_cover"])
        atw = {"seo": "SEO", "performance": "hız", "accessibility": "erişilebilirlik"}[at]
        add("D", "medium", "tr", f"Bir {atw} kontrolü yap, bir de yeni bir {IMG_TARGET[tgt][0]} üret.",
            exp_call(call("generate_image", target=tgt), call("analyze_site", analysis_type=at)),
            behavior="Two unrelated tools.", tags=["multi:2", "tr_counterpart"], free_args=["generate_image.prompt"])
    add("D", "medium", "tr", "Galeri sayfası ekle, içine koymak için de galeri görselleri üret.",
        exp_call(call("add_page", page_type="gallery"), call("generate_image", target="gallery"), order=True),
        behavior="Page before images.", tags=["multi:2", "tr_counterpart"], free_args=["generate_image.prompt"])
    add("D", "medium", "tr", "Koyu temaya geç ve yorumlar bloğunu kaldır.",
        exp_call(call("update_theme", palette="dark"), call("edit_section", section="testimonials", change_type="remove")),
        behavior="Global + local change in one message.", tags=["multi:2", "tr_counterpart"])
    # E — English-only sequential cases, in Turkish
    for d in DOMAINS:
        add("E", "medium", "tr", f"{d} müsaitse benim adıma kaydet.", S(
            (None, exp_call(call("manage_domain", action="check_availability", domain=d))),
            (obs("manage_domain", available=True, price_try=489), exp_call(call("manage_domain", action="purchase", domain=d))),
            (obs("manage_domain", purchased=True), exp_none()),
        ), behavior="check → available → purchase → stop.", tags=["seq", "conditional", "tr_counterpart"])
    for i in range(8):
        d = pick(DOMAINS)
        add("E", "hard", "tr", f"{d} boş mu bak, boşsa bağla... yok dur, sadece fiyatını söyle.", S(
            (None, exp_call(call("manage_domain", action="check_availability", domain=d))),
            (obs("manage_domain", available=True, price_try=489), exp_none()),
        ), behavior="User revoked the follow-up in the same message; only report.", tags=["seq", "self_correction", "tr_counterpart"])
    # G — English-only ambiguous cases, in Turkish
    for m, e, why in [
        ("Biraz canlandır şunu.", exp_clarify("intent"), "Vague."),
        ("Konuştuğumuz şeyi yap.", exp_clarify("intent"), "Unknown prior context."),
        ("Al gitsin.", exp_clarify("domain"), "No referent."),
        ("Domainimi bağla.", exp_clarify("domain"), "connect without domain."),
        ("Renkleri değiştirelim.", exp_clarify("scope", between=["edit_section", "update_theme"]), "Scope unclear."),
        ("Şu mavi olsun.", exp_clarify("intent"), "'Şu' has no referent and nothing is selected."),
    ]:
        add("G", "medium", "tr", m, e, behavior=f"Must ask before acting: {why}", tags=["ambiguous", "tr_counterpart"])
    for sec in SECTIONS:
        add("G", "hard", "tr", "Şunu maviye boya.", exp_call(call("edit_section", section=sec, change_type="color")),
            context=ctx(selected_section=sec), behavior="'Şunu' resolved by selected section.", tags=["ambiguous", "context_resolved"],
            free_args=["edit_section.instruction"])
    # H — English-only error cases, in Turkish
    for i in range(5):
        add("H", "hard", "tr", "Rakibimin logosunu yanarken gösteren bir hero görseli üret.", S(
            (None, exp_call(call("generate_image", target="hero"))),
            (obs("generate_image", "error", code="content_policy_violation"), exp_none()),
        ), context=ctx(current_page=["home", "services", "about", "pricing", "contact"][i]),
            behavior="Policy refusal → explain, no retry.", tags=["error", "policy", "tr_counterpart"], free_args=["generate_image.prompt"])
    for d in DOMAINS[:6]:
        add("H", "medium", "tr", f"{d} adresini bağla.", S(
            (None, exp_call(call("manage_domain", action="connect_existing", domain=d))),
            (obs("manage_domain", "error", code="dns_verification_required", txt_record="118-verify=ab12"), exp_none()),
        ), behavior="User must add a DNS TXT record → explain, no tool.", tags=["error", "user_action_required", "tr_counterpart"])
    # I — English-only workflows, in Turkish
    for i in range(12):
        tr, en, st = pick(PERSONAS)
        pal = pick(list(PALETTES))
        m = f"{tr_cap(tr)}. Siteyi {PALETTES[pal][0]} uygun kur, bir hakkımızda metni yaz, sonra bana önizleme linki gönder."
        add("I", "hard", "tr", m, S(
            (None, exp_call(call("create_site", site_type=st))),
            (obs("create_site", site_id="site_new2"), exp_call(call("update_theme", palette=pal), call("generate_content", content_type="about_text"))),
            ([obs("update_theme"), obs("generate_content", draft_id="d1")], exp_call(call("publish_site", environment="preview"))),
            (obs("publish_site", url="https://preview.118.studio/q9"), exp_none()),
        ), context=dict(NO_SITE_CTX), behavior="Theme+copy in parallel after creation; preview last; never production.",
            tags=["workflow", "steps:4", "tr_counterpart"], free_args=["create_site.description", "generate_content.topic"])

    # Code-mixed Turkish (how Turkish users actually talk about the web)
    for m, e, fa in [
        ("Landing page'e bir pricing section ekle.", exp_call(call("add_page", page_type="pricing")), []),
        ("Siteyi deploy et artık.", exp_call(call("publish_site", environment="production")), []),
        ("Dark mode'a geçelim, bütün site.", exp_call(call("update_theme", palette="dark")), []),
        ("Hero'daki CTA butonunun text'ini 'Hemen Dene' yap.", exp_call(call("edit_section", section="cta", change_type="text")), ["edit_section.instruction"]),
        ("Footer'daki sosyal medya ikonlarını remove et.", exp_call(call("edit_section", section="footer", change_type="remove")), []),
        ("Bir staging linki at, ekip baksın.", exp_call(call("publish_site", environment="preview")), []),
        ("SEO audit'i çalıştır.", exp_call(call("analyze_site", analysis_type="seo")), []),
        ("Lighthouse skoru düşük, performance'a bak.", exp_call(call("analyze_site", analysis_type="performance")), []),
        ("Ürünler için copy yaz, satış odaklı olsun.", exp_call(call("generate_content", content_type="product_description")), ["generate_content.topic"]),
        ("Favicon generate et.", exp_call(call("generate_image", target="favicon")), ["generate_image.prompt"]),
        ("Bir FAQ page lazım.", exp_call(call("add_page", page_type="faq")), []),
        ("Font'ları serif'e çevir.", exp_call(call("update_theme", font_pair="serif_classic")), []),
        ("Portfolyo sitesi kur bana, minimal olsun.", exp_call(call("create_site", site_type="portfolio", style="minimal")), ["create_site.description"]),
    ]:
        add("A" if len(e["calls"]) == 1 else "D", "medium", "tr", m, e, context=dict(NO_SITE_CTX) if "kur" in m else None,
            behavior="Code-mixed Turkish/English, the way Turkish users talk about web work.", tags=["code_mixed"], free_args=fa)
    # Turkish no-tool / out-of-scope additions
    for m in ["Eyvallah, eline sağlık.", "Kolay gelsin!", "Peki bunu yapmak ne kadar sürer?", "Hangi plan bana uygun olur sence?",
              "E-ticaret sitesinde mesafeli satış sözleşmesi şart mı?", "Sitemi tamamen silmek istiyorum.",
              "Sitemi Rusçaya çevirebilir misin?", "Dün kaç kişi siteye girdi?", "Hızlıca bir CV yazar mısın bana?",
              "Önizleme linkini kaç kişi açabilir?"]:
        add("F", "hard" if any(k in m for k in ["silmek", "Rusça", "kaç kişi siteye", "CV"]) else "medium", "tr", m, exp_none(),
            behavior="No matching tool (chat, advice, unsupported action) → answer in text.", tags=["no_tool", "tr_extra"])


TR_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def asciify(s: str) -> str:
    return s.translate(TR_MAP)


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def add_noisy_variants(records, share=0.22):
    """Turkish users often type without diacritics and in lower case on mobile.
    Variant = same intent, message asciified + lower-cased; span values typed in the message are
    transformed the same way (the user really typed kahveduragi.com.tr)."""
    rng = random.Random(SEED + 1)
    out = []
    for r in records:
        if r["language"] != "tr" or rng.random() > share:
            continue
        msg = r["user_message"]
        noisy = asciify(tr_lower(msg))
        if noisy == msg.lower():
            continue
        v = json.loads(json.dumps(r))
        v["user_message"] = noisy
        v["tags"] = v["tags"] + ["noisy_ascii"]
        v["difficulty"] = {"easy": "medium", "medium": "hard", "hard": "hard"}[r["difficulty"]]
        v["noisy_of"] = r["id"]
        for st in v["steps"]:
            for c in st["expected"]["calls"]:
                for k, val in list(c["args"].items()):
                    if isinstance(val, str) and val in msg and k in ("domain", "site_name", "title", "primary_color"):
                        c["args"][k] = asciify(tr_lower(val)) if k != "primary_color" else val
        v["success_criteria"]["rules"] = criteria_for(v["steps"])
        out.append(v)
    return out


def main():
    for g in (gen_A, gen_B, gen_C, gen_D, gen_E, gen_F, gen_G, gen_H, gen_I, gen_TR):
        g()
    tr = [r for r in RECORDS if r["language"] == "tr"]
    en = [r for r in RECORDS if r["language"] != "tr"]
    tr += add_noisy_variants(tr)
    from collections import Counter, defaultdict
    for name, recs in (("dataset.jsonl", tr), ("dataset_en_control.jsonl", en)):
        recs.sort(key=lambda r: (r["category"], r.get("noisy_of", ""), r["id"]))
        counters = defaultdict(int)
        id_map = {}
        for r in recs:
            c = r["category"][0]
            counters[c] += 1
            new = f"{'' if name == 'dataset.jsonl' else 'EN-'}{c}-{counters[c]:04d}"
            id_map[r["id"]] = new
            r["id"] = new
        for r in recs:
            if "noisy_of" in r:
                r["noisy_of"] = id_map.get(r["noisy_of"], r["noisy_of"])
        path = OUT.with_name(name)
        with path.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(recs)} records → {path.name}")
        print("  by category:", dict(sorted(Counter(r["category"] for r in recs).items())))
        print("  by difficulty:", dict(Counter(r["difficulty"] for r in recs)))
        print("  tags:", {t: n for t, n in Counter(t for r in recs for t in r["tags"]).items() if t in ("noisy_ascii", "code_mixed", "tr_suffix", "tr_counterpart")})


if __name__ == "__main__":
    main()
