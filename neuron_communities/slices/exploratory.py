"""Exploratory slices (Build A, unscored): agreement, closing delimiters, in-context copying.

Every prompt ends right before the token whose prediction exercises the behavior, and records the expected
continuation (``answer``) and, where there is one, the competing wrong form (``answer_alt``: " an" vs " a", " is" vs
" are", the other gender's adjective), so later analyses can read accuracy and a logit difference. Generated from fixed
word lists and templates with a seed; texts are unique. ``split`` is a seeded 50/50 discovery / held-out assignment per
task (these slices have no ground truth; the held-out half is for testing communities found on the other half).

Tasks (``task`` field)::

  agree_a_an      article before a noun the context fixes (Goodfire's a/an community is the recovery check)
  agree_there     "there is" / "there are" after a question that fixes the number
  agree_sv        English subject-verb number with a prepositional attractor (Linzen et al. 2016 style)
  agree_fr        French predicative adjective gender/number, copied from a first sentence with another noun
  agree_es        Spanish predicative adjective gender/number, same design
  agree_de        German verb number (3sg / 3pl), the verb copied from a first sentence with the other number
  delim_code      closing parentheses of nested Python calls (answer: the closers still owed)
  delim_html      the tag name after "</" in nested HTML
  delim_latex     closing braces of nested LaTeX commands
  copy_words      in-context copying: a random word list repeated; predict the next word of the second copy
"""
from __future__ import annotations

import random

TASKS = ("agree_a_an", "agree_there", "agree_sv", "agree_fr", "agree_es", "agree_de",
         "delim_code", "delim_html", "delim_latex", "copy_words")
DEFAULT_N = {"agree_a_an": 3000, "agree_there": 2000, "agree_sv": 3000, "agree_fr": 2000, "agree_es": 2000,
             "agree_de": 2000, "delim_code": 2000, "delim_html": 2000, "delim_latex": 2000, "copy_words": 3000}

# ---- word lists -------------------------------------------------------------------------------------------------------
AN_NOUNS = ["apple", "orange", "egg", "umbrella", "elephant", "owl", "onion", "island", "igloo", "eagle", "octopus",
            "ant", "axe", "envelope", "oven", "iguana", "ostrich", "acorn", "apron", "anchor", "arrow", "easel", "eel",
            "engine", "eraser", "avocado", "alligator", "armchair", "insect", "ocean", "airplane", "album", "angel",
            "apricot", "olive", "otter", "oyster", "elbow", "emerald", "orchid"]
A_NOUNS = ["banana", "pear", "dog", "cat", "table", "chair", "book", "pencil", "lamp", "cup", "bottle", "shirt", "hat",
           "kite", "ball", "bicycle", "car", "truck", "boat", "drum", "guitar", "piano", "hammer", "ladder", "mirror",
           "pillow", "rabbit", "horse", "tiger", "zebra", "monkey", "frog", "snake", "carrot", "potato", "tomato",
           "cookie", "sandwich", "bucket", "basket", "candle", "clock", "door", "fork", "spoon", "wallet", "ring",
           "lemon", "peach", "cherry", "plum", "mango", "coconut", "walnut"]
TREE_FRUITS = {"apple": "an", "orange": "an", "apricot": "an", "avocado": "an", "olive": "an", "pear": "a",
               "plum": "a", "peach": "a", "cherry": "a", "banana": "a", "mango": "a", "lemon": "a", "coconut": "a",
               "walnut": "a"}
NAMES = ["Anna", "Ben", "Carlos", "Dana", "Elif", "Farah", "George", "Hana", "Ivan", "Julia", "Kofi", "Lena", "Maya",
         "Noah", "Omar", "Priya", "Quinn", "Rosa", "Sam", "Tara"]
PLURALS = {"apple": "apples", "orange": "oranges", "egg": "eggs", "book": "books", "pencil": "pencils", "cup": "cups",
           "bottle": "bottles", "chair": "chairs", "table": "tables", "key": "keys", "box": "boxes", "coin": "coins",
           "cookie": "cookies", "shirt": "shirts", "ball": "balls", "lamp": "lamps", "candle": "candles",
           "spoon": "spoons", "fork": "forks", "letter": "letters", "photo": "photos", "toy": "toys", "shoe": "shoes",
           "glass": "glasses"}
SV_SUBJECTS = {"author": "authors", "key": "keys", "teacher": "teachers", "dog": "dogs", "book": "books", "car": "cars",
               "student": "students", "picture": "pictures", "box": "boxes", "painting": "paintings",
               "manager": "managers", "doctor": "doctors", "label": "labels", "player": "players", "road": "roads",
               "report": "reports"}
SV_ATTRACTORS = {"cabinet": "cabinets", "table": "tables", "window": "windows", "house": "houses", "tree": "trees",
                 "river": "rivers", "boy": "boys", "girl": "girls", "village": "villages", "machine": "machines",
                 "building": "buildings", "company": "companies"}
SV_PREPS = ["of the", "near the", "behind the", "next to the", "for the", "beside the"]
SV_FRAMES = [("{S} {P} {A}", "t0"), ("I think that the {S} {P} {A}", "t1"), ("Yesterday I noticed that the {S} {P} {A}", "t2")]

FR_NOUNS = {"m": ["chat", "chien", "livre", "stylo", "vélo", "jardin", "gâteau", "mur", "sac", "chapeau", "train",
                  "bateau", "crayon", "manteau", "pantalon", "fauteuil"],
            "f": ["maison", "voiture", "table", "chaise", "porte", "fleur", "robe", "pomme", "lampe", "fenêtre",
                  "chemise", "boîte", "tasse", "rue", "valise", "jupe"]}
FR_PLURAL = {"gâteau": "gâteaux", "bateau": "bateaux", "chapeau": "chapeaux", "manteau": "manteaux"}
FR_ADJ = [("noir", "noire", "noirs", "noires"), ("blanc", "blanche", "blancs", "blanches"),
          ("petit", "petite", "petits", "petites"), ("grand", "grande", "grands", "grandes"),
          ("vert", "verte", "verts", "vertes"), ("gris", "grise", "gris", "grises"),
          ("lourd", "lourde", "lourds", "lourdes"), ("bleu", "bleue", "bleus", "bleues"),
          ("joli", "jolie", "jolis", "jolies"), ("froid", "froide", "froids", "froides"),
          ("long", "longue", "longs", "longues"), ("neuf", "neuve", "neufs", "neuves")]
ES_NOUNS = {"m": ["perro", "libro", "coche", "gato", "vaso", "plato", "zapato", "sombrero", "edificio", "jardín",
                  "abrigo", "cuaderno", "barco", "espejo", "reloj", "sillón"],
            "f": ["casa", "mesa", "silla", "puerta", "camisa", "ventana", "botella", "caja", "manzana", "lámpara",
                  "taza", "falda", "maleta", "calle", "flor", "pared"]}
ES_PLURAL = {"jardín": "jardines", "reloj": "relojes", "sillón": "sillones", "flor": "flores", "pared": "paredes"}
ES_ADJ = [("blanco", "blanca"), ("negro", "negra"), ("rojo", "roja"), ("pequeño", "pequeña"), ("alto", "alta"),
          ("nuevo", "nueva"), ("viejo", "vieja"), ("bonito", "bonita"), ("largo", "larga"), ("frío", "fría"),
          ("limpio", "limpia"), ("caro", "cara")]
DE_NOUNS = [("der", "Hund", "Hunde"), ("die", "Katze", "Katzen"), ("das", "Kind", "Kinder"), ("der", "Vogel", "Vögel"),
            ("der", "Mann", "Männer"), ("die", "Frau", "Frauen"), ("der", "Student", "Studenten"),
            ("das", "Pferd", "Pferde"), ("der", "Junge", "Jungen"), ("die", "Ente", "Enten"), ("der", "Nachbar", "Nachbarn"),
            ("das", "Mädchen", "Mädchen"), ("der", "Arbeiter", "Arbeiter"), ("die", "Schülerin", "Schülerinnen")]
DE_VERBS = [("läuft", "laufen"), ("schläft", "schlafen"), ("singt", "singen"), ("spielt", "spielen"),
            ("wartet", "warten"), ("lacht", "lachen"), ("arbeitet", "arbeiten"), ("schwimmt", "schwimmen"),
            ("tanzt", "tanzen"), ("liest", "lesen"), ("isst", "essen"), ("ruft", "rufen")]
DE_PLACES = ["im Garten", "am Fluss", "im Park", "zu Hause", "in der Küche", "auf dem Hof", "am Strand", "im Wald"]

PY_FUNCS = ["print", "len", "max", "min", "sum", "sorted", "list", "abs", "round", "str", "int", "float", "range",
            "enumerate", "zip", "dict", "set", "tuple", "np.mean", "np.array", "math.sqrt", "os.path.join"]
PY_ARGS = ["x", "y", "z", "data", "items", "values", "a", "b", "n", "i", "count", "scores", "row", "col", "key", "0", "1",
           "2", "10", "'a'", "\"name\"", "k + 1", "n - 1", "x * 2"]
HTML_TAGS = ["div", "span", "p", "li", "ul", "b", "i", "a", "h1", "h2", "td", "tr", "table", "em", "strong", "section",
             "label", "button", "header", "footer", "nav", "article", "small", "code"]
HTML_WORDS = ["Home", "About us", "Contact", "Read more", "Login", "Welcome back", "Price: $20", "Next page", "Settings",
              "Hello world", "Submit", "Our team", "Latest news", "Cart (2)", "FAQ", "Sign up today"]
TEX_CMDS = ["\\frac{%s}{%s}", "\\sqrt{%s}", "\\mathbf{%s}", "\\hat{%s}", "\\bar{%s}", "\\textit{%s}", "%s^{%s}",
            "%s_{%s}", "\\left(%s\\right)", "\\mathrm{%s}", "\\exp{%s}", "\\log{%s}"]
TEX_ATOMS = ["x", "y", "a", "b", "n", "k", "\\alpha", "\\beta", "\\theta", "2", "i", "t", "x+1", "n-1", "\\pi", "e"]
COPY_WORDS = ("time year people way day man thing woman life child world school state family student group country "
              "problem hand part place case week company system program question work government number night point "
              "home water room mother area money story fact month lot right study book eye job word business issue "
              "side kind head house service friend father power hour game line end member law car city community name "
              "president team minute idea kid body information back parent face others level office door health person "
              "art war history party result change morning reason research girl guy moment air teacher force education "
              "foot boy age policy music market sense nation plan college interest death experience effect class "
              "control care field development role effort rate heart drug show leader light voice wife police mind "
              "price report decision son view relationship town road arm difference value building action model season "
              "society tax director position player record paper space ground form event official matter center couple "
              "site project activity star table need court oil situation cost industry figure street image phone data "
              "picture practice piece land product doctor wall patient worker news test movie north love support "
              "technology step baby computer type attention film tree source organization hair window evidence population "
              "site garden river island bridge forest mountain cloud stone glass orange purple silver yellow").split()


# ---- generators: each returns a list of (template_id, text, answer, answer_alt, features dict) --------------------------

def _a_an(rng: random.Random, n: int) -> list:
    out = []
    art = lambda w: "an" if w in AN_NOUNS else "a"
    fruits = list(TREE_FRUITS)
    for _ in range(n * 4):
        t = rng.choice(["t0", "t1", "t2", "t3"])
        if t == "t0":                                      # Miller & Neo style: the second noun is fixed by the tree
            x, y = rng.sample(fruits, 2)
            text = f"I climbed up the {x} tree and picked {TREE_FRUITS[x]} {x}. I climbed up the {y} tree and picked"
            ans = " " + TREE_FRUITS[y]
        else:
            y = rng.choice(AN_NOUNS + A_NOUNS)
            name = rng.choice(NAMES)
            if t == "t1":
                text = f"Word: {y}\nExample sentence: {name} has"
            elif t == "t2":
                text = f"{name} lost the {y}, so {name} went to the store to buy"
            else:
                x = rng.choice([w for w in AN_NOUNS + A_NOUNS if art(w) != art(y)])
                text = f"In the first box there was {art(x)} {x}, and in the second box there was the {y}. Opening the second box, {name} found"
            ans = " " + art(y)
        alt = " an" if ans == " a" else " a"
        out.append((t, text, ans, alt, {"condition": ans.strip()}))
    return out


def _there(rng, n):
    out = []
    for _ in range(n * 4):
        sg = rng.choice(list(PLURALS)); pl = PLURALS[sg]
        place = rng.choice(["in the box", "on the table", "in the bag", "on the shelf", "in the drawer", "in the kitchen"])
        t = rng.choice(["t0", "t1", "t2"])
        plural = rng.random() < 0.5
        if t == "t0":
            text = f"Q: How many {pl} are {place}? A: There" if plural else f"Q: Is there a {sg} {place}? A: Yes, there"
        elif t == "t1":
            text = f"I counted {rng.randint(2, 9)} {pl} {place}. So there" if plural else f"I found exactly one {sg} {place}. So there"
        else:
            text = f"Are there any {pl} {place}? Yes, there" if plural else f"Is there any {sg} {place}? Yes, there"
        ans, alt = (" are", " is") if plural else (" is", " are")
        out.append((t, text, ans, alt, {"number": "pl" if plural else "sg"}))
    return out


def _sv(rng, n):
    out = []
    for _ in range(n * 4):
        frame, t = rng.choice(SV_FRAMES)
        s_sg = rng.choice(list(SV_SUBJECTS)); a_sg = rng.choice(list(SV_ATTRACTORS))
        s_pl, a_pl = rng.random() < 0.5, rng.random() < 0.5
        S = SV_SUBJECTS[s_sg] if s_pl else s_sg
        A = SV_ATTRACTORS[a_sg] if a_pl else a_sg
        P = rng.choice(SV_PREPS)
        body = frame.format(S=S, P=P, A=A)
        text = ("The " + body) if t == "t0" else body
        ans, alt = (" are", " is") if s_pl else (" is", " are")
        out.append((t, text, ans, alt, {"subject": "pl" if s_pl else "sg", "attractor": "pl" if a_pl else "sg",
                                        "mismatch": str(s_pl != a_pl)}))
    return out


def _fr(rng, n):
    out = []
    def np_(g, w, pl):
        if pl:
            return "les " + FR_PLURAL.get(w, w + ("" if w.endswith(("s", "x")) else "s"))
        return ("le " if g == "m" else "la ") + w
    def adj(a, g, pl):
        return a[(2 if pl else 0) + (1 if g == "f" else 0)]
    for _ in range(n * 4):
        g1, g2 = rng.choice("mf"), rng.choice("mf")
        pl1, pl2 = rng.random() < 0.3, rng.random() < 0.3
        w1, w2 = rng.choice(FR_NOUNS[g1]), rng.choice(FR_NOUNS[g2])
        if w1 == w2:
            continue
        a = rng.choice(FR_ADJ)
        v1, v2 = ("sont" if pl1 else "est"), ("sont" if pl2 else "est")
        t = rng.choice(["t0", "t1"])
        first = f"{np_(g1, w1, pl1)} {v1} {adj(a, g1, pl1)}"
        second = f"{np_(g2, w2, pl2)} {v2}"
        text = (first[0].upper() + first[1:] + ". " + second[0].upper() + second[1:]) if t == "t0" \
            else f"Dans la pièce, {first} et {second}"
        ans = " " + adj(a, g2, pl2)
        other = adj(a, "m" if g2 == "f" else "f", pl2)
        alt = " " + other if other != adj(a, g2, pl2) else ""          # gris / gris: no gender contrast
        out.append((t, text, ans, alt, {"gender": g2, "number": "pl" if pl2 else "sg",
                                        "same_as_first": str(g1 == g2 and pl1 == pl2)}))
    return out


def _es(rng, n):
    out = []
    def np_(g, w, pl):
        if pl:
            p = ES_PLURAL.get(w, w + ("s" if w[-1] in "aeiouáéíóú" else "es"))
            return ("Los " if g == "m" else "Las ") + p
        return ("El " if g == "m" else "La ") + w
    def adj(a, g, pl):
        return a[1 if g == "f" else 0] + ("s" if pl else "")
    for _ in range(n * 4):
        g1, g2 = rng.choice("mf"), rng.choice("mf")
        pl1, pl2 = rng.random() < 0.3, rng.random() < 0.3
        w1, w2 = rng.choice(ES_NOUNS[g1]), rng.choice(ES_NOUNS[g2])
        if w1 == w2:
            continue
        a = rng.choice(ES_ADJ)
        t = rng.choice(["t0", "t1"])
        v1, v2 = ("son" if pl1 else "es"), ("son" if pl2 else "es")
        if t == "t0":
            text = f"{np_(g1, w1, pl1)} {v1} {adj(a, g1, pl1)}. {np_(g2, w2, pl2)} {v2}"
        else:
            text = f"En la habitación, {np_(g1, w1, pl1).lower()} {v1} {adj(a, g1, pl1)} y {np_(g2, w2, pl2).lower()} {v2}"
        ans = " " + adj(a, g2, pl2)
        alt = " " + adj(a, "m" if g2 == "f" else "f", pl2)
        out.append((t, text, ans, alt, {"gender": g2, "number": "pl" if pl2 else "sg",
                                        "same_as_first": str(g1 == g2 and pl1 == pl2)}))
    return out


def _de(rng, n):
    out = []
    for _ in range(n * 4):
        (ar1, n1s, n1p), (ar2, n2s, n2p) = rng.sample(DE_NOUNS, 2)
        vs, vp = rng.choice(DE_VERBS)
        pl1 = rng.random() < 0.5
        pl2 = (not pl1) if rng.random() < 0.7 else pl1
        s1 = ("die " + n1p) if pl1 else (ar1 + " " + n1s)
        s2 = ("die " + n2p) if pl2 else (ar2 + " " + n2s)
        place = rng.choice(DE_PLACES)
        t = rng.choice(["t0", "t1"])
        v1 = vp if pl1 else vs
        text = (f"{s1[0].upper() + s1[1:]} {v1} {place}. {s2[0].upper() + s2[1:]}" if t == "t0"
                else f"{s1[0].upper() + s1[1:]} {v1} {place}, und {s2}")
        ans, alt = (" " + vp, " " + vs) if pl2 else (" " + vs, " " + vp)
        out.append((t, text, ans, alt, {"number": "pl" if pl2 else "sg", "same_as_first": str(pl1 == pl2),
                                        "noun_number_marked": str(n2s != n2p)}))
    return out


def _code(rng, n):
    out = []
    for _ in range(n * 4):
        depth = rng.randint(2, 5)
        funcs = [rng.choice(PY_FUNCS) for _ in range(depth)]
        open_text = ""
        for i, f in enumerate(funcs):
            pre = ", ".join(rng.sample(PY_ARGS, rng.randint(0, 2)))
            open_text += f + "(" + (pre + ", " if pre and i < depth - 1 else (pre if i == depth - 1 else ""))
        last_arg = rng.choice(PY_ARGS)
        if open_text.endswith("(") or open_text.endswith(", "):
            open_text += last_arg
        lhs = rng.choice(["result = ", "y = ", "out = ", "", "    return ", "total += "])
        closed = rng.randint(0, depth - 1)               # calls closed before the cut; each is followed by ", <arg>" so
        text = lhs + open_text                          # the prompt always ends on an argument, never on ")"
        for _ in range(closed):
            text += "), " + rng.choice(PY_ARGS)
        ans = ")" * (depth - closed)
        out.append(("t0", text, ans, "", {"depth": str(depth), "remaining": str(len(ans))}))
    return out


def _html(rng, n):
    out = []
    for _ in range(n * 4):
        depth = rng.randint(2, 5)
        tags = [rng.choice(HTML_TAGS) for _ in range(depth)]
        text = ""
        for i, tg in enumerate(tags):
            attr = rng.choice(["", "", f' class="{rng.choice(["main", "item", "title", "nav", "btn"])}"'])
            text += f"<{tg}{attr}>"
            if i < depth - 1 and rng.random() < 0.3:              # a complete sibling element as a distractor
                sib = rng.choice(HTML_TAGS)
                text += f"<{sib}>{rng.choice(HTML_WORDS)}</{sib}>"
        text += rng.choice(HTML_WORDS)
        closed = rng.randint(0, depth - 1)
        for tg in reversed(tags[depth - closed:]):
            text += f"</{tg}>"
        text += "</"
        out.append(("t0", text, tags[depth - closed - 1], "", {"depth": str(depth), "closed_before": str(closed)}))
    return out


def _latex(rng, n):
    out = []
    for _ in range(n * 4):
        depth = rng.randint(2, 4)
        s = "%s"
        for _ in range(depth):
            c = rng.choice(TEX_CMDS)
            k = c.count("%s")
            fill = [rng.choice(TEX_ATOMS) for _ in range(k - 1)]
            s = s.replace("%s", (c % tuple(fill + ["%s"])) if k > 1 else (c % "%s"), 1)
        full = s.replace("%s", rng.choice(TEX_ATOMS))
        # cut just after the innermost atom: everything after it is the closers still owed (plus "\right)" etc.)
        atom_end = s.index("%s")
        text = "$" + full[: atom_end + len(full) - len(s) + 2]
        rest = full[atom_end + len(full) - len(s) + 2:]
        if not rest or not rest.startswith(("}", "\\right")):
            continue
        out.append(("t0", text, rest + "$", "", {"depth": str(depth)}))
    return out


def _copy(rng, n):
    out = []
    for _ in range(n * 4):
        L = rng.randint(6, 16)
        words = rng.sample(COPY_WORDS, L)
        k = rng.randint(1, L - 1)                         # words of the second copy already given
        t = rng.choice(["t0", "t1"])
        if t == "t0":
            text = " ".join(words) + ". " + " ".join(words[:k])
            ans = " " + words[k]
        else:
            text = "List: " + ", ".join(words) + "\nRepeat the list: " + ", ".join(words[:k]) + ","
            ans = " " + words[k]
        out.append((t, text, ans, "", {"length": str(L), "position": str(k)}))
    return out


GEN = {"agree_a_an": _a_an, "agree_there": _there, "agree_sv": _sv, "agree_fr": _fr, "agree_es": _es, "agree_de": _de,
       "delim_code": _code, "delim_html": _html, "delim_latex": _latex, "copy_words": _copy}


def build_slices(seed: int = 0, n: dict | None = None) -> list[dict]:
    """All exploratory prompts: per task up to ``n[task]`` unique texts (fewer when the generator's space is smaller),
    split 50/50 discovery / held-out with a seed."""
    n = {**DEFAULT_N, **(n or {})}
    recs = []
    for task in TASKS:
        rng = random.Random(f"{task}:{seed}")
        seen, rows = set(), []
        for t, text, ans, alt, feat in GEN[task](rng, n[task]):
            if text in seen:
                continue
            seen.add(text); rows.append((t, text, ans, alt, feat))
            if len(rows) >= n[task]:
                break
        order = list(range(len(rows))); random.Random(f"split:{task}:{seed}").shuffle(order)
        heldout = set(order[: len(rows) // 2])
        for i, (t, text, ans, alt, feat) in enumerate(rows):
            recs.append({"prompt_id": f"{task}:{t}:{i}", "slice": "exploratory", "task": task, "template_id": t,
                         "split": "heldout" if i in heldout else "discovery", "text": text, "answer": ans,
                         "answer_alt": alt, **{f"f_{k}": v for k, v in feat.items()}})
    return recs
