#!/usr/bin/env python3
"""Write tests/fixtures/bpe_tokenizer_parity.tsv with HF `tokenizers` as the oracle.

Each line is `<JSON string literal><TAB><ids separated by spaces>`. The corpus is
fixed text plus a seeded pseudo-random mix, so a rerun gives the same file. It
holds the default transcription prompt (rendered, one audio placeholder), every
added token, and more than 500 strings: English, Italian, German, numerals,
emoji, CJK, Hindi, Arabic, code, whitespace runs, newlines, mixed scripts and a
20 KB paragraph.

    pip install tokenizers
    python scripts/gen_bpe_tokenizer_fixture.py tokenizer.json \
        tests/fixtures/bpe_tokenizer_parity.tsv
"""
import json
import random
import sys

try:
    from tokenizers import Tokenizer
except ImportError:
    print("pip install tokenizers", file=sys.stderr)
    sys.exit(77)

DEFAULT_PROMPT = ("Audio transcription. Language: as spoken. Wording: verbatim, including fillers "
                  "and repetitions. Formatting: punctuation and capitalization. Output: transcript only.")


def prompt_text(p):
    return ("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n" + p +
            " <|audio|><|im_end|>\n<|im_start|>assistant\n<think></think>")


FIXED = [
    # English
    "Hello, world!", "The quick brown fox jumps over the lazy dog.", "  leading and trailing  ",
    "It's 5 o'clock; don't panic -- we're fine.", "McDonald's, iPhone, YouTube, LocalAI, GGUF, GPT-4o",
    "so, um, yeah, I I I think that's, that's right", "ALL CAPS SENTENCE WITH SOME lower Case",
    "camelCaseIdentifier snake_case_identifier PascalCaseIdentifier kebab-case-identifier",
    "e.g. i.e. etc. vs. approx. Dr. Mr. Mrs. Ms.", "Ladies and gentlemen, welcome to the show.",
    # Italian
    "Ciao, come stai? Io sto benissimo, grazie!", "L'altro giorno ho mangiato un'ottima pizza a Napoli.",
    "Perché non andiamo al mare? Più tardi c'è già troppa gente.", "Città, università, libertà, così, però, già",
    "Quest'anno è andato così così, ma non mi lamento.", "Gli studenti dell'università sono partiti per l'Europa.",
    # German
    "Guten Morgen! Wie geht es Ihnen heute?", "Die Straße ist größer als erwartet, übrigens.",
    "Donaudampfschifffahrtsgesellschaftskapitän", "Über die Brücke gehen wir zum Bürgermeister.",
    "Fußgängerübergänge und Äpfel, Öl, Übung", "Das ist ein Test für ß, ä, ö und ü.",
    # French / Spanish / Portuguese / others
    "Où est la bibliothèque ? J'ai déjà vu l'été dernier.", "¿Cómo estás? ¡Muy bien, gracias!", "Não sei o que você quer dizer.",
    "Привет, мир! Как дела?", "Γειά σου Κόσμε", "Merhaba dünya, İstanbul ve ışık", "Cześć, jak się masz?",
    # Digits and numerals
    "0", "1234567890", "3.14159265358979", "1,000,000 dollars", "2026-11-13 23:59:59", "+39 333 123 4567",
    "x = 42; y = 3.5e-10; z = 0xFF", "①②③ ⅷ ½ ¾ ² ³ ¹", "٠١٢٣٤٥٦٧٨٩", "०१२३४५६७८९", "１２３４５",
    "I have 2 apples and 13 oranges and 100 grapes.", "Room 101, floor 7B, 3rd street",
    # Emoji
    "😀", "Hello 😀 world 🌍!", "👨‍👩‍👧‍👦 family", "🇮🇹🇩🇪🇺🇸 flags", "👍🏽 skin tone", "❤️ heart ✨ sparkles", "🎉🎉🎉🎉🎉",
    "I ❤ pizza 🍕🍕", "☕ coffee ☕ time", "🧑‍💻 coder",
    # CJK
    "你好，世界！", "今天天气很好，我们去公园玩吧。", "東京は日本の首都です。", "こんにちは、元気ですか？", "ありがとうございます",
    "안녕하세요, 만나서 반갑습니다.", "漢字とひらがなとカタカナ", "吾輩は猫である。名前はまだ無い。", "한국어 텍스트 테스트",
    "「引用」『二重引用』（括弧）", "ｱｲｳｴｵ ﾊﾟｰﾃｨｰ",
    # Hindi and other Indic
    "नमस्ते दुनिया", "मुझे हिंदी बहुत पसंद है।", "भारत एक विशाल देश है, जहाँ अनेक भाषाएँ बोली जाती हैं।", "क्षत्रिय ज्ञान श्री",
    "தமிழ் மொழி", "বাংলা ভাষা", "తెలుగు భాష", "ગુજરાતી ભાષા", "ਪੰਜਾਬੀ ਭਾਸ਼ਾ",
    # Arabic, Hebrew, Thai, others
    "مرحبا بالعالم", "اللغة العربية جميلة جدا", "السلام عليكم ورحمة الله وبركاته", "שלום עולם", "สวัสดีชาวโลก", "ខ្ញុំស្រលាញ់ភាសាខ្មែរ",
    "فارسی هم زبان زیبایی است", "አማርኛ", "ქართული ენა", "Հայերեն",
    # Combining marks and odd letters
    "e\u0301 a\u0308 o\u0302 n\u0303", "Z\u0361\u0316a\u0351l\u0310g\u0305o\u0300", "ǅ ǈ ǋ ǲ", "ʰ ʲ ˆ ˇ ᴬ ᴮ", "ªº", "ĳ Ĳ ǆ Ǆ",
    # Code
    "def foo(x, y):\n    return x + y\n", "#include <stdio.h>\nint main(void) {\n  printf(\"hi\\n\");\n  return 0;\n}\n",
    "const x = (a, b) => { return a?.b ?? c; };", "SELECT * FROM users WHERE id = 42 AND name LIKE '%bob%';",
    "<div class=\"a\"><span>text</span></div>", "{\"key\": [1, 2, 3], \"nested\": {\"a\": null}}", "https://example.com/path?q=1&r=2#frag",
    "user@example.com", "C:\\Users\\name\\file.txt", "/usr/local/bin/python3 --version", "a/b/c/d///e", "x->y => z", "!!! ??? ... ,,, ;;;",
    "$ git commit -m 'fix: thing'", "for i in range(10): print(i)", "if (a && b || !c) { x++; y--; }", "//// comment ////",
    "`code` **bold** _italic_ ~strike~ [link](http://x.y)", "<|im_start|>user\nhi<|im_end|>", "<spk:0> hello <spk:1> hi", "<ev:breath><ev:throat_cough>",
    # Whitespace and newlines
    "", " ", "  ", "   ", "\t", "\n", "\r\n", "\n\n\n", "\r", " \n ", "a\nb", "a\n\nb", "a \n b", "a  \n  b", "a\r\nb\r\nc", "a\tb\t\tc",
    "word    word", "word     \n     word", "trailing space ", "trailing newlines\n\n", "\n\nleading newlines", "   \t  \n  \t  ",
    "line1\nline2\nline3\n", "a\u00a0b", "a\u2003b\u2009c", "a\u3000b", "a\u0085b", "a\u2028b\u2029c", "a\u200bb", "a\ufeffb", "a\u0000b", "a\u0007b\u001bc\u007fd",
    # Mixed scripts
    "Hello 世界 مرحبا नमस्ते Привет", "iPhone 15 Pro Max（苹果）价格是 $999", "Il prezzo è 5€ o $6 o £7 o ¥8", "Tokyo 東京 → Berlin 柏林 ✈ 🛫",
    "mixedCASE中文abc日本語DEF", "ab12cd34ef56", "12ab34cd", "A1B2C3 d4e5f6", "foo_bar123 baz-456",
    # Quotes, punctuation
    "“smart quotes” ‘single’ „low“ «guillemets» ‹x›", "dash — en – hyphen - minus − tilde ~", "ellipsis… … ...", "(parentheses) [brackets] {braces}",
    "a,b;c:d.e!f?g", "...!!!???", "###", "---", "===>", "*" * 20, ".  .  .", "\"\"\"", "'''",
    # Casing edge cases
    "ÀÉÎÕÜ àéîõü", "ǅungla", "ΑΒΓ αβγ", "ΣΊΣΥΦΟΣ σίσυφος", "İi ıI", "ﬁ ﬂ ﬃ", "ǰ", "ß ẞ",
    # Long runs
    "a" * 300, "ab" * 200, "the " * 150, " " * 200, "\n" * 100, "!" * 150, "1" * 100, "é" * 120, "你" * 120, "😀" * 60, "x\n" * 80,
    "Mississippi " * 40, "-" * 80 + "\n" + "=" * 80, "0123456789" * 30,
]

WORDS_EN = ("the of and to in is that it was for on are with as his they be at one have this from by hot word but what some we can out other "
            "were all there when up use your how said an each she which do their time if will way about many then them write would like so "
            "these her long make thing see him two has look more day could go come did number sound no most people my over know water than "
            "call first who may down side been now find any new work part take get place made live where after back little only round man "
            "year came show every good me give our under name very through just form sentence great think say help low line differ turn cause "
            "much mean before move right boy old too same tell does set three want air well also play small end put home read hand port large "
            "spell add even land here must big high such follow act why ask men change went light kind off need house picture try us again "
            "animal point mother world near build self earth father head stand own page should country found answer school grow study still "
            "learn plant cover food sun four between state keep eye never last let thought city tree cross farm hard start might story saw "
            "far sea draw left late run don't while press close night real life few north open seem together next white children begin got "
            "walk example ease paper group always music those both mark often letter until mile river car feet care second book carry took "
            "science eat room friend began idea fish mountain stop once base hear horse cut sure watch color face wood main enough plain girl "
            "usual young ready above ever red list though feel talk bird soon body dog family direct pose leave song measure door product "
            "black short numeral class wind question happen complete ship area half rock order fire south problem piece told knew pass since "
            "top whole king space heard best hour better true during hundred five remember step early hold west ground interest reach fast "
            "verb sing listen six table travel less morning ten simple several vowel toward war lay against pattern slow center love person "
            "money serve appear road map rain rule govern pull cold notice voice unit power town fine certain fly fall lead cry dark machine "
            "note wait plan figure star box noun field rest correct able pound done beauty drive stood contain front teach week final gave "
            "green oh quick develop ocean warm free minute strong special mind behind clear tail produce fact street inch multiply nothing "
            "course stay wheel full force blue object decide surface deep moon island foot system busy test record boat common gold possible "
            "plane stead dry wonder laugh thousand ago ran check game shape equate hot miss brought heat snow tire bring yes distant fill "
            "east paint language among").split()
WORDS_IT = ("il lo la i gli le un uno una di a da in con su per tra fra e ma però perché quando dove come che chi cui non più già "
            "anche ancora sempre mai molto poco tutto niente qualcosa casa città università libertà qualità sono sei è siamo siete "
            "ho hai ha abbiamo avete hanno fare dire andare vedere sapere volere potere dovere mangiare bere dormire parlare "
            "bellissimo bambino ragazza macchina strada tempo giorno notte mattina sera oggi domani ieri l'altro dell'anno un'ora "
            "nell'acqua all'inizio quest'anno c'è dov'è com'è senz'altro").split()
WORDS_DE = ("der die das ein eine und oder aber nicht ist sind war waren hat haben wird werden können müssen sollen wollen "
            "Haus Straße Bürgermeister Übung Äpfel Größe Mädchen schön über für gegen ohne durch zwischen Donaudampfschifffahrt "
            "Gesellschaft Wirtschaft Arbeit Zeit Jahr Mensch Welt Land Stadt Frau Mann Kind Tag Nacht Wasser Brot Käse").split()
SCRIPTS = ["你好世界", "今日は天気", "안녕하세요", "नमस्ते दुनिया", "مرحبا بالعالم", "Привет мир", "สวัสดี", "שלום", "😀", "🎉", "→", "€", "$", "%", "@", "#",
           "1", "22", "333", "4444", "-", "_", ".", ",", "!", "?", "(", ")", "\n", "\n\n", "\t", " ", "  ", "   ", "/", "\\", "'", '"', "<", ">", "[", "]",
           "é", "ü", "ñ", "ß", "ç", "ø", "å", "ł", "Ж", "Ω", "ά", "e\u0301"]


def random_strings(rng, count):
    out = []
    for _ in range(count):
        kind = rng.randrange(6)
        n = rng.randrange(1, 40)
        if kind == 0:
            s = " ".join(rng.choice(WORDS_EN) for _ in range(n))
        elif kind == 1:
            s = " ".join(rng.choice(WORDS_IT) for _ in range(n))
        elif kind == 2:
            s = " ".join(rng.choice(WORDS_DE) for _ in range(n))
        elif kind == 3:
            s = "".join(rng.choice(SCRIPTS) for _ in range(n))
        elif kind == 4:
            s = "".join(rng.choice(SCRIPTS + WORDS_EN) + rng.choice(["", " ", " ", "\n"]) for _ in range(n))
        else:
            s = "".join(chr(rng.choice([rng.randrange(32, 127), rng.randrange(0xA0, 0x2FF), rng.randrange(0x400, 0x4FF),
                                        rng.randrange(0x900, 0x97F), rng.randrange(0x4E00, 0x4F00), rng.randrange(0x1F300, 0x1F5FF)]))
                        for _ in range(n))
        if rng.random() < 0.2:
            s = s.capitalize()
        if rng.random() < 0.1:
            s = s.upper()
        if rng.random() < 0.3:
            s += rng.choice([".", "!", "?", "...", "\n", " ", ""])
        out.append(s)
    return out


def paragraph_20k(rng):
    parts, size = [], 0
    while size < 20000:
        sent = " ".join(rng.choice(WORDS_EN + WORDS_IT + WORDS_DE) for _ in range(rng.randrange(6, 20)))
        sent = sent.capitalize() + rng.choice([".", ".", "!", "?", ";", "..."])
        if rng.random() < 0.08:
            sent += "\n\n"
        elif rng.random() < 0.05:
            sent += " 你好，世界 😀 नमस्ते " + str(rng.randrange(10**6))
        parts.append(sent)
        size += len(sent) + 1
    return " ".join(parts)[:20000]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    tok = Tokenizer.from_file(sys.argv[1])
    with open(sys.argv[1], encoding="utf-8") as f:
        added = [a["content"] for a in json.load(f)["added_tokens"]]
    rng = random.Random(20261113)
    texts = [prompt_text(DEFAULT_PROMPT)]
    texts += added
    texts += FIXED
    texts += random_strings(rng, 420)
    texts.append(paragraph_20k(rng))
    seen, uniq = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
        for t in uniq:
            ids = tok.encode(t, add_special_tokens=False).ids
            f.write(json.dumps(t, ensure_ascii=True) + "\t" + " ".join(map(str, ids)) + "\n")
    print("wrote %d strings (%d added tokens)" % (len(uniq), len(added)))


if __name__ == "__main__":
    main()
