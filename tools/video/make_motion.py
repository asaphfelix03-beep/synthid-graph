"""Vidéo explicative de SynthID-Graph : animations HTML rendues image par image avec Chrome (Playwright),
sous-titres et définitions incrustés, voix off locale (voix Windows OneCore), montage ffmpeg.

    pip install -e ".[video]"
    python tools/video/make_motion.py                 # Windows : versions avec et sans voix
    python tools/video/make_motion.py --no-voice      # tout système : version sous-titrée seule
    python tools/video/make_motion.py --preview       # images témoins, une par phrase

Options : --voice "Microsoft Paul" | "Microsoft Hortense", --rate 1.1, --fps 30, --out reports/video
Le rendu utilise le Google Chrome installé (Playwright, channel « chrome »).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import qrcode
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
REPO_URL = "https://github.com/asaphfelix03-beep/synthid-graph"

# Chaque scène : (identifiant, pause finale en s, phrases). Une phrase : (texte lu, sous-titre[, (terme, définition)]).
# Les chiffres viennent de l'expérience configs/small.yaml (graine 42) : voir docs/RESULTATS.md.
SCRIPT = [
    ("intro", 0.8, [
        ("Comment une banque peut-elle repérer un client… qui n'existe pas ?",
         "Comment une banque peut-elle repérer un client… qui n'existe pas ?"),
    ]),
    ("trust", 0.6, [
        ("Quand vous ouvrez un compte, la banque vérifie votre identité, puis vous accorde un crédit.",
         "Quand vous ouvrez un compte, la banque vérifie votre identité, puis vous accorde un crédit."),
        ("Plus vous remboursez à l'heure, plus elle vous fait confiance, et plus votre plafond de crédit augmente.",
         "Plus vous remboursez à l'heure, plus elle vous fait confiance… et plus votre plafond de crédit augmente.",
         ("Plafond de crédit", "Le montant maximum que la banque vous autorise à emprunter.")),
        ("Tout ce système repose sur une idée simple : chaque client est une vraie personne.",
         "Tout ce système repose sur une idée simple : chaque client est une vraie personne."),
    ]),
    ("synth", 0.6, [
        ("Des fraudeurs exploitent justement cette confiance, en fabriquant de fausses personnes.",
         "Des fraudeurs exploitent justement cette confiance, en fabriquant de fausses personnes."),
        ("Ils partent d'un vrai numéro d'identité, souvent volé à un enfant, et y ajoutent un nom, une date de naissance "
         "et un téléphone inventés.",
         "Ils partent d'un vrai numéro d'identité, souvent volé à un enfant… et y ajoutent un nom, une date de naissance "
         "et un téléphone inventés."),
        ("C'est ce qu'on appelle une identité synthétique.",
         "C'est ce qu'on appelle une identité synthétique.",
         ("Identité synthétique", "Une fausse personne, fabriquée en mélangeant de vraies et de fausses informations.")),
    ]),
    ("bust", 0.6, [
        ("Pendant des mois, ce faux client se comporte comme un client parfait : petits achats, remboursements toujours à l'heure.",
         "Pendant des mois, ce faux client se comporte comme un client parfait : petits achats, remboursements toujours à l'heure."),
        ("La banque augmente son plafond, encore et encore.", "La banque augmente son plafond, encore et encore."),
        ("Puis, en quelques jours, il utilise tout son crédit… et disparaît. C'est le bust-out.",
         "Puis, en quelques jours, il utilise tout son crédit… et disparaît : c'est le « bust‑out ».",
         ("Bust-out", "Utiliser d'un coup tout le crédit disponible, puis disparaître sans rembourser.")),
        ("Personne ne rembourse, puisque la personne n'a jamais existé.",
         "Personne ne rembourse, puisque la personne n'a jamais existé."),
    ]),
    ("ring", 0.6, [
        ("Et un fraudeur ne crée pas une seule identité, mais des dizaines, qu'il fait vivre en même temps.",
         "Et un fraudeur ne crée pas une seule identité, mais des dizaines, qu'il fait vivre en même temps."),
        ("On parle d'un anneau de fraude.", "On parle d'un anneau de fraude.",
         ("Anneau de fraude", "Un groupe de fausses identités pilotées par les mêmes fraudeurs.")),
    ]),
    ("quiz", 0.6, [
        ("Petit test : parmi ces quatre clients, lesquels sont faux ?", "Petit test : parmi ces quatre clients, lesquels sont faux ?"),
        ("Impossible à dire. Vus un par un, ils ont tous l'air parfaits.",
         "Impossible à dire : vus un par un, ils ont tous l'air parfaits."),
        ("C'est pour ça que les règles classiques échouent : dans notre test, elles ne repèrent que trois anneaux rusés sur vingt.",
         "C'est pour ça que les règles classiques échouent : dans notre test, elles ne repèrent que 3 anneaux rusés sur 20."),
    ]),
    ("graph", 0.8, [
        ("Et si, au lieu de regarder chaque client séparément, on regardait ce qui les relie ?",
         "Et si, au lieu de regarder chaque client séparément, on regardait ce qui les relie ?"),
        ("Relions chaque client à son téléphone, à son appareil, à son employeur et à son adresse.",
         "Relions chaque client à son téléphone, à son appareil, à son employeur et à son adresse."),
        ("Trois d'entre eux partagent un téléphone, un appareil, et un employeur que personne ne connaît. "
         "Le quatrième ne partage que son adresse, avec sa famille : c'est normal.",
         "Trois d'entre eux partagent un téléphone, un appareil et un employeur inconnu. "
         "Le quatrième ne partage que son adresse… avec sa famille : c'est normal."),
        ("Ce réseau de points reliés s'appelle un graphe. Et ici, il trahit le groupe.",
         "Ce réseau de points reliés s'appelle un graphe. Et ici, il trahit le groupe.",
         ("Graphe", "Des points (clients, téléphones, employeurs…) reliés par des liens.")),
    ]),
    ("how", 0.7, [
        ("Pour tester cette idée, le projet crée d'abord un monde simulé : trois banques, vingt mille clients, "
         "et cent dix anneaux cachés parmi eux.",
         "Pour tester cette idée, le projet crée d'abord un monde simulé : 3 banques, 20 000 clients et 110 anneaux cachés parmi eux."),
        ("Pourquoi simulé ? Parce que les vraies données bancaires sont confidentielles, et qu'ici, on sait exactement qui triche.",
         "Pourquoi simulé ? Parce que les vraies données bancaires sont confidentielles… et qu'ici, on sait exactement qui triche."),
        ("Les données personnelles sont remplacées par des codes : le système voit les liens, jamais les vrais numéros.",
         "Les données personnelles sont remplacées par des codes : le système voit les liens, jamais les vrais numéros."),
        ("Tout est rangé dans Neo4j, une base de données faite pour les graphes.",
         "Tout est rangé dans Neo4j, une base de données faite pour les graphes."),
        ("Enfin, une intelligence artificielle apprend à partir d'anciens cas de fraude, et évalue chaque client "
         "en regardant aussi ses voisins.",
         "Enfin, une intelligence artificielle apprend à partir d'anciens cas de fraude, et évalue chaque client "
         "en regardant aussi ses voisins.",
         ("IA sur graphe (GNN)", "Une intelligence artificielle qui juge chaque point en regardant aussi ses voisins.")),
    ]),
    ("demo", 0.8, [
        ("Voici un vrai anneau détecté par le système, affiché dans Neo4j.",
         "Voici un vrai anneau détecté par le système, affiché dans Neo4j."),
        ("Trente-cinq clients, reliés par des téléphones, des appareils et des employeurs partagés.",
         "35 clients, reliés par des téléphones, des appareils et des employeurs partagés."),
        ("Au centre, des comptes hôtes : des membres s'y sont inscrits trente-trois fois, pour gonfler leur historique de crédit.",
         "Au centre, des comptes hôtes : des membres s'y sont inscrits 33 fois pour gonfler leur historique de crédit."),
        ("Autre requête : ce numéro d'identité est utilisé par six identités différentes.",
         "Autre requête : ce numéro d'identité est utilisé par 6 identités différentes."),
        ("Et l'enquêteur ne voit que des codes, jamais les vraies données personnelles.",
         "Et l'enquêteur ne voit que des codes, jamais les vraies données personnelles."),
        ("Pour chaque alerte, le système prépare un dossier qui explique pourquoi ce client est suspect.",
         "Pour chaque alerte, le système prépare un dossier qui explique pourquoi ce client est suspect."),
    ]),
    ("coop", 0.8, [
        ("Un fraudeur ouvre souvent des comptes dans plusieurs banques. Mais les banques n'ont pas le droit de partager "
         "les données de leurs clients.",
         "Un fraudeur ouvre souvent des comptes dans plusieurs banques… mais les banques n'ont pas le droit de partager "
         "les données de leurs clients."),
        ("La solution : chaque banque transforme ses données en codes communs, que personne ne peut décoder.",
         "La solution : chaque banque transforme ses données en codes communs, que personne ne peut décoder."),
        ("Puis elle glisse des compteurs chiffrés dans une urne scellée : le serveur les additionne sans jamais pouvoir les lire.",
         "Puis elle glisse des compteurs chiffrés dans une « urne scellée » : le serveur les additionne sans jamais pouvoir les lire.",
         ("Chiffrement homomorphe", "Faire des calculs sur des données chiffrées, sans jamais les déchiffrer.")),
        ("La banque apprend seulement l'essentiel : ce téléphone est utilisé ailleurs, sous un autre nom. "
         "Aucune donnée personnelle n'a circulé.",
         "La banque apprend seulement l'essentiel : ce téléphone est utilisé ailleurs, sous un autre nom. "
         "Aucune donnée personnelle n'a circulé."),
    ]),
    ("results", 1.0, [
        ("Résultat, sur les vingt anneaux les plus rusés du test :", "Résultat, sur les 20 anneaux les plus rusés du test :"),
        ("Les règles classiques en repèrent trois.", "Les règles classiques en repèrent 3."),
        ("Une intelligence artificielle qui ne regarde que la fiche du client : cinq.",
         "Une IA qui ne regarde que la fiche du client : 5."),
        ("En ajoutant le réseau et l'IA sur graphe : dix-sept.", "En ajoutant le réseau et l'IA sur graphe : 17."),
        ("Et avec la coopération entre banques : dix-huit sur vingt.", "Et avec la coopération entre banques : 18 sur 20."),
        ("Sur l'ensemble du test, cinquante-cinq anneaux sur cinquante-sept sont repérés avant le vol, environ sept mois à l'avance.",
         "Sur l'ensemble du test, 55 anneaux sur 57 sont repérés avant le vol, environ 7 mois à l'avance."),
        ("Et environ un client honnête sur cent seulement est signalé par erreur.",
         "Et environ 1 client honnête sur 100 seulement est signalé par erreur."),
    ]),
    ("next", 0.8, [
        ("Attention : ces résultats viennent d'un monde simulé. Ils montrent que l'idée fonctionne, "
         "pas encore qu'elle marche dans une vraie banque.",
         "Attention : ces résultats viennent d'un monde simulé. Ils montrent que l'idée fonctionne, "
         "pas encore qu'elle marche dans une vraie banque."),
        ("Prochaine étape : la tester sur de vraies données, anonymisées.",
         "Prochaine étape : la tester sur de vraies données, anonymisées."),
        ("Le code, les résultats et les tests sont en accès libre sur GitHub.",
         "Le code, les résultats et les tests sont en accès libre sur GitHub."),
    ]),
    ("outro", 3.5, [
        ("SynthID-Graph. Seul, un fraudeur rusé est invisible. Son réseau, lui, le trahit.",
         "SynthID-Graph — seul, un fraudeur rusé est invisible. Son réseau, lui, le trahit."),
    ]),
]
INTRO, SCENE_LEAD, GAP = 0.4, 0.5, 0.35


def synthesize(voice: str, rate: float) -> list[list[Path]]:
    vdir = HERE / "voice"
    vdir.mkdir(exist_ok=True)
    for old in vdir.glob("*.wav"):
        old.unlink()
    items, paths = [], []
    for si, (_, _, sentences) in enumerate(SCRIPT):
        row = []
        for k, sent in enumerate(sentences):
            p = vdir / f"s{si:02d}_{k:02d}.wav"
            items.append({"text": sent[0], "path": str(p)})
            row.append(p)
        paths.append(row)
    job = vdir / "job.json"
    job.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(HERE / "tts.ps1"),
                    "-Json", str(job), "-Voice", voice, "-Rate", str(rate)], check=True, capture_output=True)
    return paths


def read_wav(p: Path) -> tuple[np.ndarray, int]:
    """Lit un WAV mono 16 bits et retire les silences de début et de fin laissés par la synthèse vocale."""
    with wave.open(str(p)) as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1).astype(np.int16)
    loud = np.flatnonzero(np.abs(data.astype(np.int32)) > 400)
    if len(loud):
        margin = int(0.06 * sr)
        data = data[max(0, loud[0] - margin):min(len(data), loud[-1] + margin)]
    return data, sr


def voice_durations(paths: list[list[Path]]) -> list[list[float]]:
    return [[len(read_wav(wp)[0]) / read_wav(wp)[1] for wp in row] for row in paths]


def reading_durations() -> list[list[float]]:
    """Sans voix off : durée de lecture des sous-titres (environ 15 caractères par seconde)."""
    return [[max(2.0, len(sent[1]) / 15) for sent in sentences] for _, _, sentences in SCRIPT]


def build_timeline(durations: list[list[float]]) -> tuple[dict, list[tuple[float, int, int]]]:
    """Cale chaque scène sur ses phrases ; renvoie la timeline et le début absolu de chaque phrase."""
    t = INTRO
    scenes, captions, defs, starts = [], [], [], []
    for si, ((sid, hold, sentences), durs) in enumerate(zip(SCRIPT, durations)):
        lt, cues, scene_defs = SCENE_LEAD, [], []
        for k, (sent, d) in enumerate(zip(sentences, durs)):
            cues.append([round(lt, 3), round(lt + d, 3)])
            captions.append({"a": round(t + lt, 3), "b": round(t + lt + d, 3), "text": sent[1], "s": sid})
            starts.append((t + lt, si, k))
            if len(sent) > 2:
                scene_defs.append({"a": round(t + lt + 0.4, 3), "term": sent[2][0], "text": sent[2][1]})
            lt += d + GAP
        dur = lt - GAP + hold
        for dd in scene_defs:  # une définition reste affichée jusqu'à la fin de sa scène
            dd["b"] = round(t + dur - 0.3, 3)
        defs.extend(scene_defs)
        scenes.append({"id": sid, "start": round(t, 3), "dur": round(dur, 3), "cues": cues})
        t += dur
    return {"scenes": scenes, "captions": captions, "defs": defs, "total": round(t, 3)}, starts


def mix_audio(placements: list[tuple[float, Path]], total: float, out: Path):
    sr = read_wav(placements[0][1])[1]
    track = np.zeros(int((total + 1) * sr), dtype=np.float32)
    for start, wp in placements:
        data, _ = read_wav(wp)
        i = int(start * sr)
        track[i:i + len(data)] += data.astype(np.float32)
    track = np.clip(track, -32768, 32767).astype(np.int16)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(track.tobytes())


def qr_data_uri(url: str) -> str:
    img = qrcode.make(url, border=2, box_size=10)
    buf = io.BytesIO()
    img.save(buf)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def open_page(browser, timeline: dict):
    """Injecte la timeline et les médias générés (QR code) dans motion.html, puis ouvre la page à 1920 × 1080."""
    html = (HERE / "motion.html").read_text(encoding="utf-8")
    assets = {"qr": qr_data_uri(REPO_URL), "url": REPO_URL.removeprefix("https://")}
    data = (f"<script>window.TIMELINE = {json.dumps(timeline, ensure_ascii=False)};\n"
            f"window.ASSETS = {json.dumps(assets)};</script>\n")
    page_file = HERE / "_render.html"
    page_file.write_text(html.replace("<script>\nconst TL", data + "<script>\nconst TL", 1), encoding="utf-8")
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    page.goto(page_file.as_uri())
    page.wait_for_function("document.fonts.ready.then(() => true)")
    page.wait_for_function("Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)")
    page.wait_for_timeout(500)
    return page, page_file


def render_video(timeline: dict, fps: int, out: Path, poster: Path):
    n = int(timeline["total"] * fps) + 1
    enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg",
                            "-i", "-", "-vf", "scale=in_range=full:out_range=tv,format=yuv420p", "-colorspace", "bt709",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                            "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page, page_file = open_page(browser, timeline)
        for f in range(n):
            page.evaluate(f"render({f / fps})")
            enc.stdin.write(page.screenshot(type="jpeg", quality=92))
            if f % (fps * 15) == 0:
                print(f"  image {f}/{n}", flush=True)
        page.evaluate(f"render({timeline['total'] - 1.0})")  # vignette de publication : l'écran final
        page.screenshot(path=str(poster), type="png")
        browser.close()
    enc.stdin.close()
    enc.wait()
    page_file.unlink(missing_ok=True)


def preview(timeline: dict, folder: Path):
    """Images témoins : la fin de chaque phrase, pour vérifier la mise en page avant un rendu complet."""
    folder.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page, page_file = open_page(browser, timeline)
        for s in timeline["scenes"]:
            for k, (_, b) in enumerate(s["cues"]):
                page.evaluate(f"render({s['start'] + b - 0.15})")
                page.screenshot(path=str(folder / f"{s['id']}_{k}.png"))
        browser.close()
    page_file.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default="Microsoft Julie")
    ap.add_argument("--rate", type=float, default=1.1)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "reports" / "video"))
    ap.add_argument("--no-voice", action="store_true", help="sans voix off (automatique hors Windows)")
    ap.add_argument("--preview", action="store_true", help="images témoins seulement, sans vidéo")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with_voice = not args.no_voice and os.name == "nt"
    paths = synthesize(args.voice, args.rate) if with_voice else None
    timeline, starts = build_timeline(voice_durations(paths) if paths else reading_durations())
    (HERE / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"durée : {timeline['total']:.1f} s")
    if args.preview:
        preview(timeline, out / "preview")
        print("images témoins :", out / "preview")
        return
    silent = out / "SynthID-Graph_video_sans_voix.mp4"
    render_video(timeline, args.fps, silent, out / "SynthID-Graph_vignette.png")
    if paths:
        narration = HERE / "voice" / "narration.wav"
        mix_audio([(t, paths[si][k]) for t, si, k in starts], timeline["total"], narration)
        voiced = out / "SynthID-Graph_video_voix.mp4"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(narration), "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-shortest", "-movflags", "+faststart",
                        str(voiced)], check=True)
        print("vidéo avec voix :", voiced)
    print("vidéo sans voix :", silent)


if __name__ == "__main__":
    main()
