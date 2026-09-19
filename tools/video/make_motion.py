"""Vidéo « motion design » de SynthID-Graph : animations HTML rendues image par image avec Chrome (Playwright),
sous-titres incrustés, voix off locale (voix Windows OneCore), montage ffmpeg.

    pip install -e ".[video]"
    python tools/video/make_motion.py                 # Windows : versions avec et sans voix
    python tools/video/make_motion.py --no-voice      # tout système : version sous-titrée seule

Options : --voice "Microsoft Paul" | "Microsoft Hortense", --rate 1.12, --fps 30, --out reports/video
Le rendu utilise le Google Chrome installé (Playwright, channel « chrome »).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# (identifiant de scène, pause finale en s, [(texte lu, sous-titre), ...])
SCRIPT = [
    ("hook", 0.5, [
        ("Un client modèle, pendant des mois.", "Un client modèle, pendant des mois."),
        ("Puis, en quelques jours, il vide tous ses plafonds de crédit. Et disparaît.",
         "Puis, en quelques jours, il vide tous ses plafonds de crédit… et disparaît."),
        ("Ce client n'a jamais existé. C'est une identité synthétique.",
         "Ce client n'a jamais existé : c'est une identité synthétique."),
    ]),
    ("what", 0.5, [
        ("Un vrai numéro d'identité, souvent volé à un enfant…", "Un vrai numéro d'identité, souvent volé à un enfant…"),
        ("plus un nom et des contacts inventés.", "…plus un nom et des contacts inventés."),
        ("Le faux client paie bien pendant des mois, puis vide tout d'un coup. C'est le bust-out.",
         "Le faux client paie bien pendant des mois, puis vide tout d'un coup : c'est le « bust‑out »."),
        ("Et les fraudeurs en gèrent des dizaines à la fois. On parle d'anneau.",
         "Et les fraudeurs en gèrent des dizaines à la fois : on parle d'anneau."),
    ]),
    ("hard", 0.6, [
        ("Vu un par un, chaque faux client est irréprochable.", "Vu un par un, chaque faux client est irréprochable."),
        ("Résultat : les règles classiques ne repèrent que quinze pour cent des anneaux les plus rusés.",
         "Résultat : les règles classiques ne repèrent que 15 % des anneaux les plus rusés."),
    ]),
    ("idea", 0.6, [
        ("L'idée : ne plus regarder les clients un par un, mais le réseau qui les relie.",
         "L'idée : ne plus regarder les clients un par un, mais le réseau qui les relie."),
        ("Même téléphone. Même appareil. Même employeur inconnu.", "Même téléphone, même appareil, même employeur inconnu…"),
        ("Chaque fiche paraît normale. Le réseau, lui, trahit le groupe.",
         "Chaque fiche paraît normale. Le réseau, lui, trahit le groupe."),
    ]),
    ("built", 0.6, [
        ("Un simulateur crée trois banques, vingt mille clients et cent dix anneaux de fraudeurs.",
         "Un simulateur crée 3 banques, 20 000 clients et 110 anneaux de fraudeurs."),
        ("Les données personnelles deviennent des codes, rangés en graphe dans Neo4j.",
         "Les données personnelles deviennent des codes, rangés en graphe dans Neo4j."),
        ("Des modèles d'IA, dont des réseaux de neurones sur graphe, évaluent chaque client.",
         "Des modèles d'IA, dont des réseaux de neurones sur graphe (GNN), évaluent chaque client."),
        ("Les banques coopèrent par chiffrement, et les alertes arrivent en dossiers d'enquête.",
         "Les banques coopèrent par chiffrement, et les alertes arrivent en dossiers d'enquête."),
    ]),
    ("demo", 0.7, [
        ("Voici un vrai anneau détecté, dans Neo4j.", "Voici un vrai anneau détecté par le système, dans Neo4j."),
        ("Trente-cinq clients reliés par des téléphones, des appareils et des employeurs partagés.",
         "35 clients reliés par des téléphones, des appareils et des employeurs partagés."),
        ("Au centre, des comptes hôtes : trente-trois inscriptions pour gonfler des historiques de crédit.",
         "Au centre, des comptes hôtes : 33 inscriptions pour gonfler des historiques de crédit."),
        ("Ici, un même numéro d'identité porté par six identités. Et l'enquêteur ne voit que des codes.",
         "Ici, un même numéro d'identité porté par 6 identités. Et l'enquêteur ne voit que des codes."),
    ]),
    ("coop", 0.7, [
        ("Un fraudeur ouvre des comptes dans plusieurs banques, qui ne peuvent pas partager leurs données.",
         "Un fraudeur ouvre des comptes dans plusieurs banques, qui ne peuvent pas partager leurs données."),
        ("Chaque banque transforme ses données en codes communs, que personne ne peut décoder.",
         "Chaque banque transforme ses données en codes communs, que personne ne peut décoder."),
        ("Puis elle envoie des compteurs chiffrés, que le serveur additionne sans les lire.",
         "Puis elle envoie des compteurs chiffrés, que le serveur additionne sans les lire."),
        ("Aucune donnée personnelle échangée, et un signal de plus contre les anneaux.",
         "Aucune donnée personnelle échangée, et un signal de plus contre les anneaux."),
    ]),
    ("results", 0.9, [
        ("Sur les anneaux les plus rusés, les règles classiques en repèrent quinze pour cent.",
         "Sur les anneaux les plus rusés, les règles classiques en repèrent 15 %…"),
        ("Avec le réseau et les réseaux de neurones sur graphe : quatre-vingt-cinq pour cent.",
         "…avec le réseau et les réseaux de neurones sur graphe : 85 %…"),
        ("Avec la coopération entre banques : quatre-vingt-dix pour cent.",
         "…avec la coopération entre banques : 90 %."),
        ("Au total, près de quatre-vingt-quinze pour cent des anneaux sont repérés avant le vol, environ sept mois à l'avance.",
         "Au total, près de 95 % des anneaux sont repérés avant le vol, environ 7 mois à l'avance."),
    ]),
    ("outro", 3.0, [
        ("Des résultats obtenus sur données simulées, et vérifiés par dix-huit tests. Prochaine étape : de vraies données.",
         "Résultats obtenus sur données simulées, vérifiés par 18 tests automatiques. Prochaine étape : de vraies données."),
        ("SynthID-Graph. Seul, un fraudeur rusé est invisible. Son réseau, lui, le trahit.",
         "SynthID-Graph — seul, un fraudeur rusé est invisible. Son réseau, lui, le trahit."),
    ]),
]
INTRO, SCENE_LEAD, GAP = 0.5, 0.5, 0.35


def synthesize(voice: str, rate: float) -> list[list[Path]]:
    vdir = HERE / "voice"
    vdir.mkdir(exist_ok=True)
    items, paths = [], []
    for si, (_, _, sentences) in enumerate(SCRIPT):
        row = []
        for k, (say, _) in enumerate(sentences):
            p = vdir / f"s{si:02d}_{k:02d}.wav"
            items.append({"text": say, "path": str(p)})
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
    out = []
    for row in paths:
        durs = []
        for wp in row:
            data, sr = read_wav(wp)
            durs.append(len(data) / sr)
        out.append(durs)
    return out


def reading_durations() -> list[list[float]]:
    """Sans voix off : durée de lecture des sous-titres (environ 16 caractères par seconde)."""
    return [[max(1.8, len(cap) / 16) for _, cap in sentences] for _, _, sentences in SCRIPT]


def build_timeline(durations: list[list[float]]) -> tuple[dict, list[tuple[float, int, int]]]:
    """Cale chaque scène sur ses phrases ; renvoie la timeline et le début absolu de chaque phrase."""
    t = INTRO
    scenes, captions, starts = [], [], []
    for si, ((sid, hold, sentences), durs) in enumerate(zip(SCRIPT, durations)):
        lt, cues = SCENE_LEAD, []
        for k, ((_, cap), d) in enumerate(zip(sentences, durs)):
            cues.append([round(lt, 3), round(lt + d, 3)])
            captions.append({"a": round(t + lt, 3), "b": round(t + lt + d, 3), "text": cap})
            starts.append((t + lt, si, k))
            lt += d + GAP
        dur = lt - GAP + hold
        scenes.append({"id": sid, "start": round(t, 3), "dur": round(dur, 3), "cues": cues})
        t += dur
    return {"scenes": scenes, "captions": captions, "total": round(t, 3)}, starts


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


def render_video(timeline: dict, fps: int, out: Path, poster: Path):
    html = (HERE / "motion.html").read_text(encoding="utf-8")
    data = f"<script>window.TIMELINE = {json.dumps(timeline, ensure_ascii=False)};</script>\n"
    html = html.replace("<script>\nconst TL", data + "<script>\nconst TL", 1)
    page_file = HERE / "_render.html"
    page_file.write_text(html, encoding="utf-8")
    n = int(timeline["total"] * fps) + 1
    enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg",
                            "-i", "-", "-vf", "scale=in_range=full:out_range=tv,format=yuv420p", "-colorspace", "bt709",
                            "-color_primaries", "bt709", "-color_trc", "bt709", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                            "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page.goto(page_file.as_uri())
        page.wait_for_function("document.fonts.ready.then(() => true)")
        page.wait_for_function("Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)")
        page.wait_for_timeout(500)
        for f in range(n):
            page.evaluate(f"render({f / fps})")
            enc.stdin.write(page.screenshot(type="jpeg", quality=92))
            if f % (fps * 10) == 0:
                print(f"  image {f}/{n}", flush=True)
        # vignette de publication : l'écran final
        page.evaluate(f"render({timeline['total'] - 1.0})")
        page.screenshot(path=str(poster), type="png")
        browser.close()
    enc.stdin.close()
    enc.wait()
    page_file.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default="Microsoft Julie")
    ap.add_argument("--rate", type=float, default=1.12)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "reports" / "video"))
    ap.add_argument("--no-voice", action="store_true", help="sans voix off (automatique hors Windows)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with_voice = not args.no_voice and os.name == "nt"
    paths = synthesize(args.voice, args.rate) if with_voice else None
    timeline, starts = build_timeline(voice_durations(paths) if paths else reading_durations())
    (HERE / "timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"durée : {timeline['total']:.1f} s")
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
