"""Rebuild or serve the local score gallery."""

from __future__ import annotations

import argparse
from functools import partial
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .compiler import render_musescore, write_musicxml
from .evaluate import evaluate_score
from .gallery import write_gallery
from .score_ir import read_performance, read_score


def rebuild_gallery(run_dir: Path | str, rerender: bool = False) -> Path:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    score = read_score(run_dir / "score.ir.json")
    if rerender:
        write_musicxml(score, run_dir / "score.musicxml")
        manifest["render"] = render_musescore(run_dir / "score.musicxml", run_dir)
    manifest["metrics"] = evaluate_score(
        score,
        read_performance(run_dir / "performance.ir.json"),
        run_dir / "score.musicxml",
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return write_gallery(score, manifest, run_dir)


def serve(run_dir: Path | str, port: int) -> None:
    run_dir = Path(run_dir).resolve()
    handler = partial(SimpleHTTPRequestHandler, directory=str(run_dir))
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Serving {run_dir} at http://127.0.0.1:{port}/gallery.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and serve a score-first preview gallery")
    parser.add_argument("--run", required=True, help="Generation output directory")
    parser.add_argument("--rerender", action="store_true", help="Run MuseScore again before serving")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    gallery = rebuild_gallery(args.run, args.rerender)
    print(f"Gallery written to {gallery.resolve()}")
    if args.serve:
        serve(args.run, args.port)


if __name__ == "__main__":
    main()
