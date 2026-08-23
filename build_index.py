"""Create persistent FAISS indexes locally, without Gemini embeddings."""

import argparse

from app import PDFS, build_index


def main():
    parser = argparse.ArgumentParser(description="Create HawkAI local FAISS indexes.")
    parser.add_argument(
        "--rulebook",
        choices=list(PDFS),
        help="Build only one rulebook. Omit this option to build all three.",
    )
    args = parser.parse_args()
    rulebooks = [args.rulebook] if args.rulebook else list(PDFS)

    for position, rulebook_name in enumerate(rulebooks, start=1):
        print(f"\nBuilding {rulebook_name} ({position}/{len(rulebooks)})...")
        index, passages = build_index(rulebook_name)
        print(f"Saved index with {len(passages)} passages and {index.ntotal} vectors.")

    print("\nDone. Commit the new indexes folder to GitHub.")


if __name__ == "__main__":
    main()
