"""list_models.py - show every free model OpenRouter currently offers, grouped by family, so we can pick five from five different families."""

import requests

MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_free_models():
    """Ask OpenRouter for its catalogue and keep only the free models."""
    response = requests.get(MODELS_URL, timeout=30)
    response.raise_for_status()

    all_models = response.json()["data"]

    free_models = []
    for model in all_models:
        if model["id"].endswith(":free"):
            free_models.append(model)

    return free_models


def group_by_family(models):
    """Bucket models by the part of the id before the '/'."""
    families = {}

    for model in models:
        family = model["id"].split("/")[0]
        entry = (model["id"], model.get("context_length", 0))

        families.setdefault(family, []).append(entry)

    return families


def main():
    free_models = fetch_free_models()
    families = group_by_family(free_models)

    print(f"{len(free_models)} free models across {len(families)} families\n")

    for family in sorted(families):
        print(family)
        for model_id, context_length in sorted(families[family]):
            print(f"    {model_id:<58} {context_length:>9,} ctx")
        print()


if __name__ == "__main__":
    main()
