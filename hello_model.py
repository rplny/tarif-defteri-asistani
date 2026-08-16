"""Foundry Local kurulumunu doğrulayan Hello Model testi."""
from foundry_local_sdk import Configuration, FoundryLocalManager


def main():
    config = Configuration(app_name="foundry_local_hello")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    model = manager.catalog.get_model("qwen2.5-0.5b")
    model.download(lambda p: print(f"\rModel indiriliyor: {p:.1f}%", end="", flush=True))
    print()
    model.load()
    client = model.get_chat_client()
    messages = [{"role": "user", "content": "Hello, world"}]
    print("Cevap: ", end="", flush=True)
    for chunk in client.complete_streaming_chat(messages):
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", None) if delta else None
        if content:
            print(content, end="", flush=True)
    print()
    model.unload()
    print("Foundry Local çalışıyor.")


if __name__ == "__main__":
    main()
