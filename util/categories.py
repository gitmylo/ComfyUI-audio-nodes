def format_name(name: str, emoji: str = "🔉") -> str:
    return emoji + " " + name

def category(name: str) -> str:
    return format_name("AudioNodes/" + name)
