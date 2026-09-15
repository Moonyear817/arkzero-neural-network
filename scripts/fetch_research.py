import urllib.request, json, pathlib, concurrent.futures

out = pathlib.Path("research/sources")
repos = [
    "djpadbit/Arknights-RE",
    "DD-Channel/Arknights-Re-Engraved",
    "Kengxxiao/ArknightsGameData",
    "winny727/ArknightsMapViewer",
    "christwsy-zz/Arknights-Simulation",
]


def get(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ArkZero-Research"})
        return urllib.request.urlopen(req, timeout=45).read()
    except Exception as e:
        return json.dumps({"error": str(e)}).encode()


def job(repo):
    name = repo.split("/")[-1]
    meta = json.loads(get("https://api.github.com/repos/" + repo))
    (out / (name + ".json")).write_text(json.dumps(meta, indent=2))
    branch = meta.get("default_branch", "main")
    tree = get(f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1")
    (out / (name + "_tree.json")).write_bytes(tree)
    (out / (name + "_README.md")).write_bytes(
        get(f"https://raw.githubusercontent.com/{repo}/{branch}/README.md")
    )
    print(repo, meta.get("license"), meta.get("pushed_at"))


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as e:
    list(e.map(job, repos))
for q in [
    "Arknights+battle+simulator",
    "Arknights+simulator",
    "Arknights+combat+simulation",
    "Arknights+level+simulator",
]:
    (out / ("search_" + q + ".json")).write_bytes(
        get("https://api.github.com/search/repositories?q=" + q + "&per_page=10")
    )
