import json, urllib.request, pathlib, hashlib, concurrent.futures

base = pathlib.Path("research/sources")
repo = "Kengxxiao/ArknightsGameData"
sha = json.loads((base / "ArknightsGameData_tree.json").read_text())["sha"]
paths = [
    "levels/obt/main/level_main_00-01.json",
    "levels/obt/main/level_main_01-01.json",
    "levels/enemydata/enemy_database.json",
    "excel/character_table.json",
    "excel/skill_table.json",
    "excel/range_table.json",
    "excel/uniequip_table.json",
    "excel/favor_table.json",
    "excel/battle_equip_table.json",
    "battle/battle_misc_table.json",
]


def fetch(p):
    url = f"https://raw.githubusercontent.com/{repo}/{sha}/zh_CN/gamedata/{p}"
    b = urllib.request.urlopen(url, timeout=60).read()
    dest = pathlib.Path("data/real") / pathlib.Path(p).name
    dest.write_bytes(b)
    return {"path": str(dest), "url": url, "sha256": hashlib.sha256(b).hexdigest()}


with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:
    records = list(ex.map(fetch, paths))
pathlib.Path("data/manifest.json").write_text(json.dumps(records, indent=2))
for repo, filters in [
    (
        "winny727/ArknightsMapViewer",
        [
            "MoveRoute.cs",
            "StageData.cs",
            "RouteDrawer.cs",
            "Enemy.cs",
            "Wave.cs",
            "MapData.cs",
        ],
    ),
    (
        "DD-Channel/Arknights-Re-Engraved",
        ["Character.cs", "Enemy.cs", "Level.cs", "Attack.cs"],
    ),
]:
    name = repo.split("/")[-1]
    tree = json.loads((base / (name + "_tree.json")).read_text())
    sha = tree["sha"]
    pp = [
        x["path"]
        for x in tree["tree"]
        if any(x["path"].endswith("/" + f) for f in filters)
    ]
    print(name, pp)
    for p in pp:
        b = urllib.request.urlopen(
            f"https://raw.githubusercontent.com/{repo}/{sha}/{p}", timeout=45
        ).read()
        (base / (name + "_" + p.replace("/", "_"))).write_bytes(b)
