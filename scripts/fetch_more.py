import urllib.request, json, pathlib, concurrent.futures

b = pathlib.Path("research/sources")
jobs = []
for repo, paths in {
    "winny727/ArknightsMapViewer": [
        "ArknightsMapViewer/ArknightsMap/LevelData.cs",
        "ArknightsMapViewer/ArknightsMap/LevelReader.cs",
        "ArknightsMapViewer/View/TimelineSimulator.cs",
        "ArknightsMapViewer/View/Drawer/MapDrawer.cs",
        "ArknightsMapViewer/Utils/PathFinding.cs",
    ],
    "DD-Channel/Arknights-Re-Engraved": [
        "Assets/Arknights/Game/Entity/Entity.cs",
        "Assets/Arknights/Game/Entity/Monster.cs",
        "Assets/Arknights/Game/Entity/Char.cs",
        "Assets/Arknights/Game/TargetSelector.cs",
        "Assets/Arknights/Game/Dungeon.cs",
    ],
    "christwsy-zz/Arknights-Simulation": [
        "ArknightsSimulationCore/Bases/AbstractGame.cs",
        "ArknightsSimulationCore/Bases/AbstractEnemy.cs",
        "ArknightsSimulationCore/Bases/Agent/AbstractDpsAgent.cs",
    ],
}.items():
    sha = json.loads((b / (repo.split("/")[-1] + "_tree.json")).read_text())["sha"]
    for p in paths:
        jobs.append(
            (
                f"https://raw.githubusercontent.com/{repo}/{sha}/{p}",
                b / (repo.split("/")[-1] + "_" + p.replace("/", "_")),
            )
        )


def job(j):
    try:
        j[1].write_bytes(urllib.request.urlopen(j[0], timeout=40).read())
    except Exception as e:
        print(j[0], e)


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as e:
    list(e.map(job, jobs))
