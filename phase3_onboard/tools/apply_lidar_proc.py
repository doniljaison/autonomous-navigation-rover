#!/usr/bin/env python3
"""
One-time patch: make main.Runtime use the LiDAR process (lidar_proc.LidarProcess)
instead of the in-process LidarThread on the real rover (the simulator keeps the thread).

    python -m phase3_onboard.tools.apply_lidar_proc          # patch (backup: main.py.pre_lidarproc)
    python -m phase3_onboard.tools.apply_lidar_proc --undo   # restore the backup

Switch back at run time without un-patching:  --set LIDAR_IN_PROCESS=False
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(HERE, "main.py")
BAK = MAIN + ".pre_lidarproc"

OLD = "        self.lidar = LidarThread(make_lidar(P, self.live_sim), self.logger)\n"
NEW = ("        if self.live_sim is None and getattr(P, \"LIDAR_IN_PROCESS\", True):\n"
       "            from phase3_onboard.lidar_proc import LidarProcess\n"
       "            self.lidar = LidarProcess(P, self.logger)   # SDK holds the GIL: keep it out of this process\n"
       "        else:\n"
       "            self.lidar = LidarThread(make_lidar(P, self.live_sim), self.logger)\n")


def main():
    if "--undo" in sys.argv:
        if not os.path.exists(BAK):
            sys.exit("no backup found: " + BAK)
        shutil.copy2(BAK, MAIN)
        print("restored", MAIN)
        return
    src = open(MAIN).read()
    if "LidarProcess" in src:
        print("already patched:", MAIN)
        return
    if src.count(OLD) != 1:
        sys.exit("could not find the LidarThread line in main.py; nothing changed. Paste: "
                 "grep -n 'LidarThread(' phase3_onboard/main.py")
    shutil.copy2(MAIN, BAK)
    open(MAIN, "w").write(src.replace(OLD, NEW))
    print("patched", MAIN, "(backup", BAK + ")")


if __name__ == "__main__":
    main()
