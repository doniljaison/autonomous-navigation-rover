#!/usr/bin/env python3
"""
One-time patch of phase3_onboard/main.py:
  1. install the obstacle guard (nav/guard.py) on every motion command;
  2. manual turning: let MANUAL_TURN_DUTY (0.45) through.  mix() scales everything down to
     its limit, which was MANUAL_MAX_DUTY (0.40), so a pure spin only ever got 0.40.

    python -m phase3_onboard.tools.apply_guard          # backup: main.py.pre_guard
    python -m phase3_onboard.tools.apply_guard --undo
Turn the guard off for one run without un-patching:  --set GUARD_ENABLED=False
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(HERE, "main.py")
BAK = MAIN + ".pre_guard"

A_OLD = "        self.stuck = StuckMonitor(P)\n"
A_NEW = A_OLD + ("        from phase3_onboard.nav.guard import install_guard\n"
                 "        self.guard = install_guard(self)      # checks every set_intent against the scan\n")
B_RE = re.compile(r"(turn_in \* P\.MANUAL_TURN_DUTY,\s*)P\.MANUAL_MAX_DUTY\)")
B_NEW = r"\1max(P.MANUAL_MAX_DUTY, P.MANUAL_TURN_DUTY))"


def main():
    if "--undo" in sys.argv:
        if not os.path.exists(BAK):
            sys.exit("no backup found: " + BAK)
        shutil.copy2(BAK, MAIN)
        print("restored", MAIN)
        return
    src = open(MAIN).read()
    if "install_guard" in src:
        print("already patched:", MAIN)
        return
    if src.count(A_OLD) != 1 or len(B_RE.findall(src)) != 1:
        sys.exit("main.py does not look as expected; nothing changed. Paste: "
                 "grep -n -e 'StuckMonitor(P)' -e 'mix(' phase3_onboard/main.py")
    out = B_RE.sub(B_NEW, src.replace(A_OLD, A_NEW))
    shutil.copy2(MAIN, BAK)
    open(MAIN, "w").write(out)
    print("patched", MAIN, "(backup", BAK + ")")


if __name__ == "__main__":
    main()
