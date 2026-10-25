"""Run a script once per dataset (optionally once per dataset x method), each in a fresh Python process.

    python run_each.py train.py --dataset sweep --runs 1 --quiet --resume --tag v2 --out results_sweep
    python run_each.py pyagc_run.py --dataset all --runs 5 --split-methods

* Memory is released between jobs, and a crash only affects that one job: the loop carries on.
* Every job's full output is saved in run_logs/<script>/<dataset>[__<method>].txt.
* At the end, failed jobs are listed (also saved to run_logs/<script>/FAILED.txt) with their last lines.
* --job-timeout MIN (default 60): a job still running after MIN minutes is stopped and written to
  run_logs/<script>/TIMEOUT.txt (seeds already finished stay saved). On Windows a GPU that runs out of memory
  can silently spill into shared system memory and become extremely slow instead of failing, so this cap is
  the run's resource budget. Later reruns skip jobs listed in TIMEOUT.txt; add --retry-timeouts to try again.
"""
import os
import re
import subprocess
import sys

from data import expand_names

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script, args = sys.argv[1], sys.argv[2:]
    split = "--split-methods" in args
    args = [x for x in args if x != "--split-methods"]
    retry = "--retry-timeouts" in args
    args = [x for x in args if x != "--retry-timeouts"]
    timeout_min = 60.0
    if "--job-timeout" in args:
        t = args.index("--job-timeout")
        timeout_min = float(args[t + 1])
        args = args[:t] + args[t + 2:]
    if "--dataset" not in args:
        raise SystemExit("give --dataset")
    i = args.index("--dataset")
    names = expand_names(args[i + 1])
    rest = args[:i] + args[i + 2:]

    methods = [None]
    if split:
        if "--methods" in rest:
            j = rest.index("--methods")
            methods = rest[j + 1].split(",")
            rest = rest[:j] + rest[j + 2:]
        else:  # read the default list from the script without importing it
            src = open(os.path.join(HERE, script), encoding="utf-8").read()
            methods = re.findall(r'"([^"]+)"', re.search(r"^METHODS = \[(.*?)\]", src, re.M).group(1))

    logdir = os.path.join(HERE, "run_logs", os.path.splitext(os.path.basename(script))[0])
    os.makedirs(logdir, exist_ok=True)
    jobs = [(n, m) for n in names for m in methods]
    failed, stopped = [], []
    tfile = os.path.join(logdir, "TIMEOUT.txt")
    # a time-out belongs to one configuration: key = job + "@" + the --config / --tag value (if any)
    cfg = [rest[rest.index(o) + 1] for o in ("--config", "--tag") if o in rest]
    suffix = "@" + "+".join(cfg) if cfg else ""
    timed_out = set() if retry or not os.path.exists(tfile) else \
        {ln.split()[0] for ln in open(tfile, encoding="utf-8") if ln.strip()}
    for k, (name, method) in enumerate(jobs, 1):
        tag = name + (f"__{method}" if method else "")
        cmd = [sys.executable, os.path.join(HERE, script)] + rest + ["--dataset", name]
        if method:
            cmd += ["--methods", method]
        log = os.path.join(logdir, tag + ".txt")
        print(f"[{k}/{len(jobs)}] {tag}", flush=True)
        key = tag + suffix
        if key in timed_out:
            print(f"   skipped: exceeded the {timeout_min:g}-minute budget before (see TIMEOUT.txt)", flush=True)
            stopped.append(tag)
            continue
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")  # Windows consoles choke on symbols like ✓
        with open(log, "w", encoding="utf-8", errors="replace") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=HERE, env=env)
            try:
                code = proc.wait(timeout=timeout_min * 60 if timeout_min > 0 else None)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
                else:
                    proc.kill()
                proc.wait()
                code = "timeout"
                with open(tfile, "a", encoding="utf-8") as tf:
                    tf.write(f"{key}  stopped after {timeout_min:g} min\n")
        lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
        for ln in lines:  # echo only the useful lines
            if re.search(r"seed \d+:|==>|skipped|FAILED|Error|error:", ln) and not re.search(r"Warning|Unicode|Logging error|charmap", ln):
                print("   " + ln.strip()[:200], flush=True)
        if code == "timeout":
            stopped.append(tag)
            print(f"   !! stopped: still running after {timeout_min:g} minutes (resource budget); seeds already finished are kept", flush=True)
        elif code != 0:
            tail = [ln for ln in lines if ln.strip()][-3:]
            failed.append((tag, code, tail))
            print(f"   !! crashed (exit code {code}); full log: {log}", flush=True)

    with open(os.path.join(logdir, "FAILED.txt"), "w", encoding="utf-8") as f:
        for tag, code, tail in failed:
            f.write(f"{tag}  exit code {code}\n" + "".join(f"    {t}\n" for t in tail))
    print(f"\nfinished: {len(jobs) - len(failed) - len(stopped)} of {len(jobs)} jobs ran to the end")
    if stopped:
        print(f"{len(stopped)} over the time budget (listed in {tfile}): " + ", ".join(stopped))
    if failed:
        print(f"{len(failed)} crashed (listed in {os.path.join(logdir, 'FAILED.txt')}):")
        for tag, code, tail in failed:
            print(f"  {tag}  (exit {code})  {tail[-1][:150] if tail else ''}")
        print("Run the same command again to retry them; finished runs are skipped.")


if __name__ == "__main__":
    main()
