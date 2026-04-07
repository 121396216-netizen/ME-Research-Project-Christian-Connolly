#!/usr/bin/env python3
"""
vlm_to_slm.py — Tails VLM output file and activates person_follow node.

The LFM2-1.2B-Tool SLM is unreliable for binary classification, so person
detection uses keyword matching on the VLM description instead.

Setup:
    # Terminal 1 - SLM server (optional, kept for future use):
    bash /root/start_slm.sh

    # Terminal 2 - redirect VLM output to file:
    /usr/bin/python3 -u /home/camera_vlm.py > /tmp/vlm_output.txt

    # Terminal 3 - run this script:
    python3 -u /root/vlm_to_slm.py
"""

import re
import subprocess
import time

VLM_LOG = '/tmp/vlm_output.txt'

PERSON_KEYWORDS = [
    'person', 'people', 'human', 'man', 'woman', 'boy', 'girl',
    'individual', 'figure', 'someone', 'standing', 'walking', 'sitting',
]

LINE_RE = re.compile(r'^\[\d{2}:\d{2}:\d{2}\]\s+\(\S+\)\s+(.+)$')

_person_follow_proc = None


def _person_in_scene(description):
    desc = description.lower()
    # Negative phrases take priority
    negations = ['no person', 'no people', 'no human', 'nobody', 'no one',
                 'not a person', 'without a person', 'empty', 'no visible person']
    for neg in negations:
        if neg in desc:
            return False
    return any(kw in desc for kw in PERSON_KEYWORDS)


def _execute_action(action):
    global _person_follow_proc
    if action == 'follow':
        if _person_follow_proc is None or _person_follow_proc.poll() is not None:
            print('  → starting person_follow node', flush=True)
            _person_follow_proc = subprocess.Popen([
                'ros2', 'run', 'autonomous_driving_pkg', 'person_follow',
            ])
        else:
            print('  → person_follow already running', flush=True)
    else:
        if _person_follow_proc and _person_follow_proc.poll() is None:
            print('  → stopping person_follow node', flush=True)
            _person_follow_proc.terminate()
            _person_follow_proc = None
        else:
            print('  → already stopped', flush=True)


print(f'Tailing {VLM_LOG} — waiting for VLM output...', flush=True)

while not __import__('os').path.exists(VLM_LOG):
    time.sleep(0.5)

with open(VLM_LOG, 'r') as f:
    f.seek(0, 2)  # seek to end — only process new lines
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.1)
            continue
        line = line.rstrip('\n')
        m = LINE_RE.match(line)
        if not m:
            continue
        description = m.group(1)
        action = 'follow' if _person_in_scene(description) else 'stop'
        print(f'VLM: {description}', flush=True)
        print(f'Decision: {action}', flush=True)
        _execute_action(action)
        print(flush=True)
