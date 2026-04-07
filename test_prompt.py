#!/usr/bin/env python3
"""
test_prompt.py — Interactive SLM prompt tester.

Tests the scene → behavior decision without running the full pipeline.
Edit SYSTEM_PROMPT and USER_PROMPT_TMPL below, then run and type scene descriptions.

Usage:
    python3 /root/test_prompt.py
    python3 /root/test_prompt.py --scene "A man is walking toward the camera"
"""

import argparse
import requests

SLM_URL = 'http://localhost:8081/v1/chat/completions'

# ── Edit these to test different prompts ──────────────────────────────────────

USER_PROMPT_TMPL = """\
Scene: {scene}

Choose the robot behavior. Check in this order:
1. P = person_follow: HIGHEST PRIORITY. Any person, man, woman, child visible? Choose P immediately.
2. F = follow_the_gap: ONLY if walls or obstacles are close and filling the view. Do NOT choose F just because the room is open.
3. S = stop: DEFAULT. Choose S if no person and no close obstacles.

Think briefly, then on a new line write only the letter (P, F, or S):"""

# ─────────────────────────────────────────────────────────────────────────────

PERSON_KEYWORDS = {
    'person', 'people', 'human', 'man', 'woman', 'boy', 'girl',
    'child', 'individual', 'someone', 'figure', 'standing',
    'walking', 'sitting',
}
VALID_TOOLS = {'follow_the_gap', 'person_follow', 'stop'}


def query(scene: str):
    prompt = USER_PROMPT_TMPL.format(scene=scene)
    print(f'\n--- Prompt sent ---\n{prompt}\n---')
    try:
        resp = requests.post(
            SLM_URL,
            json={
                'messages': [
                    {'role': 'system', 'content': 'You select robot driving behaviors. Reply with a single letter only.'},
                    {'role': 'user', 'content': prompt},
                ],
                'max_tokens': 1,
                'temperature': 0,
                'grammar': 'root ::= [FPS]',
            },
            timeout=60,
        )
        resp.raise_for_status()
        body = resp.json()
        raw = body['choices'][0]['message']['content'].strip()
        print(f'Raw response: "{raw}"')
        t = body.get('timings', {})
        if t:
            print(f'Prefill:  {t.get("prompt_ms",0):.0f}ms  '
                  f'{t.get("prompt_n",0):.0f} tok @ {t.get("prompt_per_second",0):.0f} t/s')
            print(f'Generate: {t.get("predicted_ms",0):.0f}ms  '
                  f'{t.get("predicted_n",0):.0f} tok @ {t.get("predicted_per_second",0):.0f} t/s')

        first = raw.upper()[:1]
        if first == 'P':
            tool = 'person_follow'
        elif first == 'F':
            tool = 'follow_the_gap'
        else:
            tool = 'stop'
        print(f'Decision: {tool}')
    except Exception as e:
        print(f'Error: {e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', help='Single scene to test (non-interactive)')
    args = parser.parse_args()

    # Check server
    try:
        requests.get('http://localhost:8081/health', timeout=3)
    except Exception:
        print('ERROR: llama-server not reachable at localhost:8081')
        return

    if args.scene:
        query(args.scene)
        return

    print('Interactive mode — paste a scene (end with a blank line), Ctrl-C to quit.\n')
    while True:
        try:
            lines = []
            print('Scene> ', end='', flush=True)
            while True:
                line = input()
                if line.strip() == '':
                    break
                lines.append(line.strip())
        except (KeyboardInterrupt, EOFError):
            break
        scene = ' '.join(lines)
        if scene:
            query(scene)


if __name__ == '__main__':
    main()
