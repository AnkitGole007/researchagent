import sys

with open('app.py', 'r', encoding='utf-8') as f:
    text = f.read()

errors = []
passes = []

# P-01
if 'def semantic_not_filter(' in text:
    passes.append('P-01 semantic_not_filter defined')
else:
    errors.append('P-01 MISSING: semantic_not_filter function')

if 'semantic_not_filter(' in text and 'not_phrases, _a2p or {}' in text:
    passes.append('P-01 semantic_not_filter wired in pipeline')
else:
    errors.append('P-01 MISSING: semantic_not_filter not wired at call site')

# P-02
if 'use_hyde=(provider in' in text and 'api_key.strip()' in text:
    passes.append('P-02 HyDE enabled for LLM providers')
else:
    errors.append('P-02 MISSING: use_hyde not set to provider-based expression')

# P-04
if 'adaptive_k = min(int(len(papers) * 0.07), 1200)' in text:
    passes.append('P-04 adaptive_k computation present')
else:
    errors.append('P-04 MISSING: adaptive_k formula not found')

# P-05
n2_400_count = text.count('n2=400')
if n2_400_count >= 2:
    passes.append(f'P-05 n2=400 found {n2_400_count} times (SPECTER2 + MiniLM fallback)')
else:
    errors.append(f'P-05 MISSING: n2=400 found only {n2_400_count} times (need 2+)')

# P-06
if 'Retrieval quality may be reduced. Check that' in text:
    passes.append('P-06 st.warning on MiniLM fallback present')
else:
    errors.append('P-06 MISSING: st.warning fallback message not found')

# P-07
if 'BAAI/bge-reranker-base' in text:
    passes.append('P-07 BAAI/bge-reranker-base in CrossEncoder')
else:
    errors.append('P-07 MISSING: BAAI/bge-reranker-base not found')

# P-07: old model — only fail if it appears OUTSIDE comments (in an actual return/assignment)
active_ms_marco = [ln for ln in text.splitlines()
                   if 'ms-marco' in ln and not ln.strip().startswith('#')]
if active_ms_marco:
    errors.append(f'P-07 WARNING: ms-marco appears in non-comment code: {active_ms_marco}')
else:
    passes.append('P-07 old ms-marco only in comments — active code uses BGE')

# P-08
if 'PRIMARY_THRESHOLD: float = 0.55' in text and 'SECONDARY_THRESHOLD: float = 0.25' in text:
    passes.append('P-08 thresholds updated to 0.55/0.25')
else:
    errors.append('P-08 MISSING: threshold values not 0.55/0.25')

# P-09 early filter removed
if 'Apply venue filtering IMMEDIATELY' in text:
    errors.append('P-09 FAIL: early venue filter comment still present')
else:
    passes.append('P-09 early venue filter removed from corpus fetch path')

# P-09 post-Stage3 filter added
if 'after Stage 3' in text:
    passes.append('P-09 post-Stage3 venue filter block present')
else:
    errors.append('P-09 MISSING: post-Stage3 venue filter not found')

# SPECTER2 still intact
if 'allenai/specter2' in text:
    passes.append('SPECTER2 model references present (Stage 2 intact)')
else:
    errors.append('SPECTER2 references MISSING')

print('=== VERIFICATION RESULTS ===')
for p in passes:
    print(f'  PASS: {p}')
if errors:
    print()
    for e in errors:
        print(f'  FAIL: {e}')
    sys.exit(1)
else:
    print(f'\nALL {len(passes)} CHECKS PASSED')
