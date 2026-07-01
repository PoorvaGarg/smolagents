# Agent Comparison Results

## Simple Tasks

Tasks: Tokyo population, FIFA World Cup winner, latest Python version.

| Task | Agent | Steps | Tokens |
|------|-------|------:|-------:|
| What is the current population of Tokyo? | CodeAgent | 3 | 8,358 |
| What is the current population of Tokyo? | ProbAgent(n=3) | 2 | 34,880 |
| Who won the most recent FIFA World Cup? | CodeAgent | 3 | 7,751 |
| Who won the most recent FIFA World Cup? | ProbAgent(n=3) | 2 | 35,610 |
| What is the latest version of Python? | CodeAgent | 2 | 5,062 |
| What is the latest version of Python? | ProbAgent(n=3) | 2 | 39,726 |

**Observation:** ProbAgent uses ~4-5x more tokens for roughly the same or one fewer step. For simple tasks the diversity sampling overhead is not worth it.

---

## GAIA Tasks (first 10)

| Q | Task (truncated) | Agent | Steps | Tokens | Correct | Answer |
|---|------------------|-------|------:|-------:|:-------:|--------|
| Q1 | arXiv AI regulation paper (axis label) | CodeAgent | 6 | 40,900 | ✅ | Egalitarian |
| Q1 | | PruningAgent | 7 | 42,755 | ✅ | egalitarian |
| Q1 | | ProbAgent(n=3) | 5 | 230,917 | ✅ | Egalitarian |
| Q2 | USGS invasive species zip codes | CodeAgent | 11 | 100,642 | ✅ | 34689 |
| Q2 | | PruningAgent | 12 | 70,004 | ✅ | 34689 |
| Q2 | | ProbAgent(n=3) | 17 | 868,290 | ✅ | 34689 |
| Q3 | Nature 2020 p-value article count | CodeAgent | 2 | 6,485 | ❌ | 5% of the articles assuming a p-value of 0.04 would be incor… |
| Q3 | | PruningAgent | 14 | 126,964 | ❌ | 100 |
| Q3 | | ProbAgent(n=3) | 6 | 218,995 | ❌ | 1921 |
| Q4 | Unlambda character correction | CodeAgent | 5 | 25,304 | ❌ | Some text (`r` or a newline character could be added) |
| Q4 | | PruningAgent | 1 | 3,275 | ❌ | ` |
| Q4 | | ProbAgent(n=3) | 1 | 19,721 | ❌ | v |
| Q5 | Kipchoge marathon pace distance | CodeAgent | 6 | 30,401 | ✅ | 17 |
| Q5 | | PruningAgent | 8 | 44,391 | ✅ | 17 |
| Q5 | | ProbAgent(n=3) | 5 | 150,448 | ❌ | 17000 |
| Q6 | Oldest Blu-Ray in spreadsheet | CodeAgent | 5 | 17,637 | ❌ | The Lion King |
| Q6 | | PruningAgent | 51 | 960,424 | ❌ | (hit step limit, never resolved file-reading issue) |
| Q6 | | ProbAgent(n=3) | 33 | 1,731,763 | ❌ | Oldest Blu-Ray Title |
| Q7 | Mercedes Sosa studio albums 2000–2009 | CodeAgent | 11 | 78,590 | ❌ | 2 |
| Q7 | | PruningAgent | 20 | 240,820 | ✅ | 3 |
| Q7 | | ProbAgent(n=3) | 6 | 225,833 | ❌ | 2 |
| Q8 | British Museum object 2012,5015 | CodeAgent | 8 | 50,732 | ❌ | at least 10 thousand years old |
| Q8 | | PruningAgent | 3 | 11,467 | ✅ | 142 |
| Q8 | | ProbAgent(n=3) | 3 | 76,580 | ❌ | 142 thousand years or more |
| Q9 | Oldest closed numpy.polynomial GitHub issue | CodeAgent | 51 | 1,806,852 | ❌ | (hit step limit, never resolved) |
| Q9 | | PruningAgent | 51 | 1,497,230 | ❌ | (hit step limit, never resolved) |
| Q9 | | ProbAgent(n=3) | 9 | 519,007 | ❌ | Please manually visit the NumPy GitHub issues page to filter… |
| Q10 | Riddle | CodeAgent | 1 | 3,340 | ❌ | 1 |
| Q10 | | PruningAgent | 5 | 25,728 | ❌ | 100 |
| Q10 | | ProbAgent(n=3) | 1 | 22,346 | ❌ | 1 |
| **Total** | | **CodeAgent** | **106** | **2,160,883** | **3/10** | |
| **Total** | | **PruningAgent** | **172** | **3,023,058** | **5/10** | |
| **Total** | | **ProbAgent(n=3)** | **86** | **4,063,900** | **2/10** | |

**Observations:**
- PruningAgent wins on accuracy (5/10 vs 3/10 for CodeAgent, 2/10 for ProbAgent), picking up Q7 and Q8 that CodeAgent got wrong — but at the highest total token cost (~3.0M) of the two CodeAgent-based runs, and the most total steps (172).
- CodeAgent is the cheapest of the three (~2.16M tokens, 106 steps) and stays competitive on accuracy.
- ProbAgent(n=3) takes the fewest steps overall (86) since its diversity sampling often converges faster, but each step costs ~3 model calls, making it by far the most expensive (~4.06M tokens) for the worst accuracy (2/10).
- Q6 and Q9 are the hardest tasks in this set: all three agents hit the 50-step limit or otherwise failed to resolve them (Q6 needs `pandas`/spreadsheet parsing; Q9 needs GitHub API access that appears blocked).
- Q7 (Mercedes Sosa) is the clearest case where pruning helped: CodeAgent and ProbAgent both answered 2 (wrong, expected 3), while PruningAgent's shorter, decluttered context led it to the correct answer in 20 steps.
