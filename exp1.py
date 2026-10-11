import pandas as pd

data = pd.read_excel("ml_exp1.xlsx").astype(str).values.tolist()
n = len(data[0]) - 1

# Find-S
S = None
for row in data:
    if row[-1] == "Yes":
        if S is None:
            S = row[:-1].copy()
        else:
            for i in range(n):
                if S[i] != row[i]:
                    S[i] = '?'

print("Find-S Hypothesis:", S)

# Candidate Elimination
G = [['?'] * n]
S = ['0'] * n

for row in data:
    x, label = row[:-1], row[-1]

    if label == "Yes":
        G = [g for g in G if all(g[i] == '?' or g[i] == x[i] for i in range(n))]
        for i in range(n):
            if S[i] == '0':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'
    else:
        new_G = []
        for g in G:
            if all(g[i] == '?' or g[i] == x[i] for i in range(n)):
                for i in range(n):
                    if S[i] != '?' and S[i] != x[i]:
                        h = g.copy()
                        h[i] = S[i]
                        new_G.append(h)
            else:
                new_G.append(g)
        G = new_G

print("\nCandidate Elimination Results:")
print("Most Specific Hypotheses (S):")
print(S)
print("\nMost General Hypotheses (G):")
for g in G:
    print(g)