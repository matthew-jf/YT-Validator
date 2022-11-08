import pandas
import numpy as np
from sklearn import (
    neighbors,
    base
)
import copy

df = pandas.read_csv(r"C:\Users\Matthew.Jurewicz\Downloads\export_all_claims_202210031712.csv",
    dtype=dict(views='Int32', matching_duration='Int32', longest_match='Int32', video_duration_sec='Int32'))
df = df[df.verdict != 'U']
# df = df[df.video_duration_sec != 0]
df.verdict = np.array(df.verdict == 'Y', dtype=int)
df['claim'] = df.claim_origin + df.claim_type
df = df[[
    'views',
    'matching_duration',
    'longest_match',
    'video_duration_sec',
    'verdict',
    'claim'
]]
claim_kind = df.claim.unique()
for s in claim_kind:
    df[s] = np.array(df.claim == s, dtype=int)
df = df.drop(columns='claim')
df = df.fillna(0)
df.to_csv('YT.csv', index=False)

df = pandas.read_csv('YT.csv')
df, y = df.drop(columns='verdict'), df.verdict
soln = neighbors.KNeighborsClassifier(n_neighbors=11, p=1)
for _ in range(4):
    test = np.random.permutation(len(df))
    test = test[:len(df) // 4]
    test = np.array([i in test for i in range(len(df))])

    soln.fit(df[~test], y[~test])
    valid = soln.predict_proba(df[test])
    valid = valid[:,1]
    print(sum((valid > 1/2) == y[test]) / sum(test))
    soln = base.clone(soln)
soln.fit(df, y)

df = pandas.read_csv(r"C:\Users\Matthew.Jurewicz\Downloads\export_unprocessed_claims_202211081020.csv",
    dtype=dict(views='Int32', matching_duration='Int32', longest_match='Int32', video_duration_sec='Int32'))
df2 = copy.copy(df)
df2['claim'] = df2.claim_origin + df2.claim_type
df2 = df2[[
    'views',
    'matching_duration',
    'longest_match',
    'video_duration_sec',
    'claim'
]]
for s in claim_kind:
    df2[s] = np.array(df2.claim == s, dtype=int)
df2 = df2.drop(columns='claim')
df2 = df2.fillna(0)
valid = soln.predict_proba(df2)
valid = valid[:,1]
df['rating'] = valid
df.to_csv('export_unprocessed_claims_202211081020.csv', index=False)