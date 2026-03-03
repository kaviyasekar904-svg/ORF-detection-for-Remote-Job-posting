import pandas as pd

# 1️⃣ Load original dataset
df = pd.read_csv("data/fake_job_postings.csv")

# 2️⃣ Combine important text columns
df["text"] = (
    df["title"].fillna("") + " " +
    df["company_profile"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["requirements"].fillna("")
)

# 3️⃣ Filter ONLY remote jobs
df_remote = df[
    df["text"].str.contains("remote|work from home|WFH", case=False)
]

# 4️⃣ Keep only needed columns
df_remote = df_remote[["text", "fraudulent"]]

# 5️⃣ Save new dataset
df_remote.to_csv("data/remote_jobs.csv", index=False)

print("Remote jobs dataset created successfully!")
print("Total remote jobs:", len(df_remote))
print(df_remote["fraudulent"].value_counts())
